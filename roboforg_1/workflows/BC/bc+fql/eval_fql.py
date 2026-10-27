"""FQL 统一仿真评估, 供训练期间直接调用和独立 checkpoint 评估复用."""
from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Mapping
from importlib import import_module
import os
from pathlib import Path
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from roboforg_1.agent.algorithm.fql import FQLAGENT
from roboforg_1.data.observation import canonicalize_observation
from roboforg_1.data.utils import stack_tree

# 当前目录含 '+', 使用动态导入兼容现有路径, 不改变目录名称.
_common = import_module(f"{__package__}.common")
FQLWorkflowConfig = _common.FQLWorkflowConfig
CHECKPOINT_ROOT = _common.CHECKPOINT_ROOT
PROJECT_ROOT = _common.PROJECT_ROOT


"""------------------------- 环境与观测准备 -------------------------"""

def _make_environment(config: FQLWorkflowConfig, *, show_viewer: bool):
    """按任务映射创建仿真环境, 关闭人工干预, 视频和分类器."""
    platform_root = str(PROJECT_ROOT / "platform")
    if platform_root not in sys.path:
        sys.path.insert(0, platform_root)
    if show_viewer:
        os.environ.pop("MUJOCO_GL", None)
    else:
        os.environ.setdefault("MUJOCO_GL", "egl")

    # 渲染后端必须在首次导入 MuJoCo 前设置.
    from examples.mappings import CONFIG_MAPPING
    if config.task_name not in CONFIG_MAPPING:
        raise ValueError(f"Unknown task {config.task_name!r}; available: {sorted(CONFIG_MAPPING)}")
    task_config = CONFIG_MAPPING[config.task_name]()
    task_config.show_viewer = show_viewer
    task_config.input_device = None
    return task_config.get_environment(fake_env=False, save_video=False, classifier=False)


def _prepare_observation(observation: Mapping[str, Any], config: FQLWorkflowConfig) -> dict[str, Any]:
    """规范化环境单帧观测, 选取相机并保留状态和任务条件."""
    observation = canonicalize_observation(observation, camera_keys=config.camera_keys)
    images = {}
    for key in config.camera_keys:
        if key not in observation["images"]:
            raise KeyError(f"Environment observation is missing camera {key!r}.")
        image = np.asarray(observation["images"][key])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"Camera {key!r} must provide a single RGB uint8 image.")
        images[key] = image.copy()
    state = np.asarray(observation["state"], dtype=np.float32)
    if state.ndim != 1 or not np.all(np.isfinite(state)):
        raise ValueError("Environment state must be a finite vector.")
    result = {"images": images, "state": state.copy()}
    if "task_condition" in observation:
        result["task_condition"] = jax.tree_util.tree_map(lambda value: np.array(value, copy=True), observation["task_condition"])
    return result


def _policy_observation(history: deque, config: FQLWorkflowConfig) -> dict[str, Any]:
    """构造无 batch 维的历史观测, 非历史模式只使用最新一帧."""
    observation = stack_tree(list(history)) if config.use_temporal_history else history[-1]
    return jax.tree_util.tree_map(jnp.asarray, observation)


"""------------------------- 单回合与统一评估 -------------------------"""

def _run_episode(agent: FQLAGENT, config: FQLWorkflowConfig, env: Any, *, seed: int, rng: jax.Array, print_policy_output: bool, max_episode_steps: int | None) -> tuple[dict[str, Any], jax.Array]:
    """执行完整动作块, 逐步更新历史, 在终止或截断时立即结束回合."""
    observation, _ = env.reset(seed=seed)
    first = _prepare_observation(observation, config)
    history = deque([first] * config.observation_horizon, maxlen=config.observation_horizon)
    total_return, steps, clipped_steps, success = 0.0, 0, 0, False
    action_dim = agent.config["action_dim"]
    if env.action_space.shape != (action_dim,):
        raise ValueError(f"Environment action shape must be {(action_dim,)}, got {env.action_space.shape}.")

    while max_episode_steps is None or steps < max_episode_steps:
        rng, sample_rng = jax.random.split(rng)
        flat_actions = np.asarray(jax.device_get(agent.sample_actions(_policy_observation(history, config), sample_rng)))
        if flat_actions.shape != (config.action_horizon * action_dim,) or not np.all(np.isfinite(flat_actions)):
            raise ValueError("Policy must return a finite flattened action chunk.")
        action_chunk = flat_actions.reshape(config.action_horizon, action_dim)

        # sample_actions 已做单位球投影, 此处只适配环境 Box 范围和 dtype.
        for action in action_chunk:
            safe_action = np.clip(action, env.action_space.low, env.action_space.high).astype(env.action_space.dtype)
            clipped_steps += int(not np.array_equal(action, safe_action))
            if print_policy_output:
                print(f"Policy output step={steps + 1}: {np.array2string(safe_action, precision=6, separator=', ')}", flush=True)
            observation, reward, terminated, truncated, info = env.step(safe_action)
            steps += 1
            total_return += float(reward)
            success = success or bool(info.get("is_success", False))
            history.append(_prepare_observation(observation, config))
            if terminated or truncated or (max_episode_steps is not None and steps >= max_episode_steps):
                return {"success": success, "return": total_return, "steps": steps, "clipped_steps": clipped_steps}, rng
    return {"success": success, "return": total_return, "steps": steps, "clipped_steps": clipped_steps}, rng


def evaluate_policy(agent: FQLAGENT, config: FQLWorkflowConfig, *, num_episodes: int = 40, seed: int = 0, env: Any = None, show_viewer: bool = False, print_policy_output: bool = False, max_episode_steps: int | None = None, verbose: bool = True) -> dict[str, float]:
    """评估 Agent, 不更新参数或训练 RNG, 仅关闭自行创建的环境.

    输入: Agent、对应配置及评估选项; verbose=False 关闭回合、汇总和动作打印.
    输出: 成功率、平均回报、平均步数及平均裁剪步数的标量字典.
    """
    if not isinstance(num_episodes, int) or isinstance(num_episodes, bool) or num_episodes <= 0:
        raise ValueError("num_episodes must be a positive integer.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    if max_episode_steps is not None and (not isinstance(max_episode_steps, int) or isinstance(max_episode_steps, bool) or max_episode_steps <= 0):
        raise ValueError("max_episode_steps must be a positive integer or None.")
    for name in ("observation_horizon", "action_horizon", "use_temporal_history"):
        if agent.config[name] != getattr(config, name):
            raise ValueError(f"Evaluation config does not match Agent: {name}.")
    if tuple(agent.config["image_keys"]) != config.camera_keys:
        raise ValueError("Evaluation camera_keys must match the Agent cameras and order.")

    owns_env = env is None
    if owns_env:
        env = _make_environment(config, show_viewer=show_viewer)
    episodes, rng = [], jax.random.PRNGKey(seed)
    try:
        for index in range(num_episodes):
            result, rng = _run_episode(agent, config, env, seed=seed + index, rng=rng, print_policy_output=print_policy_output and verbose, max_episode_steps=max_episode_steps)
            episodes.append(result)
            if verbose:
                print(f"测评 {index + 1}/{num_episodes} | 成功: {'true' if result['success'] else 'false'} | 奖励: {result['return']:g}", flush=True)
        if verbose:
            successes = sum(item["success"] for item in episodes)
            print(f"测评成功率: {successes / len(episodes):.1%} ({successes}/{len(episodes)})", flush=True)
        return {
            "success_rate": float(np.mean([item["success"] for item in episodes])),
            "mean_return": float(np.mean([item["return"] for item in episodes])),
            "mean_steps": float(np.mean([item["steps"] for item in episodes])),
            "mean_clipped_steps": float(np.mean([item["clipped_steps"] for item in episodes])),
        }
    finally:
        if owns_env:
            env.close()


"""------------------------- 独立评估入口 -------------------------"""

def parse_args() -> argparse.Namespace:
    """读取模型来源和评估选项, 未指定模型时查找对应任务的最新实验."""
    parser = argparse.ArgumentParser(description="Evaluate a FQL checkpoint.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=Path, default=None)
    source.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--task-name", default=None, help="Task for latest-run lookup; defaults to pick. Explicit checkpoint uses its saved task.")
    parser.add_argument("--num-episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show-viewer", choices=("true", "false"), default="false")
    parser.add_argument("--print-policy-output", choices=("true", "false"), default="false")
    parser.add_argument("--max-episode-steps", type=int, default=None, help="Optional evaluation cutoff; otherwise use environment termination/truncation.")
    args = parser.parse_args()
    if args.num_episodes <= 0 or args.seed < 0 or (args.max_episode_steps is not None and args.max_episode_steps <= 0):
        parser.error("num-episodes and max-episode-steps must be positive; seed must be non-negative.")
    return args


def main() -> None:
    """定位并恢复 checkpoint, 调用训练期间共用的评估函数."""
    args = parse_args()
    checkpoint = _common.resolve_checkpoint(checkpoint=args.checkpoint, run_dir=args.run_dir, checkpoint_root=args.checkpoint_root, task_name=args.task_name or "pick")
    agent, config, _ = _common.load_checkpoint(checkpoint)
    if args.task_name is not None and args.task_name != config.task_name:
        raise ValueError("Requested task does not match the checkpoint task.")
    evaluate_policy(agent, config, num_episodes=args.num_episodes, seed=args.seed, show_viewer=args.show_viewer == "true", print_policy_output=args.print_policy_output == "true", max_episode_steps=args.max_episode_steps)


if __name__ == "__main__":
    main()
