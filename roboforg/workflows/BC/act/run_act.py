"""ACT 仿真执行入口：加载 checkpoint，让策略在 Pick 环境自主执行。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

import jax
import numpy as np


# Pick 配置沿用 platform 作为顶层包；模块运行时补入该目录，使 run_act 不依赖当前终端路径。
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLATFORM_ROOT = PROJECT_ROOT / "platform"
if str(PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(PLATFORM_ROOT))

from roboforg.agent.act import ActionEnsemble
from roboforg.workflows.act.common import (
    HEADLESS_NUM_ROLLOUTS,
    VIEWER_NUM_ROLLOUTS,
    create_act_agent,
    load_checkpoint_into_agent,
    read_checkpoint_setup,
)
from envs.wrappers.action_utils import project_action_to_unit_balls


# 将环境返回的单帧观测增加 batch 维，并检查 state、相机、动作空间与 checkpoint 的结构一致。
def batch_observation(
    observation: dict[str, Any], *, camera_keys: tuple[str, ...]
) -> dict[str, np.ndarray]:
    """Convert one environment observation into the batched [1,...] form required by ACTAgent."""
    if "state" not in observation:
        raise KeyError("Environment observation is missing 'state'.")
    state = np.asarray(observation["state"], dtype=np.float32)
    if state.ndim != 1 or state.size == 0 or not np.all(np.isfinite(state)):
        raise ValueError("Environment state must be a finite non-empty one-dimensional array.")

    batched: dict[str, np.ndarray] = {"state": state[None]}
    for key in camera_keys:
        if key not in observation:
            raise KeyError(f"Environment observation is missing camera {key!r}.")
        image = np.asarray(observation[key])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"Camera {key!r} must be an RGB uint8 image, got {image.shape}, {image.dtype}.")
        batched[key] = image[None]
    return batched


# 创建无人工干预的 Pick 仿真环境；是否显示 MuJoCo 窗口由运行参数决定。
def make_evaluation_environment(*, show_viewer: bool):
    """Build the visual Pick environment used for autonomous ACT rollouts."""
    # 必须在导入 MuJoCo 前选择后端；服务器离屏图像仍需要渲染。
    if not show_viewer:
        os.environ.setdefault("MUJOCO_GL", "egl")
    from examples.sim_pick.config import TrainConfig

    env_config = TrainConfig()
    env_config.show_viewer = show_viewer
    env_config.input_device = None
    return env_config.get_environment(fake_env=False, save_video=False, classifier=False)


# 执行一整回合：每 query_every 步预测一次动作 chunk，每个环境步融合并执行一个七维动作。
def run_rollout(
    *,
    env: Any,
    agent: Any,
    ensemble: ActionEnsemble,
    seed: int,
    max_steps: int,
    query_every: int,
) -> dict[str, float | bool | int]:
    """Run one autonomous episode and return success, length, return, and clipping statistics."""
    observation, _ = env.reset(seed=seed)
    if tuple(env.action_space.shape) != (agent.policy.action_dim,):
        raise ValueError(
            f"Environment action shape {env.action_space.shape} does not match "
            f"checkpoint action_dim={agent.policy.action_dim}."
        )
    ensemble.reset()
    total_reward = 0.0
    clipped_steps = 0

    for step in range(max_steps):
        # 默认每步查询；若降低查询频率，缓存 chunk 仍为中间时刻提供动作建议。
        if step % query_every == 0:
            prediction = np.asarray(agent.predict(batch_observation(
                observation, camera_keys=agent.policy.camera_keys
            )))
            action_chunk = prediction[0]
            if action_chunk.shape != (agent.policy.action_horizon, agent.policy.action_dim):
                raise RuntimeError(f"ACT returned unexpected action chunk shape {action_chunk.shape}.")
            ensemble.add_prediction(step, action_chunk)

        action = ensemble.get_action(step)
        # 先剪裁每维范围，再把平移/旋转各自投影到单位 L2 球，满足 RelativeFrameWrapper 的动作约束。
        clipped_action = np.clip(action, env.action_space.low, env.action_space.high)
        safe_action = project_action_to_unit_balls(clipped_action)
        if not np.array_equal(safe_action, action):
            clipped_steps += 1
        observation, reward, terminated, truncated, info = env.step(safe_action)
        total_reward += float(reward)
        if terminated or truncated:
            return {
                "success": bool(info.get("is_success", False)),
                "steps": step + 1,
                "return": total_reward,
                "clipped_steps": clipped_steps,
            }

    # 若用户给出的 max_steps 比环境终止条件更短，仍明确报告这一回合未成功。
    return {
        "success": False,
        "steps": max_steps,
        "return": total_reward,
        "clipped_steps": clipped_steps,
    }


def evaluate_policy(
    agent, *, num_rollouts: int | None = None, show_viewer: bool = False,
    seed: int = 0, max_steps: int | None = None, query_every: int = 1, env=None,
) -> dict[str, float | int] | None:
    """顺序执行完整回合；离屏返回指标，开屏仅观察。调用者传入的环境由调用者关闭。"""
    if num_rollouts is None:
        num_rollouts = VIEWER_NUM_ROLLOUTS if show_viewer else HEADLESS_NUM_ROLLOUTS
    if num_rollouts <= 0 or not 1 <= query_every <= agent.policy.action_horizon:
        raise ValueError("num_rollouts must be positive; query_every must be within action_horizon.")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    owns_env = env is None
    if owns_env:
        env = make_evaluation_environment(show_viewer=show_viewer)
    try:
        # 默认沿用任务的超时限制，避免在任务结束前人为截短回合。
        episode_limit = env.unwrapped.config.max_episode_length if max_steps is None else max_steps
        ensemble = ActionEnsemble(agent.policy.action_horizon, action_dim=agent.policy.action_dim, decay=0.01)
        results = []
        for index in range(num_rollouts):
            result = run_rollout(
                env=env, agent=agent, ensemble=ensemble, seed=seed + index,
                max_steps=episode_limit, query_every=query_every,
            )
            if show_viewer:
                print(f"Observation {index + 1}/{num_rollouts}: steps={result['steps']}", flush=True)
            else:
                results.append(result)
                print(f"Rollout {index + 1}/{num_rollouts}: success={result['success']}, steps={result['steps']}", flush=True)
        if show_viewer:
            return None
        successes = sum(int(result["success"]) for result in results)
        metrics = {
            "success_rate": successes / num_rollouts,
            "successes": successes,
            "num_rollouts": num_rollouts,
            "mean_steps": float(np.mean([result["steps"] for result in results])),
            "mean_return": float(np.mean([result["return"] for result in results])),
        }
        print(f"ACT rollout summary: {metrics}", flush=True)
        return metrics
    finally:
        if owns_env:
            env.close()


# 解析 checkpoint、运行回合数、可视化与动作查询频率等部署参数。
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for autonomous ACT Pick rollouts."""
    parser = argparse.ArgumentParser(description="Run a RoboForge ACT checkpoint in the Pick simulator.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-rollouts", type=int, help="Episode count; defaults to 40 headless or 10 with viewer.")
    parser.add_argument("--max-steps", type=int, help="Optional override of the environment episode limit.")
    parser.add_argument("--query-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show-viewer", action="store_true")
    args = parser.parse_args()
    if args.num_rollouts is None:
        args.num_rollouts = VIEWER_NUM_ROLLOUTS if args.show_viewer else HEADLESS_NUM_ROLLOUTS
    if args.num_rollouts <= 0 or (args.max_steps is not None and args.max_steps <= 0) or args.query_every <= 0:
        parser.error("--num-rollouts, --max-steps, and --query-every must be positive.")
    return args


# 从 checkpoint 创建同形 Agent，逐回合运行并输出真正的 Pick 成功率。
def main() -> None:
    """Load one ACT checkpoint and measure autonomous Pick rollout success."""
    args = parse_args()
    config, normalizer, metadata = read_checkpoint_setup(args.checkpoint)
    if args.query_every > config.action_horizon:
        raise ValueError("query_every cannot exceed action_horizon; the cached chunk would run out first.")

    env = make_evaluation_environment(show_viewer=args.show_viewer)
    try:
        # 用 reset 后的真实观测初始化同形参数树；零动作仅用于初始化训练期 CVAE，不会被执行。
        initial_observation, _ = env.reset(seed=args.seed)
        example_observations = batch_observation(initial_observation, camera_keys=config.camera_keys)
        example_batch = (
            example_observations,
            np.zeros((1, config.action_horizon, config.action_dim), dtype=np.float32),
        )
        rng = jax.random.key(args.seed)
        agent = create_act_agent(
            config,
            normalizer=normalizer,
            example_batch=example_batch,
            rng=rng,
            load_pretrained_backbone=False,
        )
        agent, _ = load_checkpoint_into_agent(args.checkpoint, agent)
        print(f"Loaded {args.checkpoint}, completed_epochs={metadata['epoch'] + 1}.", flush=True)
        evaluate_policy(
            agent, env=env, num_rollouts=args.num_rollouts, show_viewer=args.show_viewer,
            seed=args.seed, max_steps=args.max_steps, query_every=args.query_every,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
