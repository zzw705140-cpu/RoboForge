# ACT 仿真测评：离屏统计成功率，或在本地开屏观察指定 checkpoint 的动作。
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

import jax
import numpy as np

from roboforg.agent.act import ActionEnsemble
from roboforg.workflows.act.common import (
    create_act_agent,
    load_checkpoint_into_agent,
    read_checkpoint_setup,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLATFORM_ROOT = PROJECT_ROOT / "platform"
if str(PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(PLATFORM_ROOT))


# 每次策略查询前，把环境单帧观测转换为 ACT 所需的 batch 形式。
def _batch_observation(observation: dict[str, Any], camera_keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    state = np.asarray(observation["state"], dtype=np.float32)
    if state.ndim != 1 or state.size == 0 or not np.all(np.isfinite(state)):
        raise ValueError("Environment state must be a finite non-empty one-dimensional array")
    result = {"state": state[None]}
    for key in camera_keys:
        image = np.asarray(observation[key])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"Camera {key!r} must be an RGB uint8 image")
        result[key] = image[None]
    return result


# 训练中测评或独立测评开始时调用：按开关创建无人工干预的 Pick 环境。
def _make_environment(*, show_viewer: bool = False):
    if show_viewer:
        os.environ.pop("MUJOCO_GL", None)
    else:
        os.environ.setdefault("MUJOCO_GL", "egl")
    from examples.sim_pick.config import TrainConfig

    env_config = TrainConfig()
    env_config.show_viewer = show_viewer
    env_config.input_device = None
    return env_config.get_environment(fake_env=False, save_video=False, classifier=False)


# 执行一个完整回合：每步重新预测动作块，用 Temporal Ensemble 融合后执行。
def _run_episode(
    agent: Any,
    env: Any,
    *,
    seed: int,
    print_policy_output: bool = False,
) -> dict[str, float | bool]:
    from envs.wrappers.action_utils import project_action_to_unit_balls

    observation, _ = env.reset(seed=seed)
    if tuple(env.action_space.shape) != (agent.policy.action_dim,):
        raise ValueError("Environment action dimension does not match ACT policy")
    ensemble = ActionEnsemble(agent.policy.action_horizon, action_dim=agent.policy.action_dim, decay=0.01)
    episode_return = 0.0
    clipped_steps = 0
    limit = int(env.unwrapped.config.max_episode_length)
    for step in range(limit):
        prediction = np.asarray(jax.device_get(agent.predict(
            _batch_observation(observation, agent.policy.camera_keys)
        )))
        expected_shape = (1, agent.policy.action_horizon, agent.policy.action_dim)
        if prediction.shape != expected_shape:
            raise ValueError(f"ACT prediction must have shape {expected_shape}, got {prediction.shape}")
        action_chunk = prediction[0]
        ensemble.add_prediction(step, action_chunk)
        action = ensemble.get_action(step)
        safe_action = project_action_to_unit_balls(np.clip(action, env.action_space.low, env.action_space.high))
        if print_policy_output:
            print(
                f"Policy output step={step + 1}: {np.array2string(safe_action, precision=6, separator=', ')}",
                flush=True,
            )
        if not np.array_equal(safe_action, action):
            clipped_steps += 1
        observation, reward, terminated, truncated, info = env.step(safe_action)
        episode_return += float(reward)
        if terminated or truncated:
            return {
                "success": bool(info.get("is_success", False)),
                "return": episode_return,
                "steps": step + 1,
                "clipped_steps": clipped_steps,
            }
    return {"success": False, "return": episode_return, "steps": limit, "clipped_steps": clipped_steps}


# 训练循环定期调用，或独立测评加载模型后调用：运行多个完整回合并汇总成功率。
def evaluate_policy(agent: Any, *, num_episodes: int, seed: int, env: Any = None) -> dict[str, float]:
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    owns_env = env is None
    if owns_env:
        env = _make_environment()
    try:
        episodes = []
        for index in range(num_episodes):
            result = _run_episode(agent, env, seed=seed + index)
            episodes.append(result)
            print(f"Evaluation {index + 1}/{num_episodes}: success={result['success']}, "
                  f"steps={result['steps']}", flush=True)
        return {
            "success_rate": sum(int(item["success"]) for item in episodes) / num_episodes,
            "mean_return": float(np.mean([item["return"] for item in episodes])),
            "mean_steps": float(np.mean([item["steps"] for item in episodes])),
            "mean_clipped_steps": float(np.mean([item["clipped_steps"] for item in episodes])),
        }
    finally:
        if owns_env:
            env.close()


# 独立测评开屏时调用：逐回合观察动作，不汇总或输出成功率。
def observe_policy(
    agent: Any,
    *,
    num_episodes: int,
    seed: int,
    env: Any,
    print_policy_output: bool = True,
) -> None:
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    for index in range(num_episodes):
        result = _run_episode(
            agent,
            env,
            seed=seed + index,
            print_policy_output=print_policy_output,
        )
        print(f"Observation {index + 1}/{num_episodes}: steps={result['steps']}", flush=True)


# 独立测评启动时调用：接收 checkpoint 路径和是否开屏的选择。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved ACT checkpoint in the Pick simulator")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-episodes", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show-viewer", choices=("true", "false"), default="false")
    parser.add_argument(
        "--print-policy-output",
        choices=("true", "false"),
        default="false",              # 测评时间，是否打印出策略输出
        help="Print each action sent to the simulator during open-viewer observation.",
    )
    args = parser.parse_args()
    if args.num_episodes is None:
        args.num_episodes = 1 if args.show_viewer == "true" else 40
    if args.num_episodes <= 0:
        parser.error("--num-episodes must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    return args


# 独立测评启动时调用：加载 checkpoint；开屏观察或离屏统计成功率。
def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    config, normalizer, metadata = read_checkpoint_setup(checkpoint)
    show_viewer = args.show_viewer == "true"
    env = _make_environment(show_viewer=show_viewer)
    try:
        observation, _ = env.reset(seed=args.seed)
        example_batch = (
            _batch_observation(observation, config.camera_keys),
            np.zeros((1, config.action_horizon, config.action_dim), dtype=np.float32),
        )
        agent = create_act_agent(
            config,
            normalizer=normalizer,
            example_batch=example_batch,
            rng=jax.random.key(args.seed),
            load_pretrained_backbone=False,
        )
        agent, _ = load_checkpoint_into_agent(checkpoint, agent)
        print(f"Checkpoint: {checkpoint}; completed_epochs={metadata['epoch'] + 1}", flush=True)
        if show_viewer:
            observe_policy(
                agent,
                num_episodes=args.num_episodes,
                seed=args.seed,
                env=env,
                print_policy_output=args.print_policy_output == "true",
            )
        else:
            results = evaluate_policy(agent, num_episodes=args.num_episodes, seed=args.seed, env=env)
            print(f"Evaluation metrics: {results}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
