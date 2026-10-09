"""Diffusion Policy 仿真评估；训练中评估和独立评估共用同一入口。"""
from __future__ import annotations

import argparse
from collections import deque
import os
from pathlib import Path
import sys
from typing import Any

import jax
import numpy as np

from roboforg.utils.checkpoint_untils import find_latest_checkpoint, find_latest_run_dir
from roboforg.workflows.diffusion_policy.common import (
    checkpoint_setup,
    create_policy,
    restore_ema_params,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints" / "diffusion_policy"
PLATFORM_ROOT = PROJECT_ROOT / "platform"

# 仿真环境代码位于项目的 platform 目录，将其加入模块搜索路径。
if str(PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(PLATFORM_ROOT))


# 根据是否打开窗口设置 MuJoCo 渲染模式，并创建无人为输入的 Pick 环境。
def _make_environment(*, show_viewer: bool = False):
    # 开屏使用系统图形窗口；离屏评估使用 EGL 渲染后端。
    if show_viewer:
        os.environ.pop("MUJOCO_GL", None)
    else:
        os.environ.setdefault("MUJOCO_GL", "egl")

    # 延迟导入环境配置，避免仅查看命令帮助时提前初始化仿真依赖。
    from examples.sim_pick.config import TrainConfig

    # 关闭输入设备、视频保存和分类器，只保留策略评估需要的环境。
    config = TrainConfig()
    config.show_viewer = show_viewer
    config.input_device = None
    return config.get_environment(fake_env=False, save_video=False, classifier=False)


# 检查环境单帧观测的状态和相机格式，并统一转换为 NumPy 数组。
def _validate_observation(observation: dict[str, Any], camera_keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    # 状态固定为 23 维浮点向量，并拒绝 NaN 或无穷值。
    state = np.asarray(observation["state"], dtype=np.float32)
    if state.shape != (23,) or not np.all(np.isfinite(state)):
        raise ValueError("Environment state must have shape (23,) and contain finite values.")
    result = {"state": state}

    # 每路相机必须提供单帧 uint8 RGB 图像。
    for key in camera_keys:
        image = np.asarray(observation[key])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"Camera {key!r} must be an RGB uint8 image.")
        result[key] = image
    return result


# 将最近 n_obs_steps 帧历史观测堆叠成策略需要的 [B, To, ...] batch。
def _history_batch(history: deque[dict[str, np.ndarray]], camera_keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {
        key: np.stack([observation[key] for observation in history], axis=0)[None]
        for key in ("state", *camera_keys)
    }


# 执行一个完整回合：每次预测 16 步、连续执行其中 8 步，再用新观测重新预测。
def _run_episode(policy, params, env, *, seed: int, rng: jax.Array,
                 print_policy_output: bool = False) -> tuple[dict[str, float | bool], jax.Array]:
    # 环境要求平移动作和转动动作分别落在单位球内，执行前统一投影。
    from envs.wrappers.action_utils import project_action_to_unit_balls

    # reset 后将首帧重复 n_obs_steps 次，补齐回合开始时不存在的历史观测。
    observation, _ = env.reset(seed=seed)
    first = _validate_observation(observation, policy.camera_keys)
    history = deque([first] * policy.n_obs_steps, maxlen=policy.n_obs_steps)
    episode_return = 0.0
    clipped_steps = 0
    environment_steps = 0
    limit = int(env.unwrapped.config.max_episode_length)

    # 一个回合内不断预测动作块，直到环境终止或达到最大步数。
    while environment_steps < limit:
        # 每次预测使用新的随机键，从噪声开始生成完整动作序列。
        rng, prediction_rng = jax.random.split(rng)
        prediction = policy.predict_action(params, _history_batch(history, policy.camera_keys), prediction_rng)
        action_chunk = np.asarray(jax.device_get(prediction["action"]))[0]
        expected = (policy.n_action_steps, policy.action_dim)
        if action_chunk.shape != expected:
            raise ValueError(f"Diffusion Policy action chunk must have shape {expected}.")

        # 按顺序执行当前动作块中的 8 个动作，中途结束时立即停止。
        for action in action_chunk:
            # 先裁剪到 action space，再投影到环境允许的单位球范围。
            safe_action = project_action_to_unit_balls(
                np.clip(action, env.action_space.low, env.action_space.high)
            )

            # 开启配置后，打印真正发送给环境的安全动作。
            if print_policy_output:
                print(
                    f"Policy output step={environment_steps + 1}: "
                    f"{np.array2string(safe_action, precision=6, separator=', ')}",
                    flush=True,
                )
            if not np.array_equal(safe_action, action):
                clipped_steps += 1

            # 执行动作，累计回报，并把最新观测加入固定长度历史队列。
            observation, reward, terminated, truncated, info = env.step(safe_action)
            environment_steps += 1
            episode_return += float(reward)
            history.append(_validate_observation(observation, policy.camera_keys))

            # 回合结束时返回成功标记、回报、步数和动作裁剪次数。
            if terminated or truncated or environment_steps >= limit:
                return {
                    "success": bool(info.get("is_success", False)),
                    "return": episode_return,
                    "steps": environment_steps,
                    "clipped_steps": clipped_steps,
                }, rng

    return {"success": False, "return": episode_return, "steps": limit,
            "clipped_steps": clipped_steps}, rng


# 连续运行多个完整回合，并汇总成功率、平均回报和平均步数。
def evaluate_policy(policy, params, *, num_episodes: int, seed: int, env: Any = None,
                    show_viewer: bool = False, print_policy_output: bool = False) -> dict[str, float]:
    """连续执行 8 步动作块，多回合汇总成功率；供训练循环和独立评估复用。"""
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive.")

    # 未传入环境时由函数自行创建，并在结束后负责关闭。
    owns_env = env is None
    if owns_env:
        env = _make_environment(show_viewer=show_viewer)
    rng = jax.random.key(seed)
    episodes = []
    try:
        # 每个回合使用不同环境种子，同时连续传递扩散采样随机键。
        for index in range(num_episodes):
            result, rng = _run_episode(
                policy, params, env, seed=seed + index, rng=rng,
                print_policy_output=print_policy_output,
            )
            episodes.append(result)
            print(f"Evaluation {index + 1}/{num_episodes}: success={result['success']}, "
                  f"steps={result['steps']}", flush=True)

        # 将逐回合结果归约为训练日志和独立评估共同使用的指标。
        return {
            "success_rate": sum(int(item["success"]) for item in episodes) / num_episodes,
            "mean_return": float(np.mean([item["return"] for item in episodes])),
            "mean_steps": float(np.mean([item["steps"] for item in episodes])),
            "mean_clipped_steps": float(np.mean([item["clipped_steps"] for item in episodes])),
        }
    finally:
        # 只关闭本函数创建的环境，调用方传入的环境仍由调用方管理。
        if owns_env:
            env.close()


# 读取独立评估命令中的 checkpoint、回合数、窗口和动作打印选项。
def parse_args() -> argparse.Namespace:
    # checkpoint 可以省略；省略时 main 会自动选择最新权重。
    parser = argparse.ArgumentParser(description="Evaluate a Diffusion Policy checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Checkpoint path; omit it to use the latest Pick checkpoint.")
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show-viewer", choices=("true", "false"), default="false")
    parser.add_argument("--print-policy-output", choices=("true", "false"), default="false")
    args = parser.parse_args()

    # 用户未指定回合数时，开屏默认 10 回合，离屏默认 40 回合。
    if args.num_episodes is None:
        args.num_episodes = 10 if args.show_viewer == "true" else 40
    if args.num_episodes <= 0 or args.seed < 0:
        parser.error("num-episodes must be positive and seed must be non-negative.")
    return args


# 独立评估入口：定位 checkpoint、恢复 EMA 参数并运行一次多回合评估。
def main() -> None:
    # 读取命令行配置；没有指定权重时寻找最新实验的最新 checkpoint。
    args = parse_args()
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = find_latest_checkpoint(
            find_latest_run_dir(CHECKPOINT_ROOT, "pick", "diffusion_policy")
        )

    # 从 checkpoint 读取模型配置、normalizer 和已完成的 epoch 信息。
    checkpoint = checkpoint.expanduser().resolve()
    config, normalizer, metadata = checkpoint_setup(checkpoint)
    policy = create_policy(config, normalizer=normalizer)

    # 用正确输入形状初始化参数树，为严格恢复 checkpoint 提供结构模板。
    example = {
        "state": np.zeros((1, config.n_obs_steps, config.state_dim), dtype=np.float32),
        **{
            key: np.zeros((1, config.n_obs_steps, 128, 128, 3), dtype=np.uint8)
            for key in config.camera_keys
        },
    }

    # 评估始终恢复 checkpoint 中更平滑的 EMA 参数，而不是普通训练参数。
    initialized = policy.init_parameters(jax.random.key(args.seed), example)
    params = restore_ema_params(checkpoint, initialized)
    print(f"Checkpoint: {checkpoint}; completed_epochs={metadata['epoch'] + 1}", flush=True)

    # 按命令行选择开屏/离屏、回合数以及是否打印策略动作。
    results = evaluate_policy(
        policy, params, num_episodes=args.num_episodes, seed=args.seed,
        show_viewer=args.show_viewer == "true",
        print_policy_output=args.print_policy_output == "true",
    )
    print(f"Evaluation metrics: {results}", flush=True)


if __name__ == "__main__":
    main()
