"""Collect compact, successful human demonstrations for the Pick task.

Each saved trajectory contains observations ``obs_0 ... obs_T`` and executed
actions ``action_0 ... action_(T-1)``.  Keeping the terminal observation once
is required by :class:`DataBuffer`, but unlike the original SEPO collector we
never serialize a ``next_observations`` copy inside every transition.
"""

from __future__ import annotations

import argparse
import datetime as dt
import pickle
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLATFORM_ROOT = PROJECT_ROOT / "platform"
# 允许脚本既能通过 ``python -m``，也能从项目目录直接找到 examples 和 envs。
for import_root in (PROJECT_ROOT, PLATFORM_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from examples.sim_pick.config import TrainConfig


def _copy_tree(value: Any) -> Any:
    """Copy an observation tree so a later environment render cannot alter it."""
    # observation 是嵌套字典；叶子数组必须复制，避免下一次渲染覆盖已录制图像。
    if isinstance(value, Mapping):
        return {key: _copy_tree(child) for key, child in value.items()}
    return np.asarray(value).copy()


def _stack_tree(values: list[Any]) -> Any:
    """Stack a same-structured sequence of observations along its time axis."""
    if not values:
        raise ValueError("Cannot stack an empty trajectory field.")
    if isinstance(values[0], Mapping):
        # 对 state/front/wrist 等字段分别沿时间维递归堆叠。
        keys = tuple(values[0])
        if any(not isinstance(value, Mapping) or tuple(value) != keys for value in values[1:]):
            raise ValueError("Observation structure changed within an episode.")
        return {key: _stack_tree([value[key] for value in values]) for key in keys}
    return np.stack([np.asarray(value) for value in values], axis=0)


def _hold_action(env) -> np.ndarray:
    """Return the no-motion action while keeping the reset gripper target open."""
    # 未人工操作时不移动；夹爪仍明确带着“张开”的目标。
    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    if action.shape == (7,):
        action[6] = -1.0
    return action


def _update_held_gripper(hold_action: np.ndarray, executed_action: np.ndarray, threshold: float) -> None:
    """Carry the last commanded gripper target through non-intervention steps."""
    if hold_action.shape != (7,):
        return
    # 人工按过开/合后，后续无移动步骤也要持续执行该夹爪目标。
    if executed_action[6] > threshold:
        hold_action[6] = 1.0
    elif executed_action[6] < -threshold:
        hold_action[6] = -1.0


def _is_first_effective_action(
    executed_action: np.ndarray,
    held_gripper_target: float | None,
    motion_threshold: float,
    gripper_threshold: float,
) -> bool:
    """Detect the first arm command or a change of the gripper target."""
    arm_moved = bool(np.linalg.norm(executed_action[:6]) > motion_threshold)
    gripper_changed = False
    if held_gripper_target is not None and executed_action.shape == (7,):
        new_target = float(executed_action[6])
        gripper_changed = bool(
            (new_target > gripper_threshold and held_gripper_target < 0.0)
            or (new_target < -gripper_threshold and held_gripper_target > 0.0)
        )
    return arm_moved or gripper_changed


def _finalize_trajectory(
    observations: list[dict[str, np.ndarray]],
    actions: list[np.ndarray],
    rewards: list[np.float32],
    masks: list[np.float32],
    dones: list[np.bool_],
) -> dict[str, Any]:
    """Create the compact ``DataBuffer`` trajectory format from one episode."""
    if len(observations) != len(actions) + 1:
        raise ValueError("A trajectory must contain exactly one more observation than actions.")
    # 这是 DataBuffer 可直接读取的紧凑轨迹：观测 T+1，其他字段 T。
    return {
        "observations": _stack_tree(observations),
        "actions": np.stack(actions, axis=0).astype(np.float32, copy=False),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "masks": np.asarray(masks, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.bool_),
    }


def collect_demonstrations(
    env, *, successes_needed: int, seed: int | None, start_motion_threshold: float = 1e-6
) -> tuple[list[dict[str, Any]], dict[str, int], bool]:
    """Collect one complete batch; interrupted batches are deliberately discarded."""
    trajectories: list[dict[str, Any]] = []
    attempted_episodes = 0
    total_steps = 0
    # 每轮 episode 从当前观测开始；第一次 reset 使用指定种子，之后继续环境随机序列。
    observation, _ = env.reset(seed=seed)
    hold_action = _hold_action(env)
    control_dt = 1.0 / float(env.unwrapped.config.hz)
    gripper_threshold = float(env.unwrapped.config.gripper_threshold)

    try:
        while len(trajectories) < successes_needed:
            # 本 episode 暂存。只在成功且未作废时才加入 trajectories。
            episode_observations: list[dict[str, np.ndarray]] = []
            episode_actions: list[np.ndarray] = []
            episode_rewards: list[np.float32] = []
            episode_masks: list[np.float32] = []
            episode_dones: list[np.bool_] = []
            episode_return = 0.0
            waiting_steps = 0
            recording_started = False
            next_deadline = time.monotonic() + control_dt

            while True:
                # Wrapper 在人工接管时把真正执行的动作放入 info；否则执行保持动作。
                held_gripper_target = float(hold_action[6]) if hold_action.shape == (7,) else None
                next_observation, reward, terminated, truncated, info = env.step(hold_action)
                if "is_takeover_active" not in info:
                    raise RuntimeError("Collection requires InterventionWrapper to report is_takeover_active.")
                executed_action = np.asarray(
                    info.get("intervene_action", hold_action), dtype=env.action_space.dtype
                ).copy()
                _update_held_gripper(hold_action, executed_action, gripper_threshold)

                # 接管开关打开并首次执行有效动作时，从该动作执行前的观测开始录制。
                # 单独按空格/RB、或夹爪持续保持原目标，都不会触发录制。
                if not recording_started:
                    recording_started = bool(info["is_takeover_active"]) and _is_first_effective_action(
                        executed_action,
                        held_gripper_target,
                        start_motion_threshold,
                        gripper_threshold,
                    )
                    if recording_started:
                        episode_observations.append(_copy_tree(observation))
                    else:
                        waiting_steps += 1

                # 开始后连续保存，包括操作过程中的停顿；保持 T 个动作对应 T+1 帧观测。
                done = bool(terminated or truncated)
                if recording_started:
                    episode_actions.append(executed_action)
                    episode_rewards.append(np.float32(reward))
                    episode_masks.append(np.float32(1.0 - float(terminated)))
                    episode_dones.append(np.bool_(done))
                    episode_observations.append(_copy_tree(next_observation))
                episode_return += float(reward)
                total_steps += 1
                observation = next_observation

                # 以环境 hz 控制采样节奏，使人工动作与数据时间步一致。
                remaining_time = next_deadline - time.monotonic()
                if remaining_time > 0.0:
                    time.sleep(remaining_time)
                next_deadline += control_dt

                if not done:
                    viewer_closed = (
                        getattr(env.unwrapped, "show_viewer", False)
                        and getattr(env.unwrapped, "_viewer", None) is None
                    )
                    if viewer_closed:
                        print("Viewer was closed; stopping collection without saving the unfinished episode.")
                        return trajectories, {
                            "successful_episodes": len(trajectories),
                            "attempted_episodes": attempted_episodes,
                            "total_steps": total_steps,
                        }, False
                    continue

                attempted_episodes += 1
                is_success = bool(info.get("is_success", False))
                rerecord = bool(info.get("rerecord_episode", False))
                episode_steps = len(episode_actions)
                # 失败回合、超时回合、以及按 r 标记作废的回合均不写入数据集。
                if is_success and not rerecord and recording_started:
                    trajectories.append(
                        _finalize_trajectory(
                            episode_observations,
                            episode_actions,
                            episode_rewards,
                            episode_masks,
                            episode_dones,
                        )
                    )

                print(
                    f"attempt={attempted_episodes}, saved={len(trajectories)}/{successes_needed}, "
                    f"steps={episode_steps}, waiting_steps={waiting_steps}, return={episode_return:.3f}, "
                    f"is_success={is_success}, rerecord_episode={rerecord}"
                )
                break

            if len(trajectories) >= successes_needed:
                break
            # 下一次尝试使用新的随机方块位置，并将夹爪目标恢复为张开。
            observation, _ = env.reset()
            hold_action = _hold_action(env)
    except KeyboardInterrupt:
        print("\nCollection interrupted; this incomplete batch will be discarded.")

    collection_complete = len(trajectories) == successes_needed
    return trajectories, {
        "successful_episodes": len(trajectories),
        "attempted_episodes": attempted_episodes,
        "total_steps": total_steps,
    }, collection_complete


def _save_dataset(*, trajectories: list[dict[str, Any]], statistics: dict[str, int],
                  split: str, output_root: Path, output_dir_name: str) -> Path:
    """Write a single session file shared by future ACT and Diffusion Policy loaders."""
    # train/eval 仅通过目录隔离；二者的数据格式完全相同，可供 ACT 和 DP 共用。
    output_dir = output_root / "pick" / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = output_dir / f"pick_{len(trajectories)}_demos_{timestamp}.pkl"
    # 一个 pkl 保存本次采集的全部成功轨迹，而非每条轨迹产生一个小文件。
    payload = {
        "format_version": 1,
        "task": "pick",
        "split": split,
        "statistics": statistics,
        "trajectories": trajectories,
    }
    with output_path.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect successful human Pick demonstrations.")
    parser.add_argument("--successes-needed", type=int, default=10, help="Successful trajectories to put in this one file.")
    parser.add_argument("--split", choices=("train", "eval"), default="train")
    parser.add_argument("--output-dir-name", type=str,
                        help="Folder under datasets/pick; defaults to the split name.")
    parser.add_argument("--input-device", choices=("keyboard", "gamepad", "spacemouse"), default="keyboard")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-motion-threshold", type=float, default=1e-6,
                        help="Minimum six-axis command norm that starts recording after takeover.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "datasets")
    args = parser.parse_args()
    if args.successes_needed <= 0:
        parser.error("--successes-needed must be positive.")
    if not np.isfinite(args.start_motion_threshold) or args.start_motion_threshold < 0.0:
        parser.error("--start-motion-threshold must be finite and non-negative.")
    output_dir_name = args.output_dir_name or args.split
    if output_dir_name in {".", ".."} or Path(output_dir_name).name != output_dir_name:
        parser.error("--output-dir-name must name one folder under datasets/pick.")

    # 收集阶段始终使用可视化窗口和人工输入设备，不创建 fake_env。
    config = TrainConfig()
    config.input_device = args.input_device
    config.show_viewer = True
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)
    try:
        trajectories, statistics, collection_complete = collect_demonstrations(
            env, successes_needed=args.successes_needed, seed=args.seed,
            start_motion_threshold=args.start_motion_threshold,
        )
    finally:
        env.close()

    # 只有完整收集到本轮目标条数才落盘；中断时丢弃内存中的部分成功轨迹。
    if not collection_complete:
        print(
            f"Collection stopped at {statistics['successful_episodes']}/{args.successes_needed} "
            "successful demonstrations; this incomplete batch was discarded."
        )
        return
    output_path = _save_dataset(
        trajectories=trajectories,
        statistics=statistics,
        split=args.split,
        output_root=args.output_root,
        output_dir_name=output_dir_name,
    )
    print("statistics:", statistics)
    print("saved:", output_path)


if __name__ == "__main__":
    main()


#conda activate gym_hil
#cd ~/project/RoboForge                                           目录
#bash roboforg/workflows/collect_data/collect_data.sh             采集一轮训练数据（10条轨迹）
