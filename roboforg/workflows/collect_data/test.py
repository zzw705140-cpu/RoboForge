"""Preflight and stored-data format checks for Pick demonstrations."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLATFORM_ROOT = PROJECT_ROOT / "platform"
for import_root in (PROJECT_ROOT, PLATFORM_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from examples.sim_pick.config import TrainConfig
from roboforg.data.act_data.data_conversion import ACTBatchConverter
from roboforg.data.chunk_bc_buffer import ChunkBCBuffer


IMAGE_KEYS = ("front", "wrist")


def _validate_observation(observation: dict[str, np.ndarray]) -> None:
    """Validate the wrapper output expected by later demonstration collection."""
    expected_keys = {"state", *IMAGE_KEYS}
    if set(observation) != expected_keys:
        raise KeyError(f"Expected observation keys {sorted(expected_keys)}, got {sorted(observation)}.")

    state = observation["state"]
    if state.ndim != 1 or state.size == 0 or not np.isfinite(state).all():
        raise ValueError("state must be a non-empty finite one-dimensional array.")

    for key in IMAGE_KEYS:
        image = observation[key]
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"{key!r} must be an RGB uint8 image, got {image.shape}, {image.dtype}.")


def _save_previews(observation: dict[str, np.ndarray], output_dir: Path) -> None:
    """Save the policy camera frames for manual view-angle inspection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for key in IMAGE_KEYS:
        output_path = output_dir / f"{key}.png"
        iio.imwrite(output_path, observation[key])
        print(f"saved preview: {output_path}")


def run_preflight(*, seed: int, preview_dir: Path, show_viewer: bool) -> None:
    """Create the Pick environment, validate visual observations, then release resources."""
    config = TrainConfig()
    config.show_viewer = show_viewer
    # The preflight has no human-control dependency; collection will configure its input device later.
    config.input_device = None
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)

    try:
        observation, _ = env.reset(seed=seed)
        _validate_observation(observation)
        _save_previews(observation, preview_dir)

        first_images = {key: observation[key].copy() for key in IMAGE_KEYS}
        hold_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        if hold_action.shape == (7,):
            hold_action[6] = -1.0

        next_observation, _, _, _, _ = env.step(hold_action)
        _validate_observation(next_observation)

        for key in IMAGE_KEYS:
            if np.shares_memory(first_images[key], next_observation[key]):
                raise RuntimeError(f"{key!r} frames share memory across steps.")
            if not np.array_equal(first_images[key], observation[key]):
                raise RuntimeError(f"{key!r} initial frame changed after a later render.")

        print(f"state shape: {observation['state'].shape}")
        print(f"action shape: {env.action_space.shape}")
        for key in IMAGE_KEYS:
            print(f"{key}: shape={observation[key].shape}, dtype={observation[key].dtype}")
        print("collection preflight passed")
    finally:
        env.close()


def run_act_batch_format_check(*, dataset_root: Path, split: str, batch_size: int, action_horizon: int) -> None:
    """Check real saved data through chunk sampling and the ACT format converter."""
    if batch_size <= 0 or action_horizon <= 0:
        raise ValueError("batch_size and action_horizon must be positive.")

    # 读取同一 split 下所有采集批次，并合并其中的完整轨迹。
    data_dir = dataset_root / "pick" / split
    paths = sorted(data_dir.glob("*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No demonstration files found in {data_dir}.")

    trajectories = []
    for path in paths:
        with path.open("rb") as file:
            payload = pickle.load(file)
        if not isinstance(payload, dict) or "trajectories" not in payload:
            raise ValueError(f"{path} is not a RoboForge compact demonstration file.")
        trajectories.extend(payload["trajectories"])
    if not trajectories:
        raise ValueError(f"No trajectories found in {data_dir}.")

    # 真实轨迹 -> [B, 1, ...] 历史观测 + [B, k, 7] 动作 chunk。
    buffer = ChunkBCBuffer.from_trajectories(trajectories, seed=0)
    chunk_batch = buffer.sample_chunk(
        batch_size,
        observation_horizon=1,
        action_horizon=action_horizon,
    )
    converted_observations, actions = ACTBatchConverter(action_horizon)(chunk_batch)

    # 转换只移除长度为 1 的历史维；状态、像素和动作数值本身不能被改变。
    source_observations = chunk_batch["observations"]
    expected_state = np.asarray(source_observations["state"][:, 0])
    if not np.array_equal(np.asarray(converted_observations["state"]), expected_state):
        raise AssertionError("State values changed during ACT conversion.")
    if actions.shape != (batch_size, action_horizon, 7):
        raise AssertionError(f"Unexpected action shape: {actions.shape}.")
    if not np.array_equal(np.asarray(actions), np.asarray(chunk_batch["actions"])):
        raise AssertionError("Action values changed during ACT conversion.")

    for key in IMAGE_KEYS:
        expected_image = np.asarray(source_observations[key][:, 0])
        image = np.asarray(converted_observations[key])
        if image.shape != expected_image.shape or image.dtype != np.uint8:
            raise AssertionError(f"Unexpected {key!r} output: shape={image.shape}, dtype={image.dtype}.")
        if not np.array_equal(image, expected_image):
            raise AssertionError(f"{key!r} pixels changed during ACT conversion.")

    print(f"loaded trajectories: {len(trajectories)} from {len(paths)} files ({split})")
    print(f"chunk actions: {np.asarray(actions).shape}")
    print(f"ACT state: {np.asarray(converted_observations['state']).shape}")
    for key in IMAGE_KEYS:
        image = np.asarray(converted_observations[key])
        print(f"ACT {key}: shape={image.shape}, dtype={image.dtype}")
    print("ACT batch format check passed")


def _make_camera_preview(observation: dict[str, np.ndarray]) -> np.ndarray:
    """Combine the two RGB policy views into one labelled OpenCV preview."""
    frames = []
    for key in IMAGE_KEYS:
        frame = cv2.cvtColor(observation[key], cv2.COLOR_RGB2BGR)
        cv2.putText(frame, key, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        frames.append(frame)
    return np.concatenate(frames, axis=1)


def run_manual_check(*, seed: int, input_device: str, max_steps: int) -> None:
    """Let a human test controls and policy camera views without saving a trajectory."""
    config = TrainConfig()
    config.show_viewer = True
    config.input_device = input_device
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)
    window_name = "RoboForge policy cameras: front | wrist"

    try:
        observation, _ = env.reset(seed=seed)
        _validate_observation(observation)
        hold_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        if hold_action.shape == (7,):
            hold_action[6] = -1.0

        print("Manual collection check started. Use the configured input device to move the robot.")
        print("Keyboard: m=success, Esc=failure, r=discard/re-record. Press q in the camera window to stop.")
        control_dt = 1.0 / float(env.unwrapped.config.hz)
        next_deadline = time.monotonic()
        last_info: dict[str, object] = {}

        for step in range(max_steps):
            cv2.imshow(window_name, _make_camera_preview(observation))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Manual check stopped from the camera preview window.")
                break

            next_observation, _, terminated, truncated, info = env.step(hold_action)
            _validate_observation(next_observation)
            last_info = dict(info)
            executed_action = np.asarray(info.get("intervene_action", hold_action))

            if "intervene_action" in info:
                print(f"step={step}: human action={executed_action}")

            observation = next_observation
            if terminated or truncated:
                print(
                    f"episode ended at step={step + 1}: "
                    f"is_success={bool(info.get('is_success', False))}, "
                    f"rerecord_episode={bool(info.get('rerecord_episode', False))}, "
                    f"last_action={executed_action}"
                )
                break

            next_deadline += control_dt
            remaining_time = next_deadline - time.monotonic()
            if remaining_time > 0.0:
                time.sleep(remaining_time)
            else:
                next_deadline = time.monotonic()
        else:
            print(f"Manual check reached max_steps={max_steps}; last_info={last_info}")
    finally:
        cv2.destroyAllWindows()
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Check visual observations before collecting demonstrations.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preview-dir", type=Path, default=Path("/tmp/roboforge_collect_data_preflight"))
    parser.add_argument("--show-viewer", action="store_true", help="Also open the normal MuJoCo viewer.")
    parser.add_argument("--manual-input", choices=("keyboard", "gamepad", "spacemouse"), help="Run an interactive control and camera-view test.")
    parser.add_argument("--max-steps", type=int, default=300, help="Maximum interactive-test steps before stopping.")
    parser.add_argument("--check-act-batch", action="store_true", help="Test saved demonstrations through ChunkBCBuffer and ACTBatchConverter.")
    parser.add_argument("--split", choices=("train", "eval"), default="train", help="Stored-data split used by --check-act-batch.")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "datasets")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size used by --check-act-batch.")
    parser.add_argument("--action-horizon", type=int, default=4, help="Chunk length used by --check-act-batch.")
    args = parser.parse_args()

    if os.environ.get("MUJOCO_GL") is None:
        print("Tip: if offscreen rendering fails, launch with MUJOCO_GL=egl.")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive.")
    if args.batch_size <= 0 or args.action_horizon <= 0:
        parser.error("--batch-size and --action-horizon must be positive.")

    if args.check_act_batch:
        run_act_batch_format_check(
            dataset_root=args.dataset_root,
            split=args.split,
            batch_size=args.batch_size,
            action_horizon=args.action_horizon,
        )
    elif args.manual_input is None:
        run_preflight(seed=args.seed, preview_dir=args.preview_dir, show_viewer=args.show_viewer)
    else:
        run_manual_check(seed=args.seed, input_device=args.manual_input, max_steps=args.max_steps)


if __name__ == "__main__":
    main()


"""
已保存演示数据的 ACT 格式转换测试（不启动仿真、不修改数据集）

测试链路：datasets/pick/{split}/*.pkl
      -> ChunkBCBuffer.sample_chunk
      -> ACTBatchConverter

检查 state 的单帧时间维是否正确移除、front/wrist 像素是否原样保留，
以及动作是否保持为 [B, action_horizon, 7]。

在任意终端目录运行 train 测试：
cd ~/project/RoboForge
python -m roboforg.workflows.collect_data.test \
    --check-act-batch \
    --split=train \
    --batch-size=4 \
    --action-horizon=4

测试 eval 数据时，仅将 --split=train 改为 --split=eval。
"""
