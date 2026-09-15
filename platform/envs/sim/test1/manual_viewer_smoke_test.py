from __future__ import annotations

import sys
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np


ENVS_ROOT = Path(__file__).resolve().parents[2]
if str(ENVS_ROOT) not in sys.path:
    sys.path.insert(0, str(ENVS_ROOT))

from sim.robot_sim_env1 import RobotSimEnv, RobotSimEnvConfig


def test_manual_viewer_open_refresh_and_close() -> None:
    """人工检查 MuJoCo viewer 能否正常打开、刷新并关闭。"""

    env = RobotSimEnv(show_viewer=True)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    print_steps = {1, 20, 40, 60, 80, 100}
    viewer_was_opened = False

    try:
        obs, _ = env.reset()
        initial_position = obs["state"]["tcp_pose"][:3].copy()
        print("initial tcp position:", np.array2string(initial_position, precision=6, suppress_small=True))
        print("The viewer will run for 10 seconds. Close the viewer window to exit early.")

        for step in range(1, 101):
            step_start = time.perf_counter()
            obs, _, terminated, truncated, _ = env.step(action)

            # 第一次 step 会惰性创建 viewer；用户手动关闭后 _render 会将其设为 None。
            if env._viewer is None:
                print(f"viewer closed at step {step}")
                break
            viewer_was_opened = viewer_was_opened or env._viewer.is_running()

            if step in print_steps:
                tcp_position = obs["state"]["tcp_pose"][:3]
                print(f"step {step}, simulation time={env.data.time:.1f}s")
                print("tcp position:", np.array2string(tcp_position, precision=6, suppress_small=True))
                print("position drift:", np.array2string(tcp_position - initial_position, precision=9, suppress_small=True))

            assert not terminated
            assert truncated == (step == 100)

            # 按环境10Hz控制频率刷新，避免仿真循环远快于实际时间。
            remaining_time = 1.0 / env.config.hz - (time.perf_counter() - step_start)
            if remaining_time > 0:
                time.sleep(remaining_time)
    finally:
        env.close()
        print("is_closed:", env.is_closed)
        print("viewer:", env._viewer)
        print("renderer:", env._renderer)

    assert viewer_was_opened
    assert env.is_closed is True
    assert env._viewer is None
    assert env._renderer is None


def test_manual_move_reset_and_close_flow() -> None:
    """人工观察移动到固定点、短暂停留、重置、保持和关闭的完整流程。"""

    env = RobotSimEnv(show_viewer=True)
    goal_position = np.asarray([0.5, 0.2, 0.1], dtype=np.float64)
    reach_tolerance = 0.001
    viewer_created = False
    goal_reached = False

    def step_at_control_rate(action: np.ndarray):
        step_start = time.perf_counter()
        result = env.step(action)
        remaining_time = 1.0 / env.config.hz - (time.perf_counter() - step_start)
        if remaining_time > 0:
            time.sleep(remaining_time)
        return result

    try:
        initial_obs, _ = env.reset()
        initial_position = initial_obs["state"]["tcp_pose"][:3].copy()
        current_position = initial_position.copy()
        print("initial tcp:", np.array2string(initial_position, precision=6, suppress_small=True))
        print("goal tcp:", np.array2string(goal_position, precision=6, suppress_small=True))

        # 根据每周期观测动态修正三轴动作，最多30周期到达固定目标点。
        for step in range(1, 31):
            position_error = goal_position - current_position
            max_error = float(np.max(np.abs(position_error)))
            translation_action = position_error / max_error if max_error > env.config.action_scale[0] else position_error / env.config.action_scale[0]
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            action[:3] = translation_action.astype(np.float32)

            obs, _, terminated, truncated, _ = step_at_control_rate(action)
            current_position = obs["state"]["tcp_pose"][:3].copy()
            distance = float(np.linalg.norm(goal_position - current_position))
            viewer_created = viewer_created or (env._viewer is not None and env._viewer.is_running())

            if step in (1, 5, 10, 15, 20, 25, 30):
                print(f"move step {step} tcp:", np.array2string(current_position, precision=6, suppress_small=True))
                print(f"distance to goal: {distance * 1000.0:.6f} mm")

            assert not terminated
            assert not truncated
            if distance <= reach_tolerance:
                goal_reached = True
                print(f"goal reached at step {step}, distance={distance * 1000.0:.6f} mm")
                break

        # 到达后输入一个零动作周期，观察机械臂停止增加目标后的状态。
        stop_action = np.zeros(env.action_space.shape, dtype=np.float32)
        stopped_obs, _, _, _, _ = step_at_control_rate(stop_action)
        stopped_position = stopped_obs["state"]["tcp_pose"][:3].copy()
        print("position after one stop step:", np.array2string(stopped_position, precision=6, suppress_small=True))

        # 重置后保持5周期，检查机器人是否稳定停留在标准初始位置。
        reset_obs, _ = env.reset()
        reset_position = reset_obs["state"]["tcp_pose"][:3].copy()
        print("reset tcp:", np.array2string(reset_position, precision=6, suppress_small=True))
        hold_obs = reset_obs
        for _ in range(5):
            hold_obs, _, terminated, truncated, _ = step_at_control_rate(stop_action)
            assert not terminated
            assert not truncated
        hold_position = hold_obs["state"]["tcp_pose"][:3].copy()
        print("tcp after 5 reset hold steps:", np.array2string(hold_position, precision=6, suppress_small=True))
        print("reset hold drift:", np.array2string(hold_position - reset_position, precision=9, suppress_small=True))

        assert goal_reached
        assert viewer_created
        assert np.allclose(reset_position, initial_position)
    finally:
        env.close()
        print("viewer created successfully:", viewer_created)
        print("goal reached successfully:", goal_reached)
        print("is_closed:", env.is_closed)
        print("viewer after close:", env._viewer)
        print("renderer after close:", env._renderer)

    assert env.is_closed is True
    assert env._viewer is None
    assert env._renderer is None
