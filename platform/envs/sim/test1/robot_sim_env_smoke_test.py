from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation


ENVS_ROOT = Path(__file__).resolve().parents[2]
if str(ENVS_ROOT) not in sys.path:
    sys.path.insert(0, str(ENVS_ROOT))

from sim.robot_sim_env1 import CLOSED_GRIPPER_TARGET, MAX_GRIPPER_COMMAND, OPEN_GRIPPER_TARGET, RobotSimEnv, RobotSimEnvConfig


def test_zero_action_step() -> None:
    """测试零动作能否走通一次完整的环境交互流程。"""

    env = RobotSimEnv(show_viewer=False)

    try:
        # 重置环境并记录执行动作前的 TCP 位姿。
        initial_obs, _ = env.reset()
        initial_tcp_pose = initial_obs["state"]["tcp_pose"].copy()

        # 模拟上层算法输出七维零动作，再通过公开的 step 接口执行。
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        final_tcp_pose = obs["state"]["tcp_pose"]

        # 打印一个控制周期结束后的结果。
        print("action:", action)
        print("initial tcp pose:", initial_tcp_pose)
        print("final tcp pose:", final_tcp_pose)
        print("tcp pose change:", final_tcp_pose - initial_tcp_pose)
        print("observation:", obs)
        print("reward:", reward)
        print("terminated:", terminated)
        print("truncated:", truncated)
        print("info:", info)

        # 检查最小流程的返回格式和仿真推进时间。
        assert env.current_step == 1
        assert np.isclose(env.data.time, 1.0 / env.config.hz)
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    finally:
        env.close()


def test_observation_matches_space() -> None:
    """检查 reset 和 step 返回的观测是否符合环境声明的观测空间。"""

    env = RobotSimEnv(show_viewer=False)

    try:
        reset_obs, _ = env.reset()
        step_obs, _, _, _, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))

        print("observation keys:", tuple(step_obs["state"].keys()))
        print("observation dtypes:", {key: value.dtype for key, value in step_obs["state"].items()})

        assert env.observation_space.contains(reset_obs)
        assert env.observation_space.contains(step_obs)
    finally:
        env.close()


def test_positive_x_action() -> None:
    """测试正向 X 动作能否让 TCP 沿 X 正方向移动。"""

    env = RobotSimEnv(show_viewer=False)

    try:
        # 重置环境并记录动作执行前的位置。
        initial_obs, _ = env.reset()
        initial_position = initial_obs["state"]["tcp_pose"][:3].copy()

        # 模拟上层算法给出 X 正方向的最大归一化动作。
        action = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        target_position = initial_position.copy()
        target_position[0] += env.config.action_scale[0]
        target_position = np.clip(target_position, env.config.pose_limit_low[:3], env.config.pose_limit_high[:3])

        # 执行一个控制周期，并读取最终 TCP 位置。
        obs, _, _, _, _ = env.step(action)
        final_position = obs["state"]["tcp_pose"][:3]
        position_change = final_position - initial_position

        print("initial tcp position:", initial_position)
        print("target tcp position:", target_position)
        print("final tcp position:", final_position)
        print("tcp position change:", position_change)

        # 一个控制周期内不要求完全到达目标，只检查实际运动方向。
        assert final_position[0] > initial_position[0]
    finally:
        env.close()


def test_positive_y_action() -> None:
    """测试正向 Y 动作能否让 TCP 沿 Y 正方向移动。"""

    env = RobotSimEnv(show_viewer=False)

    try:
        # 重置环境并记录动作执行前的位置。
        initial_obs, _ = env.reset()
        initial_position = initial_obs["state"]["tcp_pose"][:3].copy()

        # 模拟上层算法给出 Y 正方向的最大归一化动作。
        action = np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        target_position = initial_position.copy()
        target_position[1] += env.config.action_scale[0]
        target_position = np.clip(target_position, env.config.pose_limit_low[:3], env.config.pose_limit_high[:3])

        # 执行一个控制周期，并读取最终 TCP 位置。
        obs, _, _, _, _ = env.step(action)
        final_position = obs["state"]["tcp_pose"][:3]
        position_change = final_position - initial_position

        print("initial tcp position:", initial_position)
        print("target tcp position:", target_position)
        print("final tcp position:", final_position)
        print("tcp position change:", position_change)

        assert final_position[1] > initial_position[1]
    finally:
        env.close()


def test_positive_z_action() -> None:
    """测试正向 Z 动作能否让 TCP 沿 Z 正方向移动。"""

    env = RobotSimEnv(show_viewer=False)

    try:
        # 重置环境并记录动作执行前的位置。
        initial_obs, _ = env.reset()
        initial_position = initial_obs["state"]["tcp_pose"][:3].copy()

        # 模拟上层算法给出 Z 正方向的最大归一化动作。
        action = np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        target_position = initial_position.copy()
        target_position[2] += env.config.action_scale[0]
        target_position = np.clip(target_position, env.config.pose_limit_low[:3], env.config.pose_limit_high[:3])

        # 执行一个控制周期，并读取最终 TCP 位置。
        obs, _, _, _, _ = env.step(action)
        final_position = obs["state"]["tcp_pose"][:3]
        position_change = final_position - initial_position

        print("initial tcp position:", initial_position)
        print("target tcp position:", target_position)
        print("final tcp position:", final_position)
        print("tcp position change:", position_change)

        assert final_position[2] > initial_position[2]
    finally:
        env.close()


def _run_long_positive_axis_motion(axis: int, axis_name: str) -> None:
    """连续30个周期输入小幅正向动作，检查指定轴的长时间运动。"""

    env = RobotSimEnv(show_viewer=False)
    num_steps = 30

    try:
        initial_obs, _ = env.reset()
        initial_position = initial_obs["state"]["tcp_pose"][:3].copy()

        # 每周期的理论目标增量为 0.1 × 0.02 = 0.002 米。
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action[axis] = 0.1

        positions = []
        target_positions = []
        for _ in range(num_steps):
            obs, _, terminated, truncated, _ = env.step(action)
            positions.append(obs["state"]["tcp_pose"][:3].copy())
            target_positions.append(env.data.mocap_pos[0].copy())
            assert not terminated
            assert not truncated

        positions = np.asarray(positions)
        target_positions = np.asarray(target_positions)
        final_position = positions[-1]
        position_change = final_position - initial_position

        print(f"{axis_name} initial position:", np.array2string(initial_position, precision=6, suppress_small=True))
        for step in (1, 10, 20, 30):
            print(f"{axis_name} step {step} position:", np.array2string(positions[step - 1], precision=6, suppress_small=True))
        print(f"{axis_name} final position:", np.array2string(final_position, precision=6, suppress_small=True))
        print(f"{axis_name} total change:", np.array2string(position_change, precision=6, suppress_small=True))

        # 检查长周期运动方向、数值稳定性和未触及工作空间边界。
        assert final_position[axis] > initial_position[axis]
        assert np.all(np.isfinite(positions))
        assert np.all(np.isfinite(target_positions))
        assert np.all(target_positions[:, axis] > env.config.pose_limit_low[axis])
        assert np.all(target_positions[:, axis] < env.config.pose_limit_high[axis])
        assert env.current_step == num_steps
        assert np.isclose(env.data.time, num_steps / env.config.hz)
    finally:
        env.close()


def test_long_positive_x_motion() -> None:
    _run_long_positive_axis_motion(axis=0, axis_name="X")


def test_long_positive_y_motion() -> None:
    _run_long_positive_axis_motion(axis=1, axis_name="Y")


def test_long_positive_z_motion() -> None:
    _run_long_positive_axis_motion(axis=2, axis_name="Z")


def test_positive_z_boundary_for_100_steps() -> None:
    """持续向 Z 正方向运动100周期，测试目标裁剪、边界稳定性和时间截断。"""

    env = RobotSimEnv(show_viewer=False)
    num_steps = 100
    print_steps = {1, 5, 10, 15, 20, 40, 70, 80, 100}

    try:
        initial_obs, _ = env.reset()
        initial_position = initial_obs["state"]["tcp_pose"][:3].copy()

        # 持续给出 Z 正方向最大归一化动作，直到达到工作空间上边界。
        action = np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        z_upper_bound = float(env.config.pose_limit_high[2])
        actual_positions = []
        target_positions = []
        first_clipped_step = None
        final_terminated = False
        final_truncated = False

        for step in range(1, num_steps + 1):
            obs, _, terminated, truncated, _ = env.step(action)
            actual_position = obs["state"]["tcp_pose"][:3].copy()
            target_position = env.data.mocap_pos[0].copy()
            actual_positions.append(actual_position)
            target_positions.append(target_position)

            # 第一次等于 Z 上界时，记录目标开始被安全范围裁剪的周期。
            if first_clipped_step is None and np.isclose(target_position[2], z_upper_bound):
                first_clipped_step = step

            if step in print_steps:
                print(f"step {step} actual tcp:", np.array2string(actual_position, precision=6, suppress_small=True))
                print(f"step {step} target tcp:", np.array2string(target_position, precision=6, suppress_small=True))

            assert not terminated
            assert truncated == (step == num_steps)
            final_terminated = terminated
            final_truncated = truncated

        actual_positions = np.asarray(actual_positions)
        target_positions = np.asarray(target_positions)
        max_actual_z = float(np.max(actual_positions[:, 2]))
        max_overshoot = max(0.0, max_actual_z - z_upper_bound)

        print("initial tcp:", np.array2string(initial_position, precision=6, suppress_small=True))
        print("first clipped step:", first_clipped_step)
        print(f"maximum actual Z: {max_actual_z:.6f} m")
        print(f"maximum Z overshoot: {max_overshoot * 1000.0:.6f} mm")
        print("final terminated:", final_terminated)
        print("final truncated:", final_truncated)
        print(f"simulation time: {env.data.time:.6f} s")

        # 目标不得越过边界，实际状态必须保持有限，第100步必须触发时间截断。
        assert first_clipped_step is not None
        assert np.all(target_positions[:, 2] <= z_upper_bound)
        assert np.all(np.isfinite(actual_positions))
        assert np.all(np.isfinite(target_positions))
        assert final_terminated is False
        assert final_truncated is True
        assert env.current_step == num_steps
        assert np.isclose(env.data.time, num_steps / env.config.hz)
    finally:
        env.close()


def test_move_to_fixed_point_and_hold() -> None:
    """三轴同时运动到固定空间点，并在到达后持续修正位置误差。"""

    env = RobotSimEnv(show_viewer=False)
    goal_position = np.asarray([0.5, 0.2, 0.1], dtype=np.float64)
    num_steps = 30
    print_steps = {1, 5, 10, 15, 20, 25, 30}
    reach_tolerance = 0.001  # 距离目标1毫米以内视为到达。

    try:
        initial_obs, _ = env.reset()
        initial_position = initial_obs["state"]["tcp_pose"][:3].copy()
        positions = []
        target_positions = []
        distances = []
        first_reached_step = None
        current_position = initial_position.copy()

        for step in range(1, num_steps + 1):
            position_error = goal_position - current_position

            # 保持三轴动作方向指向目标，并将最大位置增量限制为每周期0.02米。
            max_error = float(np.max(np.abs(position_error)))
            translation_action = position_error / max_error if max_error > env.config.action_scale[0] else position_error / env.config.action_scale[0]
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            action[:3] = translation_action.astype(np.float32)

            obs, _, terminated, truncated, _ = env.step(action)
            actual_position = obs["state"]["tcp_pose"][:3].copy()
            target_position = env.data.mocap_pos[0].copy()
            distance = float(np.linalg.norm(goal_position - actual_position))
            current_position = actual_position
            positions.append(actual_position)
            target_positions.append(target_position)
            distances.append(distance)

            if first_reached_step is None and distance <= reach_tolerance:
                first_reached_step = step

            if step in print_steps:
                print(f"step {step} actual tcp:", np.array2string(actual_position, precision=6, suppress_small=True))
                print(f"step {step} target tcp:", np.array2string(target_position, precision=6, suppress_small=True))
                print(f"step {step} distance to goal: {distance * 1000.0:.6f} mm")

            assert np.all(np.abs(action) <= 1.0)
            assert not terminated
            assert not truncated

        positions = np.asarray(positions)
        target_positions = np.asarray(target_positions)
        distances = np.asarray(distances)
        final_position = positions[-1]
        final_error = goal_position - final_position

        print("initial tcp:", np.array2string(initial_position, precision=6, suppress_small=True))
        print("goal tcp:", np.array2string(goal_position, precision=6, suppress_small=True))
        print("final tcp:", np.array2string(final_position, precision=9, suppress_small=True))
        print("final error:", np.array2string(final_error, precision=9, suppress_small=True))
        print(f"final distance: {distances[-1] * 1000.0:.6f} mm")
        print("first step within 1 mm:", first_reached_step)

        # 最终位置必须进入1毫米范围，且整个过程保持数值稳定并持续接近目标。
        assert first_reached_step is not None
        assert distances[-1] <= reach_tolerance
        assert distances[-1] < distances[0]
        assert np.all(np.isfinite(positions))
        assert np.all(np.isfinite(target_positions))
        assert env.current_step == num_steps
        assert np.isclose(env.data.time, num_steps / env.config.hz)
    finally:
        env.close()


def test_reset_after_motion() -> None:
    """机械臂运动并经历一个零动作周期后，检查 reset 能否恢复初始状态。"""

    env = RobotSimEnv(show_viewer=False)
    move_steps = 5

    try:
        # 保存第一次 reset 后的标准初始状态，作为第二次 reset 的比较基准。
        initial_obs, _ = env.reset()
        initial_tcp_pose = initial_obs["state"]["tcp_pose"].copy()
        initial_joint_positions = env.data.qpos[env.model.jnt_qposadr[env._joint_ids]].copy()
        print("initial tcp pose:", np.array2string(initial_tcp_pose, precision=9, suppress_small=True))

        # 连续运动5个周期，并记录每个周期结束后的 TCP 坐标。
        move_action = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for step in range(1, move_steps + 1):
            obs, _, terminated, truncated, _ = env.step(move_action)
            tcp_position = obs["state"]["tcp_pose"][:3]
            print(f"move step {step} tcp:", np.array2string(tcp_position, precision=9, suppress_small=True))
            assert not terminated
            assert not truncated

        # 输入一个零动作周期，让控制器停止继续增加目标并抑制当前速度。
        position_before_stop = obs["state"]["tcp_pose"][:3].copy()
        stop_action = np.zeros(env.action_space.shape, dtype=np.float32)
        stopped_obs, _, terminated, truncated, _ = env.step(stop_action)
        position_after_stop = stopped_obs["state"]["tcp_pose"][:3].copy()
        print("stop step tcp:", np.array2string(position_after_stop, precision=9, suppress_small=True))
        print("stop step change:", np.array2string(position_after_stop - position_before_stop, precision=9, suppress_small=True))
        assert not terminated
        assert not truncated

        # 再次 reset，并检查 TCP、关节角、夹爪目标、计步器和仿真时间。
        reset_obs, _ = env.reset()
        reset_tcp_pose = reset_obs["state"]["tcp_pose"].copy()
        reset_joint_positions = env.data.qpos[env.model.jnt_qposadr[env._joint_ids]].copy()
        print("reset tcp pose:", np.array2string(reset_tcp_pose, precision=9, suppress_small=True))
        print("reset joint positions:", np.array2string(reset_joint_positions, precision=9, suppress_small=True))
        print("reset gripper target:", reset_obs["state"]["gripper_target"])
        print("reset current step:", env.current_step)
        print(f"reset simulation time: {env.data.time:.6f} s")

        assert np.allclose(reset_tcp_pose, initial_tcp_pose)
        assert np.allclose(reset_joint_positions, initial_joint_positions)
        assert np.allclose(reset_joint_positions, env.config.home_position)
        assert reset_obs["state"]["gripper_target"][0] == OPEN_GRIPPER_TARGET
        assert env.current_step == 0
        assert np.isclose(env.data.time, 0.0)
    finally:
        env.close()


def test_gripper_close_hold_and_open() -> None:
    """测试夹爪闭合、持续保持以及重新张开。"""

    env = RobotSimEnv(show_viewer=False)

    try:
        obs, _ = env.reset()
        initial_gripper = float(obs["state"]["gripper_pose"][0])
        print(f"initial: gripper={initial_gripper:.6f}, target={env._gripper_target:.1f}, ctrl={env.data.ctrl[env._gripper_actuator_id]:.1f}")

        # 第1周期发出闭合命令，后续零动作应继续保持闭合目标。
        for step in range(1, 11):
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            action[6] = 1.0 if step == 1 else 0.0
            obs, _, terminated, truncated, _ = env.step(action)
            if step in (1, 5, 10):
                print(f"close step {step}: gripper={obs['state']['gripper_pose'][0]:.6f}, target={env._gripper_target:.1f}, ctrl={env.data.ctrl[env._gripper_actuator_id]:.1f}")
            assert env._gripper_target == CLOSED_GRIPPER_TARGET
            assert np.isclose(env.data.ctrl[env._gripper_actuator_id], MAX_GRIPPER_COMMAND)
            assert not terminated
            assert not truncated

        closed_gripper = float(obs["state"]["gripper_pose"][0])

        # 第1周期发出张开命令，后续零动作应继续保持张开目标。
        for step in range(1, 11):
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            action[6] = -1.0 if step == 1 else 0.0
            obs, _, terminated, truncated, _ = env.step(action)
            if step in (1, 5, 10):
                print(f"open step {step}: gripper={obs['state']['gripper_pose'][0]:.6f}, target={env._gripper_target:.1f}, ctrl={env.data.ctrl[env._gripper_actuator_id]:.1f}")
            assert env._gripper_target == OPEN_GRIPPER_TARGET
            assert np.isclose(env.data.ctrl[env._gripper_actuator_id], 0.0)
            assert not terminated
            assert not truncated

        opened_gripper = float(obs["state"]["gripper_pose"][0])
        assert closed_gripper > initial_gripper
        assert opened_gripper < closed_gripper
    finally:
        env.close()


def test_positive_x_rotation() -> None:
    """测试 TCP 能否绕 X 轴正方向旋转，并保持四元数有效。"""

    env = RobotSimEnv(show_viewer=False)

    try:
        initial_obs, _ = env.reset()
        initial_pose = initial_obs["state"]["tcp_pose"].copy()
        initial_rotation = Rotation.from_quat(initial_pose[3:])

        # X 轴动作1对应0.1弧度的目标旋转增量。
        action = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        target_rotation = Rotation.from_rotvec([env.config.action_scale[1], 0.0, 0.0]) * initial_rotation
        obs, _, terminated, truncated, _ = env.step(action)

        final_pose = obs["state"]["tcp_pose"]
        final_rotation = Rotation.from_quat(final_pose[3:])
        actual_rotvec = (final_rotation * initial_rotation.inv()).as_rotvec()
        position_drift = final_pose[:3] - initial_pose[:3]

        print("initial quaternion:", np.array2string(initial_pose[3:], precision=9, suppress_small=True))
        print("target quaternion:", np.array2string(target_rotation.as_quat(), precision=9, suppress_small=True))
        print("final quaternion:", np.array2string(final_pose[3:], precision=9, suppress_small=True))
        print("actual rotation vector:", np.array2string(actual_rotvec, precision=9, suppress_small=True))
        print(f"actual rotation angle: {np.linalg.norm(actual_rotvec):.9f} rad")
        print(f"quaternion norm: {np.linalg.norm(final_pose[3:]):.9f}")
        print("tcp position drift:", np.array2string(position_drift, precision=9, suppress_small=True))

        assert actual_rotvec[0] > 0.0
        assert (target_rotation * final_rotation.inv()).magnitude() < env.config.action_scale[1]
        assert np.isclose(np.linalg.norm(final_pose[3:]), 1.0)
        assert np.linalg.norm(position_drift) < 0.005
        assert not terminated
        assert not truncated
    finally:
        env.close()


def test_positive_y_rotation() -> None:
    """测试 TCP 能否绕 Y 轴正方向旋转，并保持四元数有效。"""

    env = RobotSimEnv(show_viewer=False)

    try:
        initial_obs, _ = env.reset()
        initial_pose = initial_obs["state"]["tcp_pose"].copy()
        initial_rotation = Rotation.from_quat(initial_pose[3:])

        # Y 轴动作1对应0.1弧度的目标旋转增量。
        action = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        target_rotation = Rotation.from_rotvec([0.0, env.config.action_scale[1], 0.0]) * initial_rotation
        obs, _, terminated, truncated, _ = env.step(action)

        final_pose = obs["state"]["tcp_pose"]
        final_rotation = Rotation.from_quat(final_pose[3:])
        actual_rotvec = (final_rotation * initial_rotation.inv()).as_rotvec()
        position_drift = final_pose[:3] - initial_pose[:3]

        print("initial quaternion:", np.array2string(initial_pose[3:], precision=9, suppress_small=True))
        print("target quaternion:", np.array2string(target_rotation.as_quat(), precision=9, suppress_small=True))
        print("final quaternion:", np.array2string(final_pose[3:], precision=9, suppress_small=True))
        print("actual rotation vector:", np.array2string(actual_rotvec, precision=9, suppress_small=True))
        print(f"actual rotation angle: {np.linalg.norm(actual_rotvec):.9f} rad")
        print(f"quaternion norm: {np.linalg.norm(final_pose[3:]):.9f}")
        print("tcp position drift:", np.array2string(position_drift, precision=9, suppress_small=True))

        assert actual_rotvec[1] > 0.0
        assert (target_rotation * final_rotation.inv()).magnitude() < env.config.action_scale[1]
        assert np.isclose(np.linalg.norm(final_pose[3:]), 1.0)
        assert np.linalg.norm(position_drift) < 0.005
        assert not terminated
        assert not truncated
    finally:
        env.close()


def test_positive_z_rotation() -> None:
    """测试 TCP 能否绕 Z 轴正方向旋转，并保持四元数有效。"""

    env = RobotSimEnv(show_viewer=False)

    try:
        initial_obs, _ = env.reset()
        initial_pose = initial_obs["state"]["tcp_pose"].copy()
        initial_rotation = Rotation.from_quat(initial_pose[3:])

        # Z 轴动作1对应0.1弧度的目标旋转增量。
        action = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        target_rotation = Rotation.from_rotvec([0.0, 0.0, env.config.action_scale[1]]) * initial_rotation
        obs, _, terminated, truncated, _ = env.step(action)

        final_pose = obs["state"]["tcp_pose"]
        final_rotation = Rotation.from_quat(final_pose[3:])
        actual_rotvec = (final_rotation * initial_rotation.inv()).as_rotvec()
        position_drift = final_pose[:3] - initial_pose[:3]

        print("initial quaternion:", np.array2string(initial_pose[3:], precision=9, suppress_small=True))
        print("target quaternion:", np.array2string(target_rotation.as_quat(), precision=9, suppress_small=True))
        print("final quaternion:", np.array2string(final_pose[3:], precision=9, suppress_small=True))
        print("actual rotation vector:", np.array2string(actual_rotvec, precision=9, suppress_small=True))
        print(f"actual rotation angle: {np.linalg.norm(actual_rotvec):.9f} rad")
        print(f"quaternion norm: {np.linalg.norm(final_pose[3:]):.9f}")
        print("tcp position drift:", np.array2string(position_drift, precision=9, suppress_small=True))

        assert actual_rotvec[2] > 0.0
        assert (target_rotation * final_rotation.inv()).magnitude() < env.config.action_scale[1]
        assert np.isclose(np.linalg.norm(final_pose[3:]), 1.0)
        assert np.linalg.norm(position_drift) < 0.005
        assert not terminated
        assert not truncated
    finally:
        env.close()


def test_close_is_idempotent_and_blocks_step() -> None:
    """测试环境可重复关闭，并阻止关闭后的继续交互。"""

    env = RobotSimEnv(show_viewer=False)
    action = np.zeros(env.action_space.shape, dtype=np.float32)

    try:
        env.reset()
        env.step(action)

        # 第一次关闭应释放图形资源并记录关闭状态。
        env.close()
        print("is_closed after first close:", env.is_closed)
        print("viewer after first close:", env._viewer)
        print("renderer after first close:", env._renderer)
        assert env.is_closed is True
        assert env._viewer is None
        assert env._renderer is None

        # 重复关闭不应再次释放资源或产生异常。
        env.close()
        assert env.is_closed is True

        # 环境关闭后不允许继续执行动作。
        with pytest.raises(RuntimeError, match="Cannot call step.*after close"):
            env.step(action)
    finally:
        env.close()
