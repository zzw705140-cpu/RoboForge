from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np


# 将 platform 加入模块搜索路径，使测试可以导入 envs 下的仿真环境和包装层。
PLATFORM_ROOT = Path(__file__).resolve().parents[3]
if str(PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(PLATFORM_ROOT))

from envs.sim.robot_sim_env1 import RobotSimEnv
from envs.wrappers.intervention_wrapper import InterventionWrapper


def create_keyboard_sim_env() -> InterventionWrapper:
    """创建带 MuJoCo 窗口和键盘人工接管功能的仿真环境。"""

    # 创建仿真基础环境。
    scene_path = PLATFORM_ROOT / "envs" / "sim" / "simulation" / "assets" / "scene.xml"
    base_env = RobotSimEnv(xml_path=scene_path, show_viewer=True)

    # 使用键盘人工接管包装层连接仿真基础环境。
    return InterventionWrapper(
        base_env,
        input_device="keyboard",
    )
# cd /home/xinyu/project/RoboForge
# python platform/envs/wrappers/test1/device_test.py
# 基础测试


######################################################################
#                        基础流程测试（人工操控）                        #
######################################################################

def _run_keyboard_episode(env: InterventionWrapper) -> str:
    """运行一个已重置的键盘控制回合，并返回结束原因。"""

    policy_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    control_period = 1.0 / env.unwrapped.config.hz

    print("Press Space to enable intervention, then use the movement keys.")
    print("Press Enter for success or ESC for failure to end the test.")

    while True:
        step_start = time.perf_counter()
        obs, _, terminated, truncated, info = env.step(policy_action)

        if info["is_intervention"]:
            print("intervene_action:", info.get("intervene_action"))
            print("tcp position:", obs["state"]["tcp_pose"][:3])

        if env.unwrapped._viewer is None:
            print("Viewer closed.")
            return "viewer_closed"
        if terminated or truncated:
            print("Episode ended. is_success:", info["is_success"])
            return "terminated" if terminated else "truncated"

        remaining_time = control_period - (time.perf_counter() - step_start)
        if remaining_time > 0:
            time.sleep(remaining_time)
# cd /home/xinyu/project/RoboForge
# python platform/envs/wrappers/test1/device_test.py
# 双回合测试重置环境


def test_keyboard_step(env: InterventionWrapper) -> str:
    """重置环境并运行一个键盘控制回合。"""

    env.reset()
    return _run_keyboard_episode(env)


def test_reset_and_second_episode(env: InterventionWrapper) -> None:
    """检查第一回合结束后能否重置并继续运行第二回合。"""

    print("Start episode 1.")
    first_end_reason = test_keyboard_step(env)
    if first_end_reason == "viewer_closed":
        return

    reset_obs, reset_info = env.reset()
    assert float(reset_obs["state"]["gripper_target"][0]) == -1.0
    assert reset_info["is_success"] is False
    assert env.controller.should_intervene() is False

    print("Reset between episodes passed.")
    print("gripper target:", reset_obs["state"]["gripper_target"])
    print("is_success:", reset_info["is_success"])
    print("intervention reset:", env.controller.should_intervene())
    print("Start episode 2. Press Space again to enable intervention.")
    _run_keyboard_episode(env)
# cd /home/xinyu/project/RoboForge
# conda activate gym_hil
# python platform/envs/wrappers/test1/device_test.py
# 单回合测试键盘


######################################################################
#                             综合测试                                #
######################################################################

def test_policy_action_passthrough(env: InterventionWrapper) -> None:
    """检查无人介入时，非零策略动作能否穿过包装层进入仿真环境。"""

    initial_obs, _ = env.reset()
    initial_x = float(initial_obs["state"]["tcp_pose"][0])
    policy_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    policy_action[0] = 1.0
    control_period = 1.0 / env.unwrapped.config.hz

    assert env.controller.should_intervene() is False
    print("Do not press Space during this test.")
    print("initial tcp x:", initial_x)

    final_obs = initial_obs
    for step in range(1, 11):
        step_start = time.perf_counter()
        final_obs, _, terminated, truncated, info = env.step(policy_action)

        assert info["is_intervention"] is False
        assert "intervene_action" not in info
        assert not terminated
        assert not truncated

        print(f"step {step}, tcp x:", float(final_obs["state"]["tcp_pose"][0]))

        remaining_time = control_period - (time.perf_counter() - step_start)
        if remaining_time > 0:
            time.sleep(remaining_time)

    final_x = float(final_obs["state"]["tcp_pose"][0])
    assert final_x > initial_x
    print("Policy action passthrough check passed.")
    print("tcp x change:", final_x - initial_x)


def test_policy_human_policy_switch(env: InterventionWrapper) -> None:
    """检查人工接管结束后，环境能否恢复执行持续的 X 正向策略动作。"""

    initial_obs, _ = env.reset()
    policy_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    policy_action[0] = 0.03
    control_period = 1.0 / env.unwrapped.config.hz
    env.unwrapped.config.max_episode_length = 300

    intervention_seen = False
    previous_intervention = False
    resume_start_x = None
    resume_steps = 0

    print("The policy is moving slowly in the positive X direction.")
    print("Press Space to intervene and control the robot manually.")
    print("Press Space again to return control to the policy.")
    print("initial tcp x:", float(initial_obs["state"]["tcp_pose"][0]))

    while True:
        step_start = time.perf_counter()
        obs, _, terminated, truncated, info = env.step(policy_action)
        intervened = bool(info["is_intervention"])
        current_x = float(obs["state"]["tcp_pose"][0])

        if intervened:
            assert "intervene_action" in info
        else:
            assert "intervene_action" not in info

        if intervened and not previous_intervention:
            intervention_seen = True
            resume_start_x = None
            resume_steps = 0
            print("Human intervention enabled at tcp x:", current_x)

        if intervention_seen and previous_intervention and not intervened:
            resume_start_x = current_x
            resume_steps = 0
            print("Human intervention disabled; policy resumed at tcp x:", current_x)

        if resume_start_x is not None and not intervened:
            resume_steps += 1
            if resume_steps >= 20:
                assert current_x > resume_start_x
                print("Policy resumed successfully.")
                print("tcp x change after resuming:", current_x - resume_start_x)
                break

        if env.unwrapped._viewer is None:
            print("Viewer closed before the test finished.")
            break
        if terminated or truncated:
            raise AssertionError("Episode ended before policy recovery was verified.")

        previous_intervention = intervened
        remaining_time = control_period - (time.perf_counter() - step_start)
        if remaining_time > 0:
            time.sleep(remaining_time)
# cd /home/xinyu/project/RoboForge
# python platform/envs/wrappers/test1/device_test.py
# 人类不断介入并停止


#######################################################################

if __name__ == "__main__":
    env = create_keyboard_sim_env()
    try:
        print("Keyboard intervention wrapper connected successfully.")
        test_policy_human_policy_switch(env)
    finally:
        env.close()
