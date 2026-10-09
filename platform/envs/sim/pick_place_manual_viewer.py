"""打开 PickPlace 场景，并用键盘直接操控 Panda 机械臂。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np


# 将 platform 加入模块搜索路径，使脚本可从项目根目录直接运行。
PLATFORM_ROOT = Path(__file__).resolve().parents[2]
if str(PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(PLATFORM_ROOT))

from envs.sim.robot_sim_env1 import RobotSimEnv, RobotSimEnvConfig
from envs.wrappers.intervention_wrapper import InterventionWrapper


def create_manual_environment() -> InterventionWrapper:
    """创建带 MuJoCo 窗口和键盘接管功能的 PickPlace 场景环境。"""

    # 单纯查看与人工操作不应受训练回合长度限制。
    config = RobotSimEnvConfig()
    config.max_episode_length = 100_000
    scene_path = PLATFORM_ROOT / "assets" / "pick_place_scene.xml"
    base_env = RobotSimEnv(config=config, xml_path=scene_path, show_viewer=True)

    # 使用现有键盘控制器，动作直接在机器人基座坐标系中执行。
    return InterventionWrapper(base_env, input_device="keyboard")


def run_manual_viewer(env: InterventionWrapper) -> None:
    """以环境控制频率持续执行键盘动作，直到用户关闭 MuJoCo 窗口。"""

    # 未人工接管时执行零动作；夹爪初始状态由底层环境保持张开。
    policy_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    control_period = 1.0 / float(env.unwrapped.config.hz)
    env.reset()

    print("PickPlace 场景已加载。关闭 MuJoCo 窗口即可结束程序。")
    print("先按一次 Space 开启接管，再按方向键移动；再次按 Space 暂停接管。")

    while True:
        # 控制频率保持为环境设置的 10 Hz，便于人工观察机械臂运动。
        step_start = time.perf_counter()
        _, _, terminated, truncated, info = env.step(policy_action)

        # m、Esc 和 r 是既有键盘控制器的回合控制键；此处自动恢复观察场景。
        if terminated or truncated:
            print(f"回合结束，is_success={info['is_success']}；已自动重置场景。")
            env.reset()

        # 用户关闭窗口后，底层环境会清空 viewer 句柄，循环随即退出。
        if env.unwrapped._viewer is None:
            return

        remaining_time = control_period - (time.perf_counter() - step_start)
        if remaining_time > 0.0:
            time.sleep(remaining_time)


def main() -> None:
    """启动场景查看器，并在退出时释放键盘监听器和 MuJoCo 资源。"""

    env = create_manual_environment()
    try:
        run_manual_viewer(env)
    finally:
        env.close()


if __name__ == "__main__":
    main()

#conda activate gym_hil
#cd ~/project/RoboForge
#python platform/envs/sim/pick_place_manual_viewer.py