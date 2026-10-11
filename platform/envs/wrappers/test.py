"""打开 Pick 仿真并用手柄测试遥操作，不采集或保存数据。

运行（在项目根目录）：
    /home/xinyu/Software/miniconda3/envs/gym_hil/bin/python platform/envs/wrappers/test.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLATFORM_ROOT = PROJECT_ROOT / "platform"
for import_root in (PROJECT_ROOT, PLATFORM_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from examples.sim_pick.config import EnvConfig, PickEnv
from envs.wrappers.intervention_wrapper import InterventionWrapper


class GamepadTestPickEnv(PickEnv):
    """保留抓取方块场景，但永不因成功、越界或步数而结束。"""

    def _is_success(self, observation) -> bool:
        del observation
        return False

    def _should_terminate(self, observation) -> bool:
        del observation
        return False


def main() -> None:
    config = EnvConfig()
    config.max_episode_length = 10**9
    base_env = GamepadTestPickEnv(
        config=config,
        show_viewer=True,
        random_block_position=True,
    )
    env = InterventionWrapper(base_env, input_device="gamepad")
    control_dt = 1.0 / float(config.hz)
    policy_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

    print("Gamepad test started. D-pad up toggles intervention; B resets; Ctrl+C exits.")
    print("This script does not collect data or reset after success/failure/rerecord signals.")

    try:
        env.reset(seed=0)
        reset_button = env.controller.controller_config["buttons"]["b"]
        b_was_pressed = False
        while True:
            step_started = time.monotonic()
            b_pressed = bool(env.controller.joystick.get_button(reset_button))
            if b_pressed and not b_was_pressed:
                env.reset()
                print("Environment reset by B.")
            b_was_pressed = b_pressed
            _, _, _, _, info = env.step(policy_action)
            if base_env._viewer is None:
                print("Viewer closed; exiting.")
                break
            if info.get("is_intervention", False):
                print("action:", np.round(info["intervene_action"], 3))
            remaining = control_dt - (time.monotonic() - step_started)
            if remaining > 0.0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
