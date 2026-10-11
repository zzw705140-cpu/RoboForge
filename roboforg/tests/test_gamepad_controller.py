"""无实体手柄的输入映射、事件及局部旋转回归测试。"""
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "platform"))
from envs.wrappers.intervention_utils import GamepadController, load_controller_config
from envs.wrappers.intervention_wrapper import InterventionWrapper


class GamepadTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.axes = [0.0] * 6
        self.buttons = [0] * 8
        self.hat = (0, 0)
        pygame = types.SimpleNamespace(
            JOYBUTTONDOWN=1, JOYBUTTONUP=2, JOYHATMOTION=3, error=RuntimeError,
            event=types.SimpleNamespace(get=self.read_events),
        )
        self.module_patch = patch.dict(sys.modules, {"pygame": pygame})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.controller = GamepadController(x_step_size=1, y_step_size=1, z_step_size=1)
        self.controller.controller_config = load_controller_config("default")
        self.controller.joystick = types.SimpleNamespace(
            get_axis=lambda i: self.axes[i], get_button=lambda i: self.buttons[i],
            get_hat=lambda i: self.hat,
        )

    def read_events(self):
        events, self.events = self.events, []
        return events

    def send(self, kind, **fields):
        self.events.append(types.SimpleNamespace(type=kind, **fields))

    def test_short_episode_buttons_are_consumed_once(self):
        for button, status in [(3, "success"), (0, "failure"), (2, "rerecord_episode")]:
            self.send(1, button=button)
            self.send(2, button=button)
            self.controller.update()
            self.assertEqual(self.controller.get_episode_end_status(), status)
            self.assertIsNone(self.controller.get_episode_end_status())

    def test_toggle_edges_and_reset_while_held(self):
        for value, expected in [((0, 1), True), ((-1, 1), True), ((0, 0), True), ((0, 1), False)]:
            self.send(3, hat=0, value=value)
            self.controller.update()
            self.assertEqual(self.controller.should_intervene(), expected)
        self.controller.reset()
        self.send(3, hat=0, value=(0, 1))
        self.controller.update()
        self.assertFalse(self.controller.should_intervene())

    def test_translation_deadzone_and_diagonal(self):
        self.axes[:2] = [0.2, -0.2]
        np.testing.assert_allclose(self.controller.get_deltas(), [0, 0, 0])
        self.axes[:2] = [0.6, -0.6]
        np.testing.assert_allclose(self.controller.get_deltas(), [0.5, 0.5, 0])

    def test_rotation_sectors_and_gaps(self):
        for angle, expected in [(0, (0.5, 0, 0)), (90, (0, 0.5, 0)),
                                (180, (-0.5, 0, 0)), (270, (0, -0.5, 0)),
                                (45, (0, 0, 0)), (135, (0, 0, 0)),
                                (225, (0, 0, 0)), (315, (0, 0, 0))]:
            self.axes[2:] = [0.6 * math.cos(math.radians(angle)), -0.6 * math.sin(math.radians(angle))]
            np.testing.assert_allclose(self.controller.get_rotation_deltas(), expected, atol=1e-7)
        self.axes[2:] = [0.2, 0]
        np.testing.assert_allclose(self.controller.get_rotation_deltas(), [0, 0, 0])

    def test_shoulders_independent_of_events_and_b_unused(self):
        for lb, rb, expected in [(1, 0, -1), (0, 1, 1), (1, 1, 0), (0, 0, 0)]:
            self.buttons[4:6] = [lb, rb]
            self.controller.update()
            self.assertEqual(self.controller.get_rotation_deltas()[2], expected)
            self.assertFalse(self.controller.should_intervene())
        self.send(1, button=1)
        self.controller.update()
        self.assertIsNone(self.controller.get_episode_end_status())

    def test_xbox_series_axes_match_the_connected_controller(self):
        self.controller.controller_config = load_controller_config("Xbox Series X Controller")
        # LT 静止为 -1，不再被误当成右摇杆横轴；右杆使用 axis 3/4。
        self.axes[:] = [0, 0, -1, 0.6, 0, -1]
        np.testing.assert_allclose(self.controller.get_rotation_deltas(), [0.5, 0, 0])
        self.axes[0] = -0.6  # 实体手柄向左应输出机器人 Y 正方向。
        self.assertGreater(self.controller.get_deltas()[1], 0.0)
        self.axes[0] = 0.0
        self.assertEqual(self.controller.get_deltas()[2], 0.0)
        self.axes[2] = 1.0
        self.assertEqual(self.controller.get_deltas()[2], -1.0)
        self.axes[2] = -1.0
        self.axes[5] = 1.0
        self.assertEqual(self.controller.get_deltas()[2], 1.0)
        self.send(3, hat=0, value=(-1, 0))
        self.controller.update()
        self.assertEqual(self.controller.gripper_command(), "close")
        self.send(3, hat=0, value=(1, 0))
        self.controller.update()
        self.assertEqual(self.controller.gripper_command(), "open")

    def test_wrapper_rotates_local_command_and_preserves_episode_flags(self):
        class Env(gym.Env):
            action_space = gym.spaces.Box(-1, 1, (7,), dtype=np.float32)
            observation_space = gym.spaces.Dict({})
            def reset(self, **kwargs):
                self.pose = np.r_[np.zeros(3), Rotation.from_euler("z", 90, degrees=True).as_quat()]
                return {"state": {"tcp_pose": self.pose}}, {}
            def step(self, action):
                self.action = action
                return {"state": {"tcp_pose": self.pose}}, 0.0, False, False, {}

        env = Env()
        with patch.object(InterventionWrapper, "_create_controller", return_value=self.controller):
            wrapper = InterventionWrapper(env, input_device="gamepad")
        wrapper.reset()
        self.axes[2] = 1.0
        self.send(3, hat=0, value=(0, 1))
        _, _, _, _, info = wrapper.step(np.zeros(7))
        np.testing.assert_allclose(env.action[3:6], [0, 1, 0], atol=1e-6)
        self.assertTrue(info["is_takeover_active"])
        for button, success, rerecord in [(3, True, False), (0, False, False), (2, False, True)]:
            self.send(1, button=button)
            self.send(2, button=button)
            _, _, terminated, _, info = wrapper.step(np.zeros(7))
            self.assertTrue(terminated)
            self.assertEqual(info["is_success"], success)
            self.assertEqual(info["rerecord_episode"], rerecord)
        wrapper.reset()
        self.assertFalse(self.controller.should_intervene())


if __name__ == "__main__":
    unittest.main()
