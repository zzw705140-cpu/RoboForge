#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

############################################################
#         读取键盘或游戏手柄的原始输入，不直接操作Gym环境         #
############################################################

from __future__ import annotations

import json
from pathlib import Path

# 从JSON中读取游戏手柄的按键和摇杆映射
def load_controller_config(controller_name: str, config_path: str | None = None) -> dict:
    """
    Load controller configuration from a JSON file.

    Args:
        controller_name: Name of the controller to load.
        config_path: Path to the config file. If None, uses the package's default config.

    Returns:
        Dictionary containing the selected controller's configuration.
    """
    if config_path is None:            #如果调用函数时没有指定配置文件路径，就自动使用当前 Python 文件所在目录中的
        config_path = Path(__file__).resolve().parent / "controller_config.json"

    with open(config_path) as f:       #打开 JSON 配置文件，并将里面的内容读取成 Python 字典。
        config = json.load(f)

    # 检查当前连接的手柄名称是否存在于配置文件中，存在则读取专用手柄配置，不存在则使用default配置
    controller_config = config[controller_name] if controller_name in config else config["default"]

    if controller_name not in config:  #如果没有找到手柄专属配置，就在终端提示用户正在使用默认映射
        print(f"Controller {controller_name} not found in config. Using default configuration.")

    return controller_config


class InputController:
    """所有输入控制器的抽象基类，定义统一接口"""

    # 初始化移动步长，夹爪命令，人工接管和回合状态
    def __init__(self, x_step_size=0.01, y_step_size=0.01, z_step_size=0.01):
        """
        Initialize the controller.

        Args:
            x_step_size: Base movement step size in meters
            y_step_size: Base movement step size in meters
            z_step_size: Base movement step size in meters
        """
        self.x_step_size = x_step_size
        self.y_step_size = y_step_size
        self.z_step_size = z_step_size
        self.running = True                     # 输入控制器是否正在运行
        self.episode_end_status = None          # 当前回合的结束状态
        self.intervention_flag = False          # 是否启用人工接管
        self.open_gripper_command = False       # 是否收到打开夹爪的命令
        self.close_gripper_command = False      # 是否收到关闭夹爪的命令

    #启动输入设备
    def start(self):
        """Start the controller and initialize resources."""
        pass

    # 停止输入设备并释放资源
    def stop(self):
        """Stop the controller and release resources."""
        pass

    # 清除上一回合保存的输入状态
    def reset(self):
        """Reset the controller."""
        self.episode_end_status = None
        self.intervention_flag = False
        self.open_gripper_command = False
        self.close_gripper_command = False

    # 返回当前的三维移动量 dx、dy、dz
    def get_deltas(self):
        """Get the current movement deltas (dx, dy, dz) in meters."""
        return 0.0, 0.0, 0.0

    # 更新输入设备状态，主要由手柄控制器实现
    def update(self):
        """Update controller state - call this once per frame."""
        pass

    # 进入 with 语句时自动启动控制器，并返回控制对象
    def __enter__(self):
        """Support for use in 'with' statements."""
        self.start()
        return self

    # 退出 with 语句时自动关闭控制器并释放资源
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure resources are released when exiting 'with' block."""
        self.stop()

    # 读取人工标记的回合结束状态，并在读取后清除
    def get_episode_end_status(self):
        """
        Get the current episode end status.

        Returns:
            None if episode should continue, "success" or "failure" otherwise
        """
        status = self.episode_end_status     #将结束状态保存到局部变量里面
        self.episode_end_status = None       # Reset after reading
        return status

    # 判断当前是否启用了人工接管
    def should_intervene(self):
        """Return True if intervention flag was set."""
        return self.intervention_flag

    # 根据按键状态返回夹爪打开、关闭或无操作命令
    def gripper_command(self):
        """Return the current gripper command."""
        if self.open_gripper_command == self.close_gripper_command:
            return "no-op"                       #什么都没按时不操作夹爪
        elif self.open_gripper_command:
            return "open"                        #打开夹爪命令
        elif self.close_gripper_command:
            return "close"                       #关闭夹爪命令


######################################################################
#                            键盘                                    #
######################################################################

class KeyboardController(InputController):
    """负责监听键盘事件，并将按键状态转换为 X/Y/Z 方向的增量动作，以及 gripper 控制和 episode 结束状态"""

    # 创建键盘按键状态表并初始化监听器
    def __init__(self, x_step_size=0.01, y_step_size=0.01, z_step_size=0.01):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.key_states = {
            "forward_x": False,      # 是否按下 X 轴正方向键
            "backward_x": False,     # 是否按下 X 轴负方向键
            "forward_y": False,      # 是否按下 Y 轴正方向键
            "backward_y": False,     # 是否按下 Y 轴负方向键
            "forward_z": False,      # 是否按下 Z 轴正方向键
            "backward_z": False,     # 是否按下 Z 轴负方向键
            "success": False,        # 是否将当前回合标记为成功
            "failure": False,        # 是否将当前回合标记为失败
            "intervention": False,   # 是否启用人工接管
            "rerecord": False,       # 是否重新采集当前回合
        }
        self.listener = None

    # 使用 pynput 启动后台键盘监听线程
    def start(self):
        """Start the keyboard listener."""
        from pynput import keyboard

        # 按键按下时记录移动、夹爪、接管或回合结束命令
        def on_press(key):
            try:
                # XY movement
                if key == keyboard.Key.up:
                    self.key_states["forward_x"] = True
                elif key == keyboard.Key.down:
                    self.key_states["backward_x"] = True
                elif key == keyboard.Key.left:
                    self.key_states["forward_y"] = True
                elif key == keyboard.Key.right:
                    self.key_states["backward_y"] = True
                # Z movement
                elif key == keyboard.Key.shift:
                    self.key_states["backward_z"] = True
                elif key == keyboard.Key.shift_r:
                    self.key_states["forward_z"] = True
                # Gripper control
                # elif key == keyboard.Key.ctrl_r:
                elif key == keyboard.Key.alt_r:
                    self.open_gripper_command = True
                # elif key == keyboard.Key.ctrl_l:
                elif key == keyboard.Key.alt_l:
                    self.close_gripper_command = True
                # Special control keys
                elif key == keyboard.Key.enter:
                    self.key_states["success"] = True
                    self.episode_end_status = "success"
                elif key == keyboard.Key.esc:
                    self.key_states["failure"] = True
                    self.episode_end_status = "failure"
                elif key == keyboard.Key.space:
                    self.key_states["intervention"] = not self.key_states["intervention"]
                elif key == keyboard.Key.r:
                    self.key_states["rerecord"] = True
            except AttributeError:
                pass

        # 按键松开时清除对应的持续按键状态
        def on_release(key):
            try:
                if key == keyboard.Key.up:
                    self.key_states["forward_x"] = False
                elif key == keyboard.Key.down:
                    self.key_states["backward_x"] = False
                elif key == keyboard.Key.left:
                    self.key_states["forward_y"] = False
                elif key == keyboard.Key.right:
                    self.key_states["backward_y"] = False
                elif key == keyboard.Key.shift:
                    self.key_states["backward_z"] = False
                elif key == keyboard.Key.shift_r:
                    self.key_states["forward_z"] = False
                # elif key == keyboard.Key.ctrl_r:
                elif key == keyboard.Key.alt_r:
                    self.open_gripper_command = False
                # elif key == keyboard.Key.ctrl_l:
                elif key == keyboard.Key.alt_l:
                    self.close_gripper_command = False
            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()

        print("Keyboard controls:")
        print("  Arrow keys: Move in X-Y plane")
        print("  Shift and Shift_R: Move in Z axis")
        # print("  Right Ctrl and Left Ctrl: Open and close gripper")
        print("  Left Alt: Close gripper")
        print("  Right Alt: Open gripper")
        print("  Enter: End episode with SUCCESS")
        print("  Backspace: End episode with FAILURE")
        print("  Space: Start/Stop Intervention")
        print("  ESC: End episode with FAILURE")

    # 停止键盘监听线程
    def stop(self):
        """Stop the keyboard listener."""
        if self.listener and self.listener.is_alive():
            self.listener.stop()

    # 把方向键和 Shift 的状态转换成 XYZ 移动量
    def get_deltas(self):
        """Get the current movement deltas from keyboard state."""
        delta_x = delta_y = delta_z = 0.0

        if self.key_states["forward_x"]:
            delta_x += self.x_step_size
        if self.key_states["backward_x"]:
            delta_x -= self.x_step_size
        if self.key_states["forward_y"]:
            delta_y += self.y_step_size
        if self.key_states["backward_y"]:
            delta_y -= self.y_step_size
        if self.key_states["forward_z"]:
            delta_z += self.z_step_size
        if self.key_states["backward_z"]:
            delta_z -= self.z_step_size

        return delta_x, delta_y, delta_z

    # 判断是否按下了成功或失败按键
    def should_save(self):
        """Return True if Enter was pressed (save episode)."""
        return self.key_states["success"] or self.key_states["failure"]

    # 返回 Space 控制的人工接管状态
    def should_intervene(self):
        """Return True if intervention flag was set."""
        return self.key_states["intervention"]

    # 清除所有键盘按键状态
    def reset(self):
        """Reset the controller."""
        super().reset()
        for key in self.key_states:
            self.key_states[key] = False


###############################################################
#                          手柄                               #
###############################################################

class GamepadController(InputController):
    """Linux/Windows 下主要使用的游戏手柄实现，基于 pygame 监听手柄事件，并将按键状态转换为 X/Y/Z 方向的增量动作，以及 gripper 控制和 episode 结束状态"""

    # 初始化手柄移动步长、摇杆死区和配置路径
    def __init__(self, x_step_size=0.01, y_step_size=0.01, z_step_size=0.01, deadzone=0.1, config_path=None):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.joystick = None
        self.intervention_flag = False
        self.config_path = config_path
        self.controller_config = None

    # 初始化 pygame、连接手柄并加载按键配置
    def start(self):
        """Initialize pygame and the gamepad."""
        import pygame

        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            print("No gamepad detected. Please connect a gamepad and try again.")
            self.running = False
            return

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        joystick_name = self.joystick.get_name()
        print(f"Initialized gamepad: {joystick_name}")

        # Load controller configuration based on joystick name
        self.controller_config = load_controller_config(joystick_name, self.config_path)

        # Get button mappings from config
        buttons = self.controller_config.get("buttons", {})

        print("Gamepad controls:")
        print(f"  {buttons.get('rb', 'RB')} button: Intervention")
        print("  Left analog stick: Move in X-Y plane")
        print("  Right analog stick (vertical): Move in Z axis")
        print(f"  {buttons.get('lt', 'LT')} button: Close gripper")
        print(f"  {buttons.get('rt', 'RT')} button: Open gripper")
        print(f"  {buttons.get('b', 'B')}/Circle button: Exit")
        print(f"  {buttons.get('y', 'Y')}/Triangle button: End episode with SUCCESS")
        print(f"  {buttons.get('a', 'A')}/Cross button: End episode with FAILURE")
        print(f"  {buttons.get('x', 'X')}/Square button: Rerecord episode")

    # 关闭手柄并释放 pygame 资源
    def stop(self):
        """Clean up pygame resources."""
        import pygame

        if pygame.joystick.get_init():
            if self.joystick:
                self.joystick.quit()
            pygame.joystick.quit()
        pygame.quit()

    # 处理手柄按键事件并更新夹爪、接管和回合状态
    def update(self):
        """Process pygame events to get fresh gamepad readings."""
        import pygame

        # Get button mappings from config
        buttons = self.controller_config.get("buttons", {})
        y_button = buttons.get("y", 3)  # Default to 3 if not found
        a_button = buttons.get("a", 0)  # Default to 0 if not found (Logitech F310)
        x_button = buttons.get("x", 2)  # Default to 2 if not found (Logitech F310)
        lt_button = buttons.get("lt", 6)  # Default to 6 if not found
        rt_button = buttons.get("rt", 7)  # Default to 7 if not found
        rb_button = buttons.get("rb", 5)  # Default to 5 if not found

        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == y_button:
                    self.episode_end_status = "success"
                elif event.button == a_button:
                    self.episode_end_status = "failure"
                elif event.button == x_button:
                    self.episode_end_status = "rerecord_episode"
                elif event.button == lt_button:
                    self.close_gripper_command = True
                elif event.button == rt_button:
                    self.open_gripper_command = True

            # Reset episode status on button release
            elif event.type == pygame.JOYBUTTONUP:
                if event.button in [x_button, a_button, y_button]:
                    self.episode_end_status = None
                elif event.button == lt_button:
                    self.close_gripper_command = False
                elif event.button == rt_button:
                    self.open_gripper_command = False

            # Check for RB button for intervention flag
            if self.joystick.get_button(rb_button):
                self.intervention_flag = True
            else:
                self.intervention_flag = False

    # 读取摇杆并处理死区和方向反转，生成 XYZ 移动量
    def get_deltas(self):
        """Get the current movement deltas from gamepad state."""
        import pygame

        try:
            # Get axis mappings from config
            axes = self.controller_config.get("axes", {})
            axis_inversion = self.controller_config.get("axis_inversion", {})

            # Get axis indices from config (with defaults if not found)
            left_x_axis = axes.get("left_x", 0)
            left_y_axis = axes.get("left_y", 1)
            right_y_axis = axes.get("right_y", 3)

            # Get axis inversion settings (with defaults if not found)
            invert_left_x = axis_inversion.get("left_x", False)
            invert_left_y = axis_inversion.get("left_y", True)
            invert_right_y = axis_inversion.get("right_y", True)

            # Read joystick axes
            x_input = self.joystick.get_axis(left_x_axis)  # Left/Right
            y_input = self.joystick.get_axis(left_y_axis)  # Up/Down
            z_input = self.joystick.get_axis(right_y_axis)  # Up/Down for Z

            # Apply deadzone to avoid drift
            x_input = 0 if abs(x_input) < self.deadzone else x_input
            y_input = 0 if abs(y_input) < self.deadzone else y_input
            z_input = 0 if abs(z_input) < self.deadzone else z_input

            # Apply inversion if configured
            if invert_left_x:
                x_input = -x_input
            if invert_left_y:
                y_input = -y_input
            if invert_right_y:
                z_input = -z_input

            # Calculate deltas
            delta_x = y_input * self.y_step_size  # Forward/backward
            delta_y = x_input * self.x_step_size  # Left/right
            delta_z = z_input * self.z_step_size  # Up/down

            return delta_x, delta_y, delta_z

        except pygame.error:
            print("Error reading gamepad. Is it still connected?")
            return 0.0, 0.0, 0.0
