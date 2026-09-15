"""Human-intervention wrapper shared by simulated and real robot environments.

The wrapped environment must use the normalized single-arm action convention:

``[dx, dy, dz, dRx, dRy, dRz, gripper]`` where translation and rotation
independently lie in unit L2 balls.  The gripper lies in ``[-1, 1]``.

The gripper convention is monotonic: ``-1`` is open and ``+1`` is closed.
The last gripper target persists after a button is released, but persistence
does not by itself count as human intervention.

When a human command replaces the policy command, this wrapper writes the
executed action to ``info["intervene_action"]``. HIL-SERL's actor uses that
exact key to store intervention transitions in its demonstration buffer.
"""
#################################################################
#        把输入设备产生的信号组成标准动作，决定执行策略动作还是人工动作   #
#################################################################

from __future__ import annotations

from typing import Literal

import gymnasium as gym
import numpy as np

from .action_utils import project_action_to_unit_balls


InputDevice = Literal["keyboard", "gamepad", "spacemouse"]
OPEN_GRIPPER_TARGET   = -1.0
CLOSED_GRIPPER_TARGET = 1.0

"""单机械臂遥操作"""

# 将键盘、手柄或 SpaceMouse 的人工动作接入 Gym 环境，并在接管时替换策略动作
class InterventionWrapper(gym.Wrapper):
    """
    人工干预，则使用人工干预动作，而非 policy 输出动作

    注意, InterventionWrapper 最好放在 GripperCloseEnv 外层，同时相对于 RelativeFrameWrapper 在内层
    """

    # 检查动作空间和控制参数，并创建对应的人工输入控制器
    def __init__(
        self,
        env: gym.Env,
        *,
        input_device: InputDevice = "spacemouse",
        intervention_threshold:   float = 1e-3,             # 判断 spacemouse 是否真的发生人工操作
        translation_action_scale: float = 1.0,              # 缩放 “平移部分”
        rotation_action_scale:    float = 1.0,              # 缩放 “旋转部分”
        use_gripper: bool = True,
        controller_config_path: str | None = None,
        action_indices: tuple[int, ...] | None = None,      # 限制人工只能控制指定的动作维度
    ) -> None:
        """初始化"""

        super().__init__(env)

        # 输入变量检查
        if not isinstance(env.action_space, gym.spaces.Box):                      #检查底层环境是否使用连续动作空间
            raise TypeError("InterventionWrapper requires a Box action space")
        if env.action_space.shape not in ((6,), (7,)):                            #检查动作必须是6维或7维
            raise ValueError(f"InterventionWrapper requires action shape (6,) or (7,), got {env.action_space.shape}")
        if input_device not in ("keyboard", "gamepad", "spacemouse"):             #检查输入设备名称是否属于支持的三种设备
            raise ValueError(f"unsupported input_device: {input_device!r}")
        if intervention_threshold < 0:                                            #确保人工介入判断阈值不是负数
            raise ValueError("intervention_threshold must be non-negative")
        if translation_action_scale < 0 or rotation_action_scale < 0:             #确保平移和旋转缩放系数不是负数，防止意外翻转操作
            raise ValueError("action scales must be non-negative")

        self.input_device = input_device
        self.intervention_threshold = float(intervention_threshold)
        self.translation_action_scale = float(translation_action_scale)
        self.rotation_action_scale = float(rotation_action_scale)
        self.action_dim = env.action_space.shape[0]                          #获取底层环境的动作维度，是多少就是多少
        self.use_gripper = bool(use_gripper and self.action_dim == 7)        #同时满足两个条件才启用夹爪控制
        self.action_indices = action_indices                                 #保存允许人工控制的动作维度

        # 该值只表示人工动作需要携带的当前夹爪目标，不表示人仍在接管
        self._gripper_target = OPEN_GRIPPER_TARGET

        # 是否限制人工只能控制指定的动作维度
        # 确保人工控制指定的动作下标都处于合法的范围，防止后面访问动作数组时越界
        if action_indices is not None:
            invalid = set(action_indices) - set(range(self.action_dim))
            if invalid:
                raise ValueError(f"invalid action indices: {sorted(invalid)}")

        # 创建输入设备对象
        self.controller = self._create_controller(controller_config_path)   

    # 根据 input_device 创建并启动键盘、手柄或 SpaceMouse 控制器
    def _create_controller(self, controller_config_path: str | None):
        """根据 input_device 创建对应的输入设备对象"""

        if self.input_device == "spacemouse":
            # spacemouse 创建时内部自动 “启动后台读取进程”
            from spacemouse.spacemouse_expert import SpaceMouseExpert
            return SpaceMouseExpert()

        # 键盘/手柄输入是 [-1, 1] 的归一化动作，需要进行缩放
        controller_kwargs = {
            "x_step_size": self.translation_action_scale,
            "y_step_size": self.translation_action_scale,
            "z_step_size": self.translation_action_scale,
        }

        if self.input_device == "keyboard":
            from .intervention_utils import KeyboardController
            controller = KeyboardController(**controller_kwargs)
        else:
            from .intervention_utils import GamepadController
            controller = GamepadController(**controller_kwargs, config_path=controller_config_path)

        controller.start()
        return controller

    # 将包装层保存的夹爪目标与底层环境实际执行的目标保持一致
    def _sync_gripper_target(self, observation=None) -> None:
        """Synchronize the carried human target with the target executed by the environment."""

        target = None
        if isinstance(observation, dict):
            state = observation.get("state")
            if isinstance(state, dict) and "gripper_target" in state:
                target = float(np.asarray(state["gripper_target"]).reshape(-1)[0])

        # InterventionWrapper 通常直接包装基础环境；该回退也兼容不暴露原始 Dict observation 的包装顺序。
        if target is None:
            target = getattr(self.env.unwrapped, "_gripper_target", None)

        if target is not None:
            self._gripper_target = (OPEN_GRIPPER_TARGET if float(target) < 0.0 else CLOSED_GRIPPER_TARGET)

    # 读取人工输入并返回接管状态、标准动作和人工指定的回合状态
    def _read_human_input(self) -> tuple[bool, np.ndarray, str | None]:
        """读取当前设备, 并返回接管状态、7 维动作和 episode 状态"""

        if self.input_device == "spacemouse":
            # 从 SpaceMouse 获取六维人工动作，如 [dx, dy, dz, dRx, dRy, dRz]
            expert_action, buttons = self.controller.get_action()
            expert_action = np.asarray(expert_action, dtype=np.float32).reshape(-1)
            if expert_action.size < 6:
                raise RuntimeError(f"SpaceMouse returned {expert_action.size} axes; " "expected at least 6")

            # 计算人工控制指令
            action = np.zeros(self.action_dim, dtype=np.float32)
            action[:3]  = expert_action[:3]  * self.translation_action_scale
            action[3:6] = expert_action[3:6] * self.rotation_action_scale
            if self.use_gripper:
                action[6] = self._gripper_target

            # 约定：第一个按钮关闭夹爪，第二个按钮打开夹爪
            buttons = list(buttons)
            close_pressed = bool(buttons[0]) if len(buttons) > 0 else False
            open_pressed  = bool(buttons[1]) if len(buttons) > 1 else False
            gripper_commanded = self.use_gripper and (close_pressed != open_pressed)    # 判断是否同时按下两个夹爪
            if self.use_gripper:
                if close_pressed and not open_pressed:
                    self._gripper_target = CLOSED_GRIPPER_TARGET
                elif open_pressed and not close_pressed:
                    self._gripper_target = OPEN_GRIPPER_TARGET
                action[6] = self._gripper_target

            # 操作是否手动结束当前 episode
            episode_status = None
        else:
            # KeyboardController 和 GamepadController 具有相同的控制接口

            # 更新控制器状态
            self.controller.update()

            action = np.zeros(self.action_dim, dtype=np.float32)
            if self.use_gripper:
                action[6] = self._gripper_target

            # 获取 XYZ 方向归一化动作，并写入人工动作
            dx, dy, dz = self.controller.get_deltas()
            action[:3] = np.asarray([dx, dy, dz], dtype=np.float32)

            gripper_command = "no-op"
            if self.use_gripper:
                gripper_command = self.controller.gripper_command()
                if gripper_command == "close":
                    self._gripper_target = CLOSED_GRIPPER_TARGET
                elif gripper_command == "open":
                    self._gripper_target = OPEN_GRIPPER_TARGET
                action[6] = self._gripper_target

            # 夹爪按钮本身是一次人工接管；松开后仅保留目标，不继续计为接管
            gripper_commanded = self.use_gripper and gripper_command in ("close", "open")
            controller_intervened = bool(self.controller.should_intervene())

            # 操作是否手动结束当前 episode
            episode_status = self.controller.get_episode_end_status()

        # 可选地只允许人工控制指定动作维度，其他维度动作全部清零
        if self.action_indices is not None:
            filtered = np.zeros_like(action)
            indices = list(self.action_indices)
            filtered[indices] = action[indices]

            # 第七维携带当前目标，避免只干预机械臂时把夹爪标签写成无意义的 0
            if self.use_gripper:
                filtered[6] = self._gripper_target
            action = filtered

        # 人类动作在 base 坐标系执行。平移和旋转分别投影到单位球后，旋转到任意坐标系仍保持合法；第七维夹爪由同一函数独立裁剪到 [-1, 1]
        action = project_action_to_unit_balls(action).astype(self.action_space.dtype)

        # SpaceMouse 以投影且过滤后的六维运动判断是否接管；键盘/手柄保留其显式接管开关语义。持续的夹爪目标本身不算接管，只有本步按钮命令才算
        if self.input_device == "spacemouse":
            intervened = bool(np.linalg.norm(action[:6]) > self.intervention_threshold or gripper_commanded)
        else:
            intervened = bool(controller_intervened or gripper_commanded)

        return intervened, action, episode_status

    # 接管时执行人工动作，否则执行策略动作，再调用底层环境的 step
    def step(self, action: np.ndarray):
        """覆盖原 step"""

        # 检查策略输出动作是否合法
        policy_action = np.asarray(action, dtype=self.action_space.dtype)
        if policy_action.shape != self.action_space.shape:
            raise ValueError(f"action must have shape {self.action_space.shape}, got {policy_action.shape}")

        # 读取人工动作
        intervened, human_action, episode_status = self._read_human_input()

        # 如果人工干预，则执行动作采用人工
        executed_action = human_action if intervened else policy_action

        # 执行原环境 step
        obs, reward, terminated, truncated, info = self.env.step(executed_action)
        # 当底层环境执行完动作后，让人工层控制层保存的夹爪目标与底层环境保持一致，所以同步的是目标状态
        self._sync_gripper_target(obs)
        info = dict(info)

        # 如果干预，记录信息
        if intervened:
            info["intervene_action"] = executed_action.copy()
        else:
            info.pop("intervene_action", None)
        info["is_intervention"] = intervened

        # 判断操作人是否按下: success，failure，rerecord_episode（当前轨迹作废，重新录制）
        manual_end = episode_status in ("success", "failure", "rerecord_episode")
        if episode_status == "success":
            reward = 1.0
            info["is_success"] = True
        elif episode_status in ("failure", "rerecord_episode"):
            info["is_success"] = False

        if episode_status == "rerecord_episode":
            info["rerecord_episode"] = True
        else:
            info["rerecord_episode"] = False

        terminated = bool(terminated or manual_end)
        return obs, reward, terminated, bool(truncated), info

    # 重置输入控制器和底层环境，并重新同步夹爪目标
    def reset(self, **kwargs):
        """重置环境"""

        self._gripper_target = OPEN_GRIPPER_TARGET

        # 重置键盘或手柄控制器内部保存的输入状态，防止上一个 episode 的按键状态延续到下一个 episode
        if self.input_device != "spacemouse":
            self.controller.reset()

        # 重置环境
        obs, info = self.env.reset(**kwargs)
        self._sync_gripper_target(obs)
        info = dict(info)
        info["is_success"] = False

        return obs, info

    # 关闭输入控制器和底层环境并释放相关资源
    def close(self) -> None:
        """释放输入设备和底层环境资源"""

        # SpaceMouseExpert 提供 close()
        close = getattr(self.controller, "close", None)

        # KeyboardController 和 GamepadController 提供 stop()
        stop  = getattr(self.controller, "stop",  None)

        # 关闭
        if callable(close):
            close()
        elif callable(stop):
            stop()

        # 调用主环境，释放资源
        self.env.close()

"""双机械臂遥操作"""
