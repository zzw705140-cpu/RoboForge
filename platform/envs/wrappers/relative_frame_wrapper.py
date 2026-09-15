"""Robot observations and actions expressed in relative/end-effector frames."""

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from .action_utils import validate_action_in_unit_balls

#######################################################
#                    坐标徐转换                         #
#######################################################
#上层使用相对坐标，底层环境使用机器人基准坐标


def _rotation_transform(tcp_pose):
    """
    计算 "同时旋转平移量和旋转量" 的 6×6 变换矩阵
    ```
    [ R_base_tcp      0      ]
    [     0       R_base_tcp ]
    ```
    """

    rotation  = Rotation.from_quat(tcp_pose[3:]).as_matrix()
    transform = np.zeros((6, 6), dtype=np.float64)
    transform[:3, :3] = rotation
    transform[3:, 3:] = rotation

    return transform

def _homogeneous_transform(tcp_pose):
    """计算 4×4 位姿变换矩阵"""

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(tcp_pose[3:]).as_matrix()
    transform[:3, 3]  = tcp_pose[:3]

    return transform


class RelativeFrameWrapper(gym.Wrapper):
    """
    神经网络 “输入-输出” 坐标系调整

    输入：
    tcp_vel     → 当前末端坐标系 (body frame), 与 action 保持一直
    tcp_pose    → reset 初始末端坐标系 (relative reset frame)

    输出：
    action      → 当前末端坐标系 (body frame)，平移和旋转分别位于单位二范数球

    注意
    1. RelativeFrameWrapper 必须放在 Quat2EulerWrapper 或 Quat2R2Wrapper 的内层，因为它需要读取原始的 ``xyz + quaternion`` 位姿
    2. RelativeFrameWrapper 必须放在 InterventionWrapper 的外部
    """

    def __init__(self, env, include_relative_pose=True):
        super().__init__(env)

        # 动作空间检查
        if not isinstance(env.action_space, gym.spaces.Box):
            raise TypeError("RelativeFrameWrapper requires a Box action space.")
        if env.action_space.shape not in ((6,), (7,)):
            raise ValueError("RelativeFrameWrapper requires action shape (6,) or (7,), " f"got {env.action_space.shape}.")

        # 状态空间检查
        state_space = env.observation_space["state"]
        if state_space["tcp_pose"].shape != (7,):
            raise ValueError("RelativeFrameWrapper requires tcp_pose with shape (7,) using xyz + xyzw quaternion.")
        if state_space["tcp_vel"].shape != (6,):
            raise ValueError("RelativeFrameWrapper requires tcp_vel with shape (6,).")

        # include_relative_pose = true,  则 tcp_pose 转化为 “相对于 reset 初始位姿”
        # include_relative_pose = false, 则 tcp_pose 仍然保持在 base 下
        self.include_relative_pose = bool(include_relative_pose)

        # 存 "当前末端" 到 base 的 6×6 旋转变换矩阵
        self._action_transform = np.eye(6, dtype=np.float64)

        # 存 "初始末端" 到 base 的 4×4 位姿变换矩阵 的 "逆矩阵"
        self._initial_pose_inverse = np.eye(4, dtype=np.float64)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        # 获取末端执行器在 base 下的位姿
        tcp_pose = self._get_tcp_pose(observation)

        self._action_transform = _rotation_transform(tcp_pose)
        if self.include_relative_pose:
            self._initial_pose_inverse = np.linalg.inv(_homogeneous_transform(tcp_pose))

        return self._transform_observation(observation), info

    def step(self, action):
        
        # 将神经网络输出相对于 “当前末端坐标系” 位姿增量转换至 base 下
        base_frame_action = self.transform_action(action)

        # 执行 step 
        observation, reward, terminated, truncated, info = self.env.step(base_frame_action)

        # InterventionWrapper 位于该 wrapper 内层时，返回的是基坐标系动作；对外转换回策略所使用的末端坐标系
        info = dict(info)
        if "intervene_action" in info:
            info["intervene_action"] = self.inverse_transform_action(info["intervene_action"])

        # 更新 observation
        tcp_pose = self._get_tcp_pose(observation)
        self._action_transform = _rotation_transform(tcp_pose)
        observation = self._transform_observation(observation)

        return observation, reward, terminated, truncated, info

    def transform_action(self, action):
        """将神经网络输出相对于 “当前末端坐标系” 位姿增量转换至 base 下"""

        action = np.asarray(action, dtype=self.action_space.dtype).copy()
        if action.shape != self.action_space.shape:
            raise ValueError(f"action must have shape {self.action_space.shape}, " f"got {action.shape}.")

        # 检查上游给出的策略动作，"位置分量" 和 "动作分量" 是否都在单位球内
        validate_action_in_unit_balls(action)

        # "末端坐标系" 下的 action → "base" 下的 action
        action[:6] = self._action_transform @ action[:6]
        return action

    def inverse_transform_action(self, action):
        """将机器人基坐标系动作转换回末端坐标系"""

        action = np.asarray(action, dtype=self.action_space.dtype).copy()
        if action.shape != self.action_space.shape:
            raise ValueError(f"action must have shape {self.action_space.shape}, got {action.shape}.")
        action[:6] = self._action_transform.T @ action[:6]
        return action

    def _transform_observation(self, observation):
        """
        更新 tcp_pose, tcp_vel 所服从坐标系
        1. tcp_pose → reset 初始末端坐标系 (relative reset frame)
        2. tcp_vel  → 当前末端坐标系 (body frame)
        """

        # 计算 base 下的当前 tcp_pose 和 tcp_vel
        tcp_pose = self._get_tcp_pose(observation)
        tcp_vel  = np.asarray(observation["state"]["tcp_vel"], dtype=np.float64)

        # 将 tcp_vel 从 base 转换到 "当前末端坐标系"
        observation["state"]["tcp_vel"] = (self._action_transform.T @ tcp_vel).astype(np.float32)

        # 将 tcp_pose 从 base 转换到 "初始末端坐标系"
        if self.include_relative_pose:
            relative_transform = (self._initial_pose_inverse @ _homogeneous_transform(tcp_pose))
            relative_pose = np.concatenate(
                (
                    relative_transform[:3, 3],
                    Rotation.from_matrix(relative_transform[:3, :3]).as_quat(),
                )
            )
            observation["state"]["tcp_pose"] = relative_pose.astype(np.float32)

        return observation

    @staticmethod
    def _get_tcp_pose(observation):
        """获取末端执行器在 base 下的位姿"""

        tcp_pose = np.asarray(observation["state"]["tcp_pose"], dtype=np.float64)

        if tcp_pose.shape != (7,):
            raise ValueError(f"tcp_pose must have shape (7,), got {tcp_pose.shape}.")
        if np.linalg.norm(tcp_pose[3:]) < 1e-8:
            raise ValueError("tcp_pose contains a zero quaternion.")

        return tcp_pose
