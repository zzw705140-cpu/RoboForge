"""Observation wrappers"""

import gymnasium as gym
import numpy as np
from gymnasium.spaces import flatten, flatten_space
from scipy.spatial.transform import Rotation

##################################################################
#                        观测适配层                                #
##################################################################
#负责将环境返回的观测转换成上层算法更方便使用的格式


# 把 TCP 的“位置 + 四元数（7维）”转换成“位置 + 欧拉角（6维）”
class Quat2EulerWrapper(gym.ObservationWrapper):
    """Convert TCP pose from xyz + quaternion to xyz + Euler angles."""

    def __init__(self, env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        self.observation_space["state"]["tcp_pose"] = gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)

    def observation(self, observation):
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate((tcp_pose[:3], Rotation.from_quat(tcp_pose[3:]).as_euler("xyz"))).astype(np.float32)
        return observation


# 把四元数转换为旋转矩阵的前两列，最终得到 9 维 TCP 位姿
class Quat2R2Wrapper(gym.ObservationWrapper):
    """Convert TCP pose to xyz + first two columns of its rotation matrix."""

    def __init__(self, env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        self.observation_space["state"]["tcp_pose"] = gym.spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32)

    def observation(self, observation):
        tcp_pose = observation["state"]["tcp_pose"]
        rotation = Rotation.from_quat(tcp_pose[3:]).as_matrix()
        observation["state"]["tcp_pose"] = np.concatenate((tcp_pose[:3], rotation[:, :2].flatten())).astype(np.float32)
        return observation


# 把字典中的多个机器人状态拼接成一个一维向量，同时保留图像观测（方便直接神经网络输入）
class FlattenStateObservationWrapper(gym.ObservationWrapper):
    """将选定的机器人状态拼成一维向量，并将相机图像提到最外层

    示例: 
        输入：
        ```
            {
                "state":  {"tcp_pose": ..., ...},
                "images": {"front": ..., "wrist": ...}
            }
        ```
        输出：
        ```
            {
                "state": 一维向量,
                "front": ...,
                "wrist": ...
            }
        ```
    注意, FlattenStateObservationWrapper 必须放到 Quat2EulerWrapper/Quat2R2Wrapper 的外层
    """

    def __init__(self, env, proprio_keys=None):
        super().__init__(env)

        # proprio_keys 用来指定哪些机器人状态需要拼接进神经网络的 state 向量。
        # gripper_pose 是实际连续开度 [0, 1]，gripper_target 是迟滞状态机的持续目标 {-1, +1}；
        # 两者都进入观测可避免夹爪控制目标成为网络不可见的隐藏状态。

        if proprio_keys is None:
            proprio_keys = list(self.env.observation_space["state"].keys())
        self.proprio_keys = list(proprio_keys)

        """
        假定原始观测空间是：
        {
            "state": {
                "tcp_pose": Box(shape=(9,)),
                "tcp_vel": Box(shape=(6,)),
                "gripper_pose": Box(shape=(1,)),
                "gripper_target": Box(shape=(1,)),
                "tcp_force": Box(shape=(3,)),
                "tcp_torque": Box(shape=(3,)),
            },
            "images": {
                "front": Box(shape=(128, 128, 3)),
                "wrist": Box(shape=(128, 128, 3)),
            },
        }
        """

        """
        如果说: self.proprio_keys = ["tcp_pose", "tcp_vel", "gripper_pose", "gripper_target"], 则 self.proprio_space 就是：
        {
            "tcp_pose": Box(shape=(9,)),
            "tcp_vel": Box(shape=(6,)),
            "gripper_pose": Box(shape=(1,)),
            "gripper_target": Box(shape=(1,)),
        }
        后续的: flatten_space(self.proprio_space) 会将这些字段拼成 Box(shape=(17,))
        """
        self.proprio_space = gym.spaces.Dict(
            {
                key: self.env.observation_space["state"][key] for key in self.proprio_keys
            }
        )

        """
        取出 images 中每个相机的空间定义, 即 image_spaces 是
        {
            "front": Box(shape=(128, 128, 3)),
            "wrist": Box(shape=(128, 128, 3)),
        }
        """
        image_spaces = {}
        if "images" in self.env.observation_space.spaces:
            image_spaces = dict(self.env.observation_space["images"].spaces)

        """
        最终得到观测空间:
        {
            "state": Box(shape=(16,)),
            "front": Box(shape=(128, 128, 3)),
            "wrist": Box(shape=(128, 128, 3)),
        }
        """
        self.observation_space = gym.spaces.Dict(
            {
                "state": flatten_space(self.proprio_space),
                **image_spaces,
            }
        )

    def observation(self, observation):
        """环境原始观测转换成神经网络需要的格式"""

        state = flatten(
            self.proprio_space,                                                 # 数据结构和拼接规则
            {
                key: observation["state"][key] for key in self.proprio_keys     # 当前时刻实际数据
            },
        ).astype(np.float32)

        return {"state": state, **observation.get("images", {})}    # observation.get("images", {}) 是在 Python 字典中读取 "images"，并提供默认值 {}
