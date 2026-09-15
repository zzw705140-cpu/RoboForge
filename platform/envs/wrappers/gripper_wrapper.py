"""Action wrappers for fixed-gripper single-arm tasks."""

import gymnasium as gym
import numpy as np

########################################################
#                 夹爪相关包装层                         #
########################################################


# 用于夹爪始终闭合的任务：上层策略输出 6 维动作，自动补上第 7 维 +1 后传给底层环境
class GripperCloseEnv(gym.ActionWrapper):
    """
    固定夹爪环境

    建议放在最靠近底层环境的位置
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        action_space = self.env.action_space
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError("GripperCloseEnv requires a Box action space")
        if action_space.shape != (7,):
            raise ValueError(f"GripperCloseEnv requires action shape (7,), got {action_space.shape}")

        self.action_space = gym.spaces.Box(
            low=action_space.low[:6],
            high=action_space.high[:6],
            dtype=action_space.dtype,
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        """Append a persistent closed target to the six-dimensional policy action."""

        action = np.asarray(action, dtype=self.action_space.dtype)
        if action.shape != (6,):
            raise ValueError(f"action must have shape (6,), got {action.shape}")

        robot_action = np.zeros(7, dtype=self.env.action_space.dtype)
        robot_action[:6] = action
        robot_action[6] = 1.0
        return robot_action

    def step(self, action: np.ndarray):
        """为六维策略动作补上 ``+1``，使夹爪始终以闭合为目标。"""

        # 神经网络输出动作 6 维，需要补 +1（张开=-1，闭合=+1）。
        robot_action = self.action(action)

        obs, reward, terminated, truncated, info = self.env.step(robot_action)
        info = dict(info)

        """
        情况一: InterventionWrapper 在内层

        env = RobotSimEnv()               # 7维
        env = InterventionWrapper(env)    # 人工动作7维
        env = GripperCloseEnv(env)        # 对外动作6维
        
        情况二: InterventionWrapper 在外层

        env = RobotSimEnv()               # 7维
        env = GripperCloseEnv(env)        # 对外动作6维
        env = InterventionWrapper(env)    # 人工动作6维
        """

        # 如果 “InterventionWrapper” 在内层，则 info["intervene_action"] 动作是 7 维，需要剪裁
        if "intervene_action" in info:
            info["intervene_action"] = np.asarray(info["intervene_action"][:6], dtype=self.action_space.dtype)

        return obs, reward, terminated, truncated, info


# 夹爪真正发生打开或闭合切换时扣除奖励，避免策略频繁开合夹爪
class GripperPenaltyWrapper(gym.RewardWrapper):
    """
    夹爪发生打开或关闭操作时施加惩罚，抑制策略频繁开合夹爪

    注意, GripperPenaltyWrapper 必须放在 FlattenStateObservationWrapper 的内层, 因为需要读取原始观测中的 ``state["gripper_target"]``
    """

    def __init__(self, env: gym.Env, penalty: float = 0.1):
        super().__init__(env)

        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.gripper_threshold = self.env.unwrapped.config.gripper_threshold
        self.gripper_action_scale = float(self.env.unwrapped.config.action_scale[2])
        self.last_gripper_target = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        # 记录当前持续目标，而不是有物理延迟的实际开度。
        self.last_gripper_target = float(observation["state"]["gripper_target"][0])

        return observation, info

    def reward(self, reward, action):
        """只有夹爪实际需要改变开合状态时才施加惩罚"""

        # 与基础环境使用同一迟滞语义：张开目标下越过正阈值才闭合，闭合目标下越过负阈值才张开。
        gripper_action = float(action[6]) * self.gripper_action_scale
        close_gripper = (
            self.last_gripper_target < 0.0
            and gripper_action > self.gripper_threshold
        )
        open_gripper = (
            self.last_gripper_target > 0.0
            and gripper_action < -self.gripper_threshold
        )

        if close_gripper or open_gripper:
            return reward - self.penalty
        return reward

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # 人工干预时，根据真正执行的动作计算惩罚
        executed_action = info.get("intervene_action", action)

        # 修正奖励
        original_reward = reward
        reward = self.reward(reward, executed_action)
        info["grasp_penalty"] = np.float32(original_reward - reward)

        # 更新环境真正采用的持续夹爪目标。
        self.last_gripper_target = float(observation["state"]["gripper_target"][0])

        return observation, reward, terminated, truncated, info
