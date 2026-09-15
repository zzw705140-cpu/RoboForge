"""Pick task environment and training configuration."""

from typing import Any, Literal

import gymnasium as gym
import mujoco
import numpy as np

from examples.default_config import DefaultConfig                         #导入的对应的父类，任务共有配置
from envs.sim.robot_sim_env1 import RobotSimEnv, RobotSimEnvConfig        #导入的对应的父类，仿真环境和仿真环境的配置
from envs.wrappers.gripper_wrapper import GripperPenaltyWrapper          #添加夹爪惩罚
from envs.wrappers.intervention_wrapper import InterventionWrapper       #键盘手柄控制
from envs.wrappers.observation_wrapper import (
    FlattenStateObservationWrapper,                                    #按照顺序拼成一维向量便于输入神经网络
    Quat2R2Wrapper,                                                    #
)
# 将绝对坐标系中的观测和动作转换到相对坐标系或末端执行器坐标系下，使策略更关注相对运动关系
from envs.wrappers.relative_frame_wrapper import RelativeFrameWrapper    

###################################################################
#                          抓取方块任务                             #
###################################################################


# 任务的底层环境配置
class EnvConfig(RobotSimEnvConfig):

    max_episode_length = 100

    # 方块随机出现范围
    block_xy_low  = np.asarray([0.3, -0.15], dtype=np.float64)
    block_xy_high = np.asarray([0.5, 0.15],  dtype=np.float64)

    # 方块超出采样区域一定距离后提前结束 episode
    block_out_of_bounds_margin = 0.05

    # 成功条件：方块相对初始高度至少抬升 10 cm
    lift_success_height = 0.10

    # 稠密奖励中 TCP 与方块距离的指数衰减系数
    distance_reward_scale = 20.0


# 抓取方块任务的具体环境
class PickEnv(RobotSimEnv):

    def __init__(
        self,
        *,
        config: EnvConfig | None = None,
        reward_type: Literal["sparse", "dense"] = "sparse",    #默认使用稀疏奖励
        random_block_position: bool = True,                    #默认开启重置方块位置
        **kwargs,                                              #接收其他基础环境参数
    ):
        self.reward_type = reward_type
        self.random_block_position = bool(random_block_position)
        super().__init__(config=config or EnvConfig(), **kwargs)

        # fake_env 只用于 learner 创建网络和 replay buffer，不加载 MuJoCo，所以手动输入 _block_z 高度
        self._block_z = (0.02 if self.fake_env else float(self.model.geom("block").size[2]))

    # 定义一个函数并返回一个Gymnasium字典空间
    def _make_observation_space(self) -> gym.spaces.Dict:
        observation_space = super()._make_observation_space()

        # 不启用图像模式时人为添加方块的高度值
        if not self.config.use_images:
            observation_space["state"].spaces["block_pos"] = gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
            
        return observation_space

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        # 调用父类重置机器人，夹爪和部署
        observation, info = super().reset(seed=seed, options=options)

        # 随机采样 或 固定方块xy
        if self.random_block_position:
            block_xy = self.np_random.uniform(self.config.block_xy_low, self.config.block_xy_high)
        else:
            block_xy = np.asarray([0.5, 0.0], dtype=np.float64)

        block_joint_id = self.model.joint("block").id
        block_qpos_address = self.model.jnt_qposadr[block_joint_id]
        self.data.qpos[block_qpos_address : block_qpos_address + 3] = np.asarray(
            [block_xy[0], block_xy[1], self._block_z], dtype=np.float64
        )
        mujoco.mj_forward(self.model, self.data)

        # 重置后的方块高度
        self._z_init = float(self.data.sensor("block_pos").data[2])
        # 之前重置的一次方块仍在旧的位置上，重置方块位置后需再次读取返回方块新位置对应的观测
        observation = self._get_obs()

        if not self.config.use_images:
            observation["state"]["block_pos"] = self.data.sensor("block_pos").data.copy().astype(np.float32)

        if self.show_viewer:
            self._render()

        info = dict(info)
        info["is_success"] = False
        return observation, info

    def _get_robot_state(self) -> dict[str, np.ndarray]:
        state = super()._get_robot_state()

        if not self.config.use_images: 
            state["block_pos"] = (self.data.sensor("block_pos").data.copy().astype(np.float32))

        return state

########################################################################################

    def _compute_reward(self, observation: Any) -> float:
        """奖励计算函数"""

        # 稀疏奖励
        block_pos = self.data.sensor("block_pos").data
        lift = float(block_pos[2] - self._z_init)

        if self.reward_type == "sparse":
            return float(lift > self.config.lift_success_height)

        # 稠密奖励
        tcp_pos = self.data.site_xpos[self._tcp_site_id]
        distance = np.linalg.norm(block_pos - tcp_pos)
        reach_reward = np.exp(-self.config.distance_reward_scale * distance)
        lift_reward = np.clip(
            lift / self.config.lift_success_height,
            0.0,
            1.0,
        )
        return float(0.3 * reach_reward + 0.7 * lift_reward)

    def _is_success(self, observation: Any) -> bool:
        """成功判定的定义"""
        current_z = float(self.data.sensor("block_pos").data[2])
        return bool(current_z - self._z_init >= self.config.lift_success_height)

    def _should_terminate(self, observation: Any) -> bool:
        """提前终止函数"""

        # 判断方块是否超出边界
        block_xy = self.data.sensor("block_pos").data[:2]
        lower_bound = self.config.block_xy_low - self.config.block_out_of_bounds_margin
        upper_bound = self.config.block_xy_high + self.config.block_out_of_bounds_margin

        return bool(np.any(block_xy < lower_bound) or np.any(block_xy > upper_bound))


class TrainConfig(DefaultConfig):
    """Pick 任务的训练配置及环境构造入口。"""

################################
#         保存默认配置           #
################################

    checkpoint_period = 20000  # Learner 每 20000 次更新保存模型
    buffer_period = 10000      # Actor 每 1000 个环境步保存在线数据

    image_keys   = ["front", "wrist"]   # 表示使用前置相机和腕部相机图像
    proprio_keys = [
        "tcp_pose",                     # 末端位置和姿态
        "tcp_vel",                      # 末端平移与旋转速度
        "gripper_pose",                 # 夹爪当前实际开合位置
        "gripper_target",               # 环境当前保存的夹爪开合目标
        "tcp_force",                    # 末端受到的三维力
        "tcp_torque",                   # 末端受到的三维力矩
    ]
    setup_mode = "single-arm-learned-gripper"     #模式名称

    reward_type = "sparse"              # 表示只在成功或失败时得到奖励/dense时使用连续奖励
    random_block_position = True        # 表示启用方块位置随机生成   /False时方块固定
    observation_horizon = 1             # 规定算法能够接收1帧       /4帧则是将最近4帧作为输入
    gripper_penalty = None              # 当前实验仅使用稀疏任务奖励，暂时不对夹爪开合施加额外惩罚

    # actor 默认使用 SpaceMouse；本地键盘测试可以在实例上改为 keyboard
    input_device = "spacemouse"         #默认使用spacemouse
    show_viewer = True                  #运行环境时默认打开Mijoco可视化窗口

################################
#         创建具体环境           #
################################

    def get_environment(
        self, 
        fake_env: bool = False,         #默认创建仿真环境
        save_video: bool = False,       #不保存视频
        classifier: bool = False        #不启用分类器/段别器辅助判断
    ):
        """构造 Pick 环境"""

        del save_video, classifier

        # 开始创建抓方块任务环境
        env = PickEnv(
            config=EnvConfig(),
            fake_env=fake_env,
            show_viewer=self.show_viewer and not fake_env,     #启用仿真窗口且不启用假的环境时启用
            reward_type=self.reward_type,                      #设置稀疏奖励
            random_block_position=self.random_block_position,     
        )

        # 根据配置决定是否添加夹爪惩罚，人工干预能力
        if self.gripper_penalty is not None:                   #如果启用了夹爪惩罚，系数设为0.1
            env = GripperPenaltyWrapper(env, penalty=self.gripper_penalty)
        if not fake_env and self.input_device is not None:     #如果不是假环境并且设置了输入设备启用人工干预
            env = InterventionWrapper(env, input_device=self.input_device)

        # 环境调用链路，决定训练流程
#################################
#            
        env = RelativeFrameWrapper(env)
        env = Quat2R2Wrapper(env)
        env = FlattenStateObservationWrapper(env, proprio_keys=self.proprio_keys)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        return env
