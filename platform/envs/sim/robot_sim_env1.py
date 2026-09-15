from __future__ import annotations

from typing import Any

import mujoco
import mujoco.viewer
import gymnasium as gym
import numpy as np
import time
from scipy.spatial.transform import Rotation

from pathlib import Path

from .opspace import opspace


MAX_GRIPPER_COMMAND = 255.0
OPEN_GRIPPER_TARGET = -1.0
CLOSED_GRIPPER_TARGET = 1.0

class RobotSimEnvConfig:
    """Default simulation config; task configs can override class attributes."""

    # 将策略输出的归一化动作 [-1, 1] 转化为实际控制量：TCP 平移缩放（米）、TCP 旋转缩放（弧度）、夹爪动作缩放
    action_scale = np.asarray([0.02, 0.10, 1.0], dtype=np.float64)

    # TCP 安全范围：前三维为 xyz，后三维为 XYZ 欧拉角。
    pose_limit_low  = np.asarray([0.2, -0.3, 0.0, -np.pi, -np.pi, -np.pi], dtype=np.float64)
    pose_limit_high = np.asarray([0.6, 0.3, 0.5, np.pi, np.pi, np.pi],     dtype=np.float64)

    home_position = np.asarray([0.0, 0.195, 0.0, -2.43, 0.0, 2.62, np.pi / 4], dtype=np.float64)

    hz = 10.0                                       #每0.1秒更新一次
    physics_dt = 0.002                              #物理引擎每仿真0.002秒更新一步
    max_episode_length = 100                        #最大时间步
    image_size = (128, 128)
    camera_names = ("front", "handcam_rgb")        #两个相机的名称
    camera_keys = ("front", "wrist")               #名称对应的键值
    use_images = False                             #是否使用图像输入
    gripper_threshold = 0.5                        #定义夹爪阈值，超过0.5关闭

class RobotSimEnv(gym.Env):
    def __init__(
        self,
        *,
        config: RobotSimEnvConfig | None = None,
        xml_path: str | Path | None = None,
        show_viewer: bool = True,
        fake_env: bool = False,                            #假环境
    ) -> None:
        super().__init__()

        self.config = config or RobotSimEnvConfig()
        self.show_viewer      = bool(show_viewer)
        self.fake_env         = bool(fake_env)             #是否创建假环境
        self.current_step     = 0
        self.is_closed        = False                      #先不考虑实现判断环境是否关闭的功能
        self._gripper_target  = OPEN_GRIPPER_TARGET        #reset 时，默认夹爪张开

        # 设置动作空间，定义接口规格，告诉上层接收的是什么结构 类型 范围的数据
        self.action_space = gym.spaces.Box(
            low=-np.ones(7, dtype=np.float32),
            high=np.ones(7, dtype=np.float32),
            dtype=np.float32,
        )
        # 设置观测空间，定义接口规格
        self.observation_space = self._make_observation_space()

        # 初始化仿真可视化窗口，方便后续进行观测以及测试
        self._renderer: mujoco.Renderer | None = None
        self._viewer = None

        # 以后需要learner时在写出来，并修改上面的model data
        if self.fake_env:
            return

        # 根据当前文件位置生成默认 MuJoCo 场景 XML 路径。
        if xml_path is None:
            xml_path = Path(__file__).resolve().parent / "simulation" / "assets" / "scene.xml"
        self.xml_path = Path(xml_path).resolve()
        self._initialize_mujoco()


##########################################################################

    def _initialize_mujoco(self) -> None:
        """前置工作，加载文件,读取id"""
        # 先调用初始化函数，然后在内部创建model data。两个作用分别为加载保存XML内部编译好的配置文件 创建并保存仿真时产生的动态状态
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        # 查找XML文件内各个配件的id并组成数组
        # 7个关节的id  7各关节下角标的id
        self._joint_ids     = np.asarray( [self.model.joint(f"joint{i}").id for i in range(1, 8)] )
        self._joint_dof_ids = self.model.jnt_dofadr[self._joint_ids]
        # 7个执行器id
        self._actuator_ids = np.asarray( [self.model.actuator(f"actuator{i}").id for i in range(1, 8)] )

        # 夹爪id 将命令写进这个id，读取左右夹爪id，获得qpos下标
        self._gripper_actuator_id = self.model.actuator("fingers_actuator").id
        self._gripper_joint_ids = np.asarray([self.model.joint(name).id for name in ("right_driver_joint", "left_driver_joint")])
        self._gripper_qpos_ids = self.model.jnt_qposadr[self._gripper_joint_ids]
        # 机械臂末端TCPid
        # mocap负责保存目标位置姿态，TCP是机械臂末端当前位置姿态，通过opspace()函数计算目标位置姿态并保存到mocap，控制器负责调控位置
        self._tcp_site_id = self.model.site("pinch").id

        #
        self._n_substeps = round((1.0 / self.config.hz) / self.config.physics_dt)

    def _make_observation_space(self) -> gym.spaces.Dict:
        """创建观测空间"""

        # 创建字典形式的状态观测空间，每个键对应一项机器人状态
        state_space = gym.spaces.Dict(
            # 保存每项状态的名称及其空间定义
            {
                # TCP 位姿：3 维位置加 4 维四元数，共 7 维，数值范围暂不限制
                "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                # TCP 速度：3 维线速度加 3 维角速度，共 6 维，数值范围暂不限制
                "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                # 夹爪当前实际开合程度：0 表示张开，1 表示闭合
                "gripper_pose": gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                # 夹爪控制目标：-1 表示张开，1 表示闭合
                "gripper_target": gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
                # TCP 在三个方向受到的力，共 3 维，数值范围暂不限制
                "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                # TCP 绕三个方向受到的力矩，共 3 维，数值范围暂不限制
                "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            }
        )

        spaces: dict[str, gym.Space] = {"state": state_space}

        if self.config.use_images:                                 # 根据配置决定是否加入图像观测
            height, width = self.config.image_size                 # 读取图像的高度和宽度
            spaces["images"] = gym.spaces.Dict(                    # 创建包含多个相机图像的字典空间
                {
                    key: gym.spaces.Box(                           # 为每个相机定义一个图像空间
                        0,                                         # 图像像素最小值
                        255,                                       # 图像像素最大值
                        shape=(height, width, 3),                  # 图像形状为高度、宽度和 RGB 三通道
                        dtype=np.uint8                             # 每个像素使用 8 位无符号整数
                    )
                    for key in self.config.camera_keys             # 根据相机键名创建对应的图像空间
                }
            )

        return gym.spaces.Dict(spaces)                             # 合并状态和图像并返回完整观测空间

##########################################################################


    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """统一 reset 主流程。"""

        # 设置随机种子 初始化环境内部和GYM接口相关的一些基础状态
        super().reset(seed=seed)
        self.current_step = 0

        # 调用具体的重置法去执行仿真环境的重置
        self._reset_env(options=options)

        # 重置完成一次之后读取一次当前初始观测
        observation = self._get_obs()
        # 准备一个额外信息字典
        info: dict[str, Any] = {}
        return observation, info


    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        # 这五个为标准的obs, reward, terminated, truncated, info 
        """统一 step 主流程。"""

        if self.is_closed:   ####尚未创建
            raise RuntimeError("Cannot call step() after close().")

        # 将传入的动作转换为np数组并统一成float32
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"action must have shape {self.action_space.shape}, got {action.shape}."
            )

        # 获取当前状态
        state = self._get_robot_state()

        # 将完整策略动作转换为 TCP 和夹爪目标，并写入 MuJoCo
        self._apply_action(action, state)
        self.current_step += 1

        #推进一个仿真周期
        self._advance_simulation()

        # 同步可视化窗口
        if self.show_viewer:
            self._render()

        observation = self._get_obs()                                 #执行动作后读取新的观测

        # 这三个都是子类负责实现的具体标准等功能
        reward      = float(self._compute_reward(observation))                #根据新的观测计算奖励          
        success     = bool(self._is_success(observation))                     #判断这一步之后的任务是否成功
        terminated  = bool(success or self._should_terminate(observation))    #判断是否终止

        truncated   = bool(
        self.current_step >= self.config.max_episode_length and not terminated       #判断是否因步数达到上限而被截断
        )

        info = {"is_success": success}                                        #这附加信息字典，现在只有一个成功标志
        return observation, reward, terminated, truncated, info


    def close(self) -> None:
        """统一资源释放入口。"""

        if self.is_closed:
            return

        # 调用具体关闭函数去释放真正的底层资源
        self._close_env()
        self.is_closed = True      #表示这个环境已关闭


#######################################################################
#                      主要流程的主要内置函数                            #
#######################################################################

    def _reset_env(self, *, options: dict[str, Any] | None = None) -> None:
        """仿真重置钩子，子类父类可按需逐层扩展。"""

        del options

        # 清除上一回合的仿真状态。
        mujoco.mj_resetData(self.model, self.data)

        # 根据关节 ID 找到对应的 qpos 地址，并设置初始关节角。
        joint_qpos_indices = self.model.jnt_qposadr[self._joint_ids]
        self.data.qpos[joint_qpos_indices] = self.config.home_position

        # 将机械臂和夹爪执行器的控制量清零。
        self.data.ctrl[self._actuator_ids] = 0.0
        self.data.ctrl[self._gripper_actuator_id] = 0.0

        # 根据初始关节角更新机器人各部分的位置和姿态。
        mujoco.mj_forward(self.model, self.data)

        # 将 mocap 控制目标对齐当前 TCP，避免首次控制时发生跳动。
        tcp_quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(
            tcp_quaternion,
            self.data.site_xmat[self._tcp_site_id],
        )
        self.data.mocap_pos[0] = self.data.site_xpos[self._tcp_site_id]
        self.data.mocap_quat[0] = tcp_quaternion

        # 重置夹爪目标，并更新对齐 mocap 后的仿真状态。
        self._gripper_target = OPEN_GRIPPER_TARGET
        mujoco.mj_forward(self.model, self.data)

    def _advance_simulation(self) -> None:
        """使用操作空间控制器推进一个控制周期。"""

        for _ in range(self._n_substeps):                         #一个控制周期内循环推进多个物理子步
            joint_torques = opspace(
                model=self.model,                                 #MuJoCo模型配置
                data=self.data,                                   #当前仿真动态状态
                site_id=self._tcp_site_id,                        #需要控制的TCP位置
                dof_ids=self._joint_dof_ids,                      #机械臂七个关节的自由度下标
                pos=self.data.mocap_pos[0],                       #TCP目标位置
                ori=self.data.mocap_quat[0],                      #TCP目标姿态
                joint=self.config.home_position,                  #机械臂零空间参考关节角
                gravity_comp=True,                                #启用重力补偿
            )
            # #将关节力矩写入七个机械臂执行器
            self.data.ctrl[self._actuator_ids] = joint_torques 
            # 推进一个MuJoCo物理子步
            mujoco.mj_step(self.model, self.data)                 
            
    def _apply_action(self, action: np.ndarray, state: dict[str, np.ndarray]) -> None:
        """将上层动作转换为 TCP 和夹爪控制目标，并写入 MuJoCo。"""

        # 读取当前 TCP 位姿              注！！没有限制一次运动的范围
        current_pose = np.asarray(state["tcp_pose"], dtype=np.float64)
        if current_pose.shape != (7,):
            raise ValueError(f"tcp_pose must have shape (7,), got {current_pose.shape}.")

        # 根据动作计算 TCP 目标位置
        target_pose = current_pose.copy()
        target_pose[:3] += action[:3] * self.config.action_scale[0]

        # 根据旋转增量计算 TCP 目标姿态
        delta_rotation = Rotation.from_rotvec(action[3:6] * self.config.action_scale[1])
        target_pose[3:] = (delta_rotation * Rotation.from_quat(current_pose[3:])).as_quat()

        # 限制 TCP 目标位姿
        target_pose = self._clip_safety_box(target_pose)

        # 检查并归一化目标四元数
        target_quat_xyzw = target_pose[3:]
        target_quat_norm = np.linalg.norm(target_quat_xyzw)
        if target_quat_norm < 1e-8:
            raise ValueError("target pose contains a zero quaternion")

        # 将目标位置和姿态写入 mocap
        self.data.mocap_pos[0] = target_pose[:3]
        self.data.mocap_quat[0] = self._xyzw_to_wxyz(target_quat_xyzw / target_quat_norm)

        # 根据动作第七维切换夹爪目标
        gripper_action = float(action[6]) * self.config.action_scale[2]
        if self._gripper_target == OPEN_GRIPPER_TARGET and gripper_action > self.config.gripper_threshold:
            self._gripper_target = CLOSED_GRIPPER_TARGET
        elif self._gripper_target == CLOSED_GRIPPER_TARGET and gripper_action < -self.config.gripper_threshold:
            self._gripper_target = OPEN_GRIPPER_TARGET

        # 将 [-1, 1] 夹爪目标映射为 [0, 255] 控制量
        normalized_target = (self._gripper_target + 1.0) / 2.0
        self.data.ctrl[self._gripper_actuator_id] = normalized_target * MAX_GRIPPER_COMMAND

    def _clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """限制 TCP 目标位置不超出基础工作空间。"""   #未实现姿态的限制

        #将输入转换为 float64位的numpy 数组，并赋值一份防止被修改原数据
        pose = np.asarray(pose, dtype=np.float64).copy()

        if pose.shape != (7,):
            raise ValueError(f"pose must have shape (7,), got {pose.shape}.")

        #限制TCP位置
        pose[:3] = np.clip(
            pose[:3],
            self.config.pose_limit_low[:3],
            self.config.pose_limit_high[:3],
        )

        return pose

    def _get_images(self) -> dict[str, np.ndarray]:
        """获取相机图像，后续按需实现。"""

        return {}

    def _get_robot_state(self) -> dict[str, np.ndarray]:
        """读取 TCP 位姿和速度。"""

        # 读取 TCP 位置。
        tcp_position = self.data.site_xpos[self._tcp_site_id].copy()

        # 将 TCP 旋转矩阵转换为 MuJoCo 使用的 wxyz 四元数。
        tcp_quat_wxyz = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat( tcp_quat_wxyz, self.data.site_xmat[self._tcp_site_id],)


        # 调整为环境统一使用的 xyzw 四元数，并与位置拼成七维位姿。
        tcp_quat_xyzw = self._wxyz_to_xyzw(tcp_quat_wxyz)
        tcp_pose = np.concatenate((tcp_position, tcp_quat_xyzw))

        # MuJoCo 返回 [角速度, 线速度]，调整为 [线速度, 角速度]。
        tcp_velocity_raw = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity( self.model, self.data,
            mujoco.mjtObj.mjOBJ_SITE, self._tcp_site_id, tcp_velocity_raw, 0,
        )

        tcp_velocity = np.concatenate( (tcp_velocity_raw[3:], tcp_velocity_raw[:3]))

        # 读取并归一化夹爪实际开合状态
        driver_qpos  = self.data.qpos[self._gripper_qpos_ids].copy()
        driver_range = self.model.jnt_range[self._gripper_joint_ids]
        driver_low   = driver_range[:, 0]
        driver_high  = driver_range[:, 1]

        normalized_driver_qpos = np.clip((driver_qpos - driver_low) / (driver_high - driver_low), 0.0, 1.0)
        gripper = np.asarray([np.mean(normalized_driver_qpos)], dtype=np.float64)

        # 读取夹爪目标状态
        gripper_target = np.asarray([self._gripper_target], dtype=np.float64)

        # 当前尚未读取完整的 TCP 力和力矩，先用零数组保持仿真与真机观测接口一致。
        tcp_force = np.zeros(3, dtype=np.float64)
        tcp_torque = np.zeros(3, dtype=np.float64)

        # MuJoCo 内部保持 float64，返回上层算法时统一转换为 float32。
        return {
            "tcp_pose": tcp_pose.astype(np.float32),
            "tcp_vel": tcp_velocity.astype(np.float32),
            "gripper_pose": gripper.astype(np.float32),
            "gripper_target": gripper_target.astype(np.float32),
            "tcp_force": tcp_force.astype(np.float32),
            "tcp_torque": tcp_torque.astype(np.float32),
        }

    def _get_obs(self) -> Any:
        """从仿真底层读取观测值"""

        obs: dict[str, Any] = {"state": self._get_robot_state()}

        if self.config.use_images:
            obs["images"] = self._get_images()

        return obs

    @staticmethod
    def _xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
        return np.asarray(quat)[[3, 0, 1, 2]]

    @staticmethod
    def _wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
        return np.asarray(quat)[[1, 2, 3, 0]]

######################################################################
#                          具体任务具体实现                            #
######################################################################


    def _compute_reward(self, observation: Any) -> float:
        """奖励计算钩子，子类父类可按需逐层扩展。"""

        del observation
        return 0.0

    def _is_success(self, observation: Any) -> bool:
        """成功判定钩子，子类父类可按需逐层扩展。"""

        del observation
        return False

    def _should_terminate(self, observation: Any) -> bool:
        """提前终止钩子，子类父类可按需逐层扩展。"""

        del observation
        return False

######################################################################
#                         渲染与可视化窗口                             #
######################################################################

    def _render(self) -> None:
        """显示并刷新供操作者观察的仿真窗口"""

        # 第一次调用时惰性打开被动窗口；该窗口只显示，不推进物理状态。
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # 用户手动关闭窗口后及时释放句柄；后续 render 可重新打开。
        if not self._viewer.is_running():
            self._viewer.close()
            self._viewer = None
            return

        # 每个控制周期同步一次，而不是每个物理子步同步，避免拖慢训练。
        self._viewer.sync()

    def _close_env(self) -> None:
        """资源释放钩子，子类父类可按需逐层扩展。"""

        if self._renderer is not None:
            close = getattr(self._renderer, "close", None)      #如果又close这个属性，就取出来
            if callable(close):                                 #判断这个close是不是可调用的
                close()
            self._renderer = None                               #表示这个渲染器引用已经清理掉了

        # 释放供操作者观察和 HIL 介入使用的 MuJoCo viewer
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
            time.sleep(0.2)                                     #设置一段时间关闭Mujoco viewer
