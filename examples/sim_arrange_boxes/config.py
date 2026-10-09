"""Arrange-boxes task environment and training configuration."""

from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import mujoco
import numpy as np

from examples.default_config import DefaultConfig as TaskConfig
from envs.sim.robot_sim_env1 import (
    RobotSimEnv,
    RobotSimEnvConfig,
)
from envs.wrappers.gripper_wrapper      import GripperPenaltyWrapper
from envs.wrappers.intervention_wrapper import InterventionWrapper
from envs.wrappers.observation_wrapper  import (
    FlattenStateObservationWrapper,
    Quat2R2Wrapper,
)
from envs.wrappers.relative_frame_wrapper import RelativeFrameWrapper


ARRANGE_BOXES_XML = (Path(__file__).resolve().parents[2]/ "platform"/ "assets"/ "arrange_boxes_scene.xml")


class EnvConfig(RobotSimEnvConfig):
    """ArrangeBoxes 任务的底层环境配置"""

    max_episode_length = 1000

    home_position = np.asarray([0.0, -0.785, 0.0, -2.35, 0.0, 1.57, np.pi / 4], dtype=np.float64)
    pose_limit_low = np.asarray([0.2, -0.5, 0.0, -np.pi, -np.pi, -np.pi],       dtype=np.float64)
    pose_limit_high = np.asarray([0.6, 0.5, 0.5, np.pi, np.pi, np.pi],          dtype=np.float64)

    num_blocks   = 5
    block_y_low  = -0.30
    block_y_high = 0.30

    # 任意方块离开该区域即提前结束 episode。
    block_xy_low = np.asarray([-0.05, -0.50], dtype=np.float64)
    block_xy_high = np.asarray([0.85, 0.50], dtype=np.float64)

    success_distance = 0.03
    distance_reward_scale = 20.0


class ArrangeBoxesEnv(RobotSimEnv):
    """将多个彩色方块移动到对应目标区域的任务环境。"""

    def __init__(
        self,
        *,
        config: EnvConfig | None = None,
        reward_type: Literal["sparse", "dense"] = "sparse",
        random_block_order: bool = True,
        xml_path: str | Path = ARRANGE_BOXES_XML,
        **kwargs,
    ):
        self.reward_type = reward_type
        self.random_block_order = bool(random_block_order)
        super().__init__(config=config or EnvConfig(), xml_path=xml_path, **kwargs)

        self.block_names  = [f"block{i}" for i in range(1, self.config.num_blocks + 1)]
        self.target_names = [f"target{i}" for i in range(1, self.config.num_blocks + 1)]

        # fake_env 不加载 XML，几何尺寸使用当前任务 XML 的已知默认值。
        self._block_z = (0.02 if self.fake_env else float(self.model.geom("block1").size[2]))

    def _make_observation_space(self) -> gym.spaces.Dict:
        observation_space = super()._make_observation_space()

        if not self.config.use_images:
            observation_space["state"]["block_positions"] = gym.spaces.Box(
                -np.inf,
                np.inf,
                shape=(self.config.num_blocks, 3),
                dtype=np.float32,
            )

        return observation_space

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        _, info = super().reset(seed=seed, options=options)

        y_positions = np.linspace(self.config.block_y_low, self.config.block_y_high, self.config.num_blocks)
        block_names = list(self.block_names)

        if self.random_block_order:
            block_names = list(self.np_random.permutation(block_names))

        # 每个方块保留 XML 中的初始 X 坐标，仅随机打乱其 Y 轨道
        for block_name, y_position in zip(block_names, y_positions):
            block_qpos = self.data.jnt(block_name).qpos
            block_qpos[:3] = (
                block_qpos[0], 
                y_position, 
                self._block_z
            )

        mujoco.mj_forward(self.model, self.data)
        observation = self._get_obs()

        if self.show_viewer:
            self._render()

        info = dict(info)
        info["succeed"] = False
        return observation, info

    def _get_robot_state(self) -> dict[str, np.ndarray]:
        state = super()._get_robot_state()

        if not self.config.use_images:
            state["block_positions"] = np.stack(
                [
                    self.data.sensor(f"block{i}_pos").data.copy() for i in range(1, self.config.num_blocks + 1)
                ]
            ).astype(np.float32)

        return state

    def get_block_target_distances(self) -> np.ndarray:
        """返回每个方块到对应目标中心的距离。"""

        return np.asarray(
            [
                np.linalg.norm(
                    self.data.sensor(f"block{i}_pos").data
                    - self.data.sensor(f"target{i}_pos").data
                )
                for i in range(1, self.config.num_blocks + 1)
            ],
            dtype=np.float64,
        )

    def _compute_reward(self, obs: dict[str, Any]) -> float:
        distances = self.get_block_target_distances()

        if self.reward_type == "dense":
            return float(np.exp(-self.config.distance_reward_scale * distances).sum())

        return float(np.all(distances < self.config.success_distance))

    def _is_success(self, obs: dict[str, Any]) -> bool:
        distances = self.get_block_target_distances()
        return bool(np.all(distances < self.config.success_distance))

    def _should_terminate(self, obs: dict[str, Any]) -> bool:
        block_positions = np.stack(
            [
                self.data.sensor(f"block{i}_pos").data[:2] for i in range(1, self.config.num_blocks + 1)
            ]
        )
        outside = np.logical_or(
            block_positions < self.config.block_xy_low,
            block_positions > self.config.block_xy_high,
        )
        return bool(np.any(outside))


class ArrangeBoxesTaskConfig(TaskConfig):
    """ArrangeBoxes 任务的环境配置及构造入口。"""

    image_keys   = ["front", "wrist"]
    proprio_keys = [
        "tcp_pose",
        "tcp_vel",
        "gripper_pose",
        "gripper_target",
        "tcp_force",
        "tcp_torque",
    ]
    setup_mode  = "single-arm-learned-gripper"

    reward_type = "sparse"
    random_block_order = True
    observation_horizon = 1
    gripper_penalty = 0.1

    input_device = "spacemouse"
    show_viewer  = True

    def get_environment(
        self,
        fake_env: bool = False,
        save_video: bool = False,
        classifier: bool = False,
    ):
        """构造 learner 或 actor 使用的 ArrangeBoxes 环境。"""

        del save_video, classifier

        env = ArrangeBoxesEnv(
            config=EnvConfig(),
            fake_env=fake_env,
            show_viewer=self.show_viewer and not fake_env,
            reward_type=self.reward_type,
            random_block_order=self.random_block_order,
        )
        if self.gripper_penalty is not None:
            env = GripperPenaltyWrapper(env, penalty=self.gripper_penalty)
        if not fake_env and self.input_device is not None:
            env = InterventionWrapper(env, input_device=self.input_device)

        env = RelativeFrameWrapper(env)
        env = Quat2R2Wrapper(env)
        env = FlattenStateObservationWrapper(env, proprio_keys=self.proprio_keys)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        return env
