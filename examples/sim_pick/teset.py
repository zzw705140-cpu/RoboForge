from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLATFORM_ROOT = PROJECT_ROOT / "platform"
for module_root in (PROJECT_ROOT, PLATFORM_ROOT):
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))

from examples.sim_pick.config import EnvConfig, PickEnv, TrainConfig

##########################################################
#                       功能函数测试                       #
##########################################################


@pytest.fixture
def pick_env() -> PickEnv:
    """创建只包含任务函数测试所需状态的 PickEnv。"""

    scene_path = PLATFORM_ROOT / "envs" / "sim" / "simulation" / "assets" / "scene.xml"
    env = PickEnv.__new__(PickEnv)
    env.config = EnvConfig()
    env.model = mujoco.MjModel.from_xml_path(str(scene_path))
    env.data = mujoco.MjData(env.model)
    env._tcp_site_id = env.model.site("pinch").id
    env._z_init = float(env.data.sensor("block_pos").data[2])
    env.reward_type = "sparse"
    mujoco.mj_forward(env.model, env.data)
    return env


def _set_block_position(env: PickEnv, position: np.ndarray) -> None:
    """设置方块 XYZ，并更新 MuJoCo 派生数据。"""

    block_joint_id = env.model.joint("block").id
    block_qpos_address = env.model.jnt_qposadr[block_joint_id]
    env.data.qpos[block_qpos_address : block_qpos_address + 3] = np.asarray(position, dtype=np.float64)
    mujoco.mj_forward(env.model, env.data)


def test_compute_reward(pick_env: PickEnv) -> None:
    """检查稀疏奖励和稠密奖励。"""

    initial_pos = pick_env.data.sensor("block_pos").data.copy()
    pick_env._z_init = float(initial_pos[2])

    pick_env.reward_type = "sparse"
    assert pick_env._compute_reward({}) == 0.0
    _set_block_position(pick_env, initial_pos + np.asarray([0.0, 0.0, 0.11]))
    assert pick_env._compute_reward({}) == 1.0

    tcp_pos = pick_env.data.site_xpos[pick_env._tcp_site_id].copy()
    _set_block_position(pick_env, tcp_pos)
    pick_env._z_init = float(tcp_pos[2])
    pick_env.reward_type = "dense"
    assert np.isclose(pick_env._compute_reward({}), 0.3)


def test_is_success(pick_env: PickEnv) -> None:
    """检查方块抬升10 cm的成功条件。"""

    initial_pos = pick_env.data.sensor("block_pos").data.copy()
    pick_env._z_init = float(initial_pos[2])

    _set_block_position(pick_env, initial_pos + np.asarray([0.0, 0.0, 0.099]))
    assert pick_env._is_success({}) is False

    _set_block_position(pick_env, initial_pos + np.asarray([0.0, 0.0, 0.10]))
    assert pick_env._is_success({}) is True


def test_should_terminate(pick_env: PickEnv) -> None:
    """检查方块是否超出带5 cm余量的XY边界。"""

    z = float(pick_env.data.sensor("block_pos").data[2])

    _set_block_position(pick_env, np.asarray([0.40, 0.00, z]))
    assert pick_env._should_terminate({}) is False

    _set_block_position(pick_env, np.asarray([0.56, 0.00, z]))
    assert pick_env._should_terminate({}) is True

    _set_block_position(pick_env, np.asarray([0.40, -0.21, z]))
    assert pick_env._should_terminate({}) is True


#############################################################
#                        调用链路测试                         #
#############################################################

class TestTrainConfigEnvironmentChain:
    """测试 TrainConfig 到最终 Gym 环境的构造链路。"""

    def _config(self) -> TrainConfig:
        config = TrainConfig()
        config.show_viewer = False
        config.input_device = None
        return config

    def test_task_mapping_can_find_pick_train_config(self) -> None:
        """局部任务索引能通过 pick 找到 TrainConfig，并能创建配置对象。"""

        config_mapping = {"pick": TrainConfig}
        config_class = config_mapping["pick"]

        assert config_class is TrainConfig
        assert isinstance(config_class(), TrainConfig)

    def test_fake_env_can_build_spaces(self) -> None:
        """fake_env 只构造接口空间，不加载 MuJoCo。"""

        env = self._config().get_environment(fake_env=True)

        try:
            assert env.action_space is not None
            assert env.observation_space is not None
            assert "state" in env.observation_space.spaces
        finally:
            env.close()

    def test_real_env_can_reset(self) -> None:
        """真实 MuJoCo 环境能创建并返回初始观测。"""

        env = self._config().get_environment(fake_env=False)

        try:
            observation, info = env.reset()

            assert "state" in observation
            assert env.observation_space.contains(observation)
            assert info["is_success"] is False
        finally:
            env.close()

    def test_real_env_can_step_through_wrappers(self) -> None:
        """真实环境能 step 一步，并检查 wrapper 后的标准五元组。"""

        env = self._config().get_environment(fake_env=False)

        try:
            env.reset()
            action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
            observation, reward, terminated, truncated, info = env.step(action)

            assert "state" in observation
            assert observation["state"].ndim == 1
            assert env.observation_space.contains(observation)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert "is_success" in info
        finally:
            env.close()
