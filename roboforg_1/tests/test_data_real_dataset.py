"""真实数据集集成检查，显式开启以避免日常单元测试占用大量 RAM。

项目根目录运行：ROBOFORGE_TEST_REAL_DATA=1 python -m pytest -q roboforg_1/tests
只读取 train_1、eval_1，不修改原始文件，也不启动仿真环境。
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from roboforg_1.data.chunk_bc_replay_buffer import ChunkBCReplayBuffer
from roboforg_1.data.data_store import InMemoryDataStore
from roboforg_1.data.observation import load_dataset
from roboforg_1.data.utils import tree_take


pytestmark = pytest.mark.skipif(os.environ.get("ROBOFORGE_TEST_REAL_DATA") != "1", reason="Set ROBOFORGE_TEST_REAL_DATA=1 to check local datasets.")


@pytest.mark.parametrize("split", ["train_1", "eval_1"])
def test_real_dataset_loading_and_sampling(split) -> None:
    files = sorted((Path(__file__).resolve().parents[2] / "datasets" / "pick" / split).glob("*.pkl"))
    assert files, f"Dataset not found: {split}"
    trajectories = []
    for path in files:
        payload = load_dataset(path)
        with path.open("rb") as stream:
            original = pickle.load(stream)
        assert len(payload["trajectories"]) == len(original["trajectories"])
        for converted, raw in zip(payload["trajectories"], original["trajectories"]):
            # 格式转换只移动相机字段，逐项确认原始数值未改变。
            np.testing.assert_array_equal(converted["observations"]["state"], raw["observations"]["state"])
            for camera in ("front", "wrist"):
                source = raw["observations"].get("images", raw["observations"])[camera]
                np.testing.assert_array_equal(converted["observations"]["images"][camera], source)
            for field in ("actions", "rewards", "masks", "dones"):
                np.testing.assert_array_equal(converted[field], raw[field])
                assert np.isfinite(converted[field]).all()
            assert np.isfinite(converted["observations"]["state"]).all()
        trajectories.extend(payload["trajectories"])

    store = InMemoryDataStore.from_trajectories(trajectories)
    buffer = ChunkBCReplayBuffer(store, seed=17)
    starts = [(trajectory, step) for trajectory in trajectories for step in range(len(trajectory["actions"]))]
    assert len(starts) == len(buffer) == store.capacity

    # 三组常见配置均遍历完整数据；尾部起点、长短轨迹与最后不足一批都参与检查。
    for observation_horizon, action_horizon in ((1, 1), (2, 8), (4, 16)):
        count = 0
        iterator = buffer.get_epoch_iterator(batch_size=16, observation_horizon=observation_horizon, action_horizon=action_horizon, shuffle=False)
        for batch in iterator:
            assert batch["actions"].shape[1:] == (action_horizon, 7)
            assert batch["observations"]["state"].shape[1:] == (observation_horizon, 23)
            assert batch["action_valid"].dtype == np.bool_
            for row in range(len(batch["actions"])):
                trajectory, step = starts[count + row]
                valid_actions = trajectory["actions"][step : step + action_horizon]
                valid_count = len(valid_actions)
                np.testing.assert_array_equal(batch["actions"][row, :valid_count], valid_actions)
                np.testing.assert_array_equal(batch["action_valid"][row], np.arange(action_horizon) < valid_count)
                if valid_count < action_horizon:
                    np.testing.assert_array_equal(batch["actions"][row, valid_count:], np.repeat(valid_actions[-1:], action_horizon - valid_count, axis=0))
                current_observation = tree_take(trajectory["observations"], step)
                np.testing.assert_array_equal(batch["observations"]["state"][row, -1], current_observation["state"])
                for camera in ("front", "wrist"):
                    images = batch["observations"]["images"][camera]
                    assert images.dtype == np.uint8
                    np.testing.assert_array_equal(images[row, -1], current_observation["images"][camera])
            count += len(batch["actions"])
        assert count == len(buffer)

    # 随机入口也能直接使用相同 Store 和配置，且每条样本至少有一个真实动作。
    random_batch = buffer.sample_chunk(32, observation_horizon=2, action_horizon=8)
    assert random_batch["actions"].shape == (32, 8, 7)
    assert random_batch["action_valid"].any(axis=1).all()
    print(f"{split}: {len(files)} files, {store.num_trajectories} trajectories, {len(buffer)} transitions; all three epochs passed.")
