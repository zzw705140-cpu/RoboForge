"""验证 chunk 的时序对齐、轨迹边界和两种采样方式。"""

import numpy as np
import pytest

from roboforg_1.data.chunk_bc_replay_buffer import ChunkBCReplayBuffer
from roboforg_1.data.data_store import InMemoryDataStore


def make_trajectory(start: int, length: int) -> dict:
    """用不同编号区分轨迹，并让观测与动作编号对应，便于发现错位。"""
    states = np.arange(start, start + length + 1, dtype=np.float32)[:, None]
    return {
        "observations": {"state": states, "images": {"front": states.astype(np.uint8)[:, None, None]}},
        "actions": np.concatenate((states[:-1], states[:-1] + 0.5), axis=-1),
        "rewards": np.zeros(length, dtype=np.float32),
        "masks": np.ones(length, dtype=np.float32),
        "dones": np.arange(length) == length - 1,
    }


def make_buffer(seed: int = 0) -> ChunkBCReplayBuffer:
    store = InMemoryDataStore.from_trajectories([make_trajectory(0, 3), make_trajectory(10, 1)])
    return ChunkBCReplayBuffer(store, seed=seed)


def test_history_tail_padding_and_episode_boundaries() -> None:
    buffer = make_buffer()
    batch = next(buffer.get_epoch_iterator(batch_size=4, observation_horizon=3, action_horizon=4, shuffle=False))

    # 全部真实动作起点都参与训练，包括单步轨迹；末帧观测不作为起点。
    np.testing.assert_array_equal(batch["observations"]["state"][..., 0], [[0, 0, 0], [0, 0, 1], [0, 1, 2], [10, 10, 10]])
    np.testing.assert_array_equal(batch["actions"][..., 0], [[0, 1, 2, 2], [1, 2, 2, 2], [2, 2, 2, 2], [10, 10, 10, 10]])
    np.testing.assert_array_equal(batch["action_valid"], [[1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
    assert batch["actions"].shape == (4, 4, 2)
    assert batch["action_valid"].dtype == np.bool_
    assert batch["observations"]["images"]["front"].dtype == np.uint8
    np.testing.assert_array_equal(batch["observations"]["images"]["front"][..., 0, 0, 0], batch["observations"]["state"][..., 0])

    # 修改已采样 batch 不应污染 Store 中的轨迹。
    batch["actions"][:] = -1
    batch["observations"]["state"][:] = -1
    np.testing.assert_array_equal(buffer.store.get_trajectory(0)["actions"][:, 0], [0, 1, 2])
    np.testing.assert_array_equal(buffer.store.get_trajectory(0)["observations"]["state"][:, 0], [0, 1, 2, 3])


def test_single_step_and_complete_chunk() -> None:
    buffer = make_buffer()
    single = next(buffer.get_epoch_iterator(batch_size=4, observation_horizon=1, action_horizon=1, shuffle=False))
    assert single["actions"].shape == (4, 1, 2)
    assert single["action_valid"].all()
    np.testing.assert_array_equal(single["observations"]["state"][:, 0, 0], single["actions"][:, 0, 0])
    full = next(buffer.get_epoch_iterator(batch_size=1, observation_horizon=1, action_horizon=3, shuffle=False))
    assert full["action_valid"].all()
    np.testing.assert_array_equal(full["actions"][0, :, 0], [0, 1, 2])


def test_epoch_coverage_shuffle_and_drop_last() -> None:
    buffer = make_buffer(seed=7)
    args = {"batch_size": 3, "observation_horizon": 2, "action_horizon": 4}
    batches = list(buffer.get_epoch_iterator(**args))
    assert [len(batch["actions"]) for batch in batches] == [3, 1]
    starts = np.concatenate([batch["actions"][:, 0, 0] for batch in batches])
    np.testing.assert_array_equal(np.sort(starts), [0, 1, 2, 10])
    buffer.seed(7)
    repeated = list(buffer.get_epoch_iterator(**args))
    np.testing.assert_array_equal(np.concatenate([batch["actions"][:, 0, 0] for batch in repeated]), starts)
    assert len(list(buffer.get_epoch_iterator(**args, drop_last=True))) == 1
    assert list(buffer.get_epoch_iterator(batch_size=5, observation_horizon=1, action_horizon=1, drop_last=True)) == []


def test_random_sampling_and_iterator_are_reproducible() -> None:
    buffer = make_buffer(seed=42)
    args = {"batch_size": 32, "observation_horizon": 2, "action_horizon": 4}
    first, second = buffer.sample_chunk(**args), buffer.sample_chunk(**args)
    buffer.seed(42)
    iterator = buffer.get_chunk_iterator(**args)
    np.testing.assert_array_equal(next(iterator)["actions"], first["actions"])
    np.testing.assert_array_equal(next(iterator)["actions"], second["actions"])
    assert set(first["actions"][:, 0, 0]) == {0, 1, 2, 10}
    assert first["action_valid"].any(axis=1).all()


@pytest.mark.parametrize("key", ["batch_size", "observation_horizon", "action_horizon"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_lengths_are_rejected(key: str, value) -> None:
    args = {"batch_size": 1, "observation_horizon": 1, "action_horizon": 1, key: value}
    buffer = make_buffer()
    with pytest.raises(ValueError, match=key):
        buffer.sample_chunk(**args)
    with pytest.raises(ValueError, match=key):
        next(buffer.get_epoch_iterator(**args))


def test_empty_store_is_rejected() -> None:
    buffer = ChunkBCReplayBuffer(InMemoryDataStore(capacity=1))
    args = {"batch_size": 1, "observation_horizon": 1, "action_horizon": 1}
    with pytest.raises(ValueError, match="empty DataStore"):
        buffer.sample_chunk(**args)
    with pytest.raises(ValueError, match="empty DataStore"):
        next(buffer.get_epoch_iterator(**args))
