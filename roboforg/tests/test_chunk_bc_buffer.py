import numpy as np
import pytest
from flax.core import frozen_dict

from roboforg.data.chunk_bc_buffer import ChunkBCBuffer


def _trajectory(start: int, length: int) -> dict:
    steps = np.arange(start, start + length, dtype=np.float32)
    return {
        "observations": {
            "state": np.arange(start, start + length + 1, dtype=np.float32)[:, None],
            "camera": np.arange(start, start + length + 1, dtype=np.uint8)[:, None, None, None],
        },
        "actions": np.stack((steps, steps + 0.25), axis=-1),
        "rewards": steps.copy(),
        "masks": np.asarray([1.0] * (length - 1) + [0.0], dtype=np.float32),
        "dones": np.asarray([False] * (length - 1) + [True]),
    }


def test_num_valid_chunks_per_trajectory() -> None:
    buffer = ChunkBCBuffer.from_trajectories(
        [_trajectory(0, 4), _trajectory(10, 2), _trajectory(20, 1)]
    )

    np.testing.assert_array_equal(
        buffer._num_valid_chunks_per_trajectory(action_horizon=2),
        np.asarray([3, 1, 0]),
    )


def test_build_chunk_sample_left_pads_observations_and_flattens_actions() -> None:
    trajectory = _trajectory(0, 4)

    sample = ChunkBCBuffer._build_chunk_sample(
        trajectory,
        start_step=1,
        observation_horizon=4,
        action_horizon=2,
    )

    np.testing.assert_array_equal(sample["observations"]["state"][:, 0], [0, 0, 0, 1])
    np.testing.assert_array_equal(sample["observations"]["camera"][:, 0, 0, 0], [0, 0, 0, 1])
    np.testing.assert_allclose(sample["actions"], [[1.0, 1.25], [2.0, 2.25]])


def test_sample_chunk_returns_valid_batched_chunks() -> None:
    buffer = ChunkBCBuffer.from_trajectories(
        [_trajectory(0, 4), _trajectory(10, 3)], seed=5
    )

    batch = buffer.sample_chunk(
        batch_size=16,
        observation_horizon=3,
        action_horizon=2,
    )

    assert isinstance(batch, frozen_dict.FrozenDict)
    assert batch["observations"]["state"].shape == (16, 3, 1)
    assert batch["observations"]["camera"].shape == (16, 3, 1, 1, 1)
    assert batch["actions"].shape == (16, 2, 2)
    np.testing.assert_allclose(batch["actions"][:, 1, 0], batch["actions"][:, 0, 0] + 1.0)
    assert set(batch["actions"][:, 0, 0]).issubset({0.0, 1.0, 2.0, 10.0, 11.0})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"batch_size": 0, "observation_horizon": 1, "action_horizon": 1}, "batch_size"),
        ({"batch_size": 1, "observation_horizon": 0, "action_horizon": 1}, "observation_horizon"),
        ({"batch_size": 1, "observation_horizon": 1, "action_horizon": 0}, "action_horizon"),
        ({"batch_size": 1, "observation_horizon": 1, "action_horizon": 5}, "No stored trajectory"),
    ],
)
def test_sample_chunk_rejects_invalid_requests(kwargs: dict, message: str) -> None:
    buffer = ChunkBCBuffer.from_trajectories([_trajectory(0, 4)])

    with pytest.raises(ValueError, match=message):
        buffer.sample_chunk(**kwargs)


def test_epoch_iterator_visits_each_valid_chunk_once() -> None:
    buffer = ChunkBCBuffer.from_trajectories([_trajectory(0, 4), _trajectory(10, 3)])

    batches = list(
        buffer.get_epoch_iterator(
            batch_size=2,
            observation_horizon=1,
            action_horizon=2,
            shuffle=False,
            drop_last=False,
        )
    )

    assert [batch["actions"].shape[0] for batch in batches] == [2, 2, 1]
    sampled_starts = np.concatenate([np.asarray(batch["actions"])[:, 0, 0] for batch in batches])
    np.testing.assert_array_equal(sampled_starts, [0.0, 1.0, 2.0, 10.0, 11.0])


def test_epoch_iterator_can_drop_incomplete_final_batch() -> None:
    buffer = ChunkBCBuffer.from_trajectories([_trajectory(0, 4), _trajectory(10, 3)])

    batches = list(
        buffer.get_epoch_iterator(
            batch_size=2,
            observation_horizon=1,
            action_horizon=2,
            shuffle=False,
            drop_last=True,
        )
    )

    assert len(batches) == 2
    assert sum(batch["actions"].shape[0] for batch in batches) == 4


def test_chunk_iterator_continuously_supplies_batches() -> None:
    buffer = ChunkBCBuffer.from_trajectories([_trajectory(0, 4)], seed=3)
    iterator = buffer.get_chunk_iterator(
        sample_args={"batch_size": 2, "observation_horizon": 2, "action_horizon": 2},
        queue_size=2,
    )

    first_batch = next(iterator)
    second_batch = next(iterator)

    assert first_batch["observations"]["state"].shape == (2, 2, 1)
    assert first_batch["actions"].shape == (2, 2, 2)
    assert second_batch["actions"].shape == (2, 2, 2)
