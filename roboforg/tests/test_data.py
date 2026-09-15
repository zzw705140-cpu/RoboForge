import numpy as np
import pytest
from flax.core import frozen_dict

from roboforg.data.data_buffer import DataBuffer


def _transition(step: int, *, done: bool) -> dict:
    return {
        "observations": {
            "state": np.asarray([step, step + 0.5], dtype=np.float32),
            "images": {"front": np.full((2, 2, 1), step, dtype=np.uint8)},
        },
        "actions": np.asarray([step, -step], dtype=np.float32),
        "rewards": float(step),
        "dones": done,
        "terminated": done,
        "next_observations": {
            "state": np.asarray([step + 1, step + 1.5], dtype=np.float32),
            "images": {"front": np.full((2, 2, 1), step + 1, dtype=np.uint8)},
        },
    }


def _trajectory(start: int, length: int) -> dict:
    return {
        "observations": {
            "state": np.asarray(
                [[step, step + 0.5] for step in range(start, start + length + 1)],
                dtype=np.float32,
            )
        },
        "actions": np.asarray(
            [[step, -step] for step in range(start, start + length)],
            dtype=np.float32,
        ),
        "rewards": np.arange(start, start + length, dtype=np.float32),
        "masks": np.asarray([1.0] * (length - 1) + [0.0], dtype=np.float32),
        "dones": np.asarray([False] * (length - 1) + [True]),
    }


def test_insert_builds_complete_trajectories() -> None:
    transitions = [
        _transition(0, done=False),
        _transition(1, done=True),
        _transition(10, done=True),
    ]

    buffer = DataBuffer.from_transitions(transitions, seed=7)

    assert len(buffer) == 3
    assert buffer.num_trajectories == 2
    assert buffer.has_incomplete_episode is False
    np.testing.assert_array_equal(buffer.trajectory_lengths, np.asarray([2, 1]))
    first = buffer.get_trajectory(0)
    assert isinstance(first, frozen_dict.FrozenDict)
    assert first["observations"]["state"].shape == (3, 2)
    assert first["actions"].shape == (2, 2)


def test_incomplete_and_invalid_trajectories_are_rejected() -> None:
    with pytest.raises(ValueError, match="dones=True"):
        DataBuffer.from_transitions([_transition(0, done=False)])

    invalid = _trajectory(0, 3)
    invalid["dones"] = np.asarray([False, True, True])
    with pytest.raises(ValueError, match="before the final"):
        DataBuffer.from_trajectories([invalid])


def test_capacity_evicts_the_oldest_complete_trajectory() -> None:
    buffer = DataBuffer(capacity=3)
    buffer.add_trajectory(_trajectory(0, 2))
    buffer.add_trajectory(_trajectory(10, 2))

    assert len(buffer) == 2
    assert buffer.num_trajectories == 1
    np.testing.assert_array_equal(buffer.get_trajectory(0)["actions"][0], [10.0, -10.0])


def test_sample_returns_consistent_transition_batch() -> None:
    buffer = DataBuffer.from_trajectories([_trajectory(0, 3), _trajectory(10, 2)], seed=4)

    batch = buffer.sample(batch_size=8)

    assert isinstance(batch, frozen_dict.FrozenDict)
    assert batch["actions"].shape == (8, 2)
    assert batch["rewards"].shape == (8,)
    assert batch["observations"]["state"].shape == (8, 2)
    np.testing.assert_allclose(
        batch["next_observations"]["state"] - batch["observations"]["state"],
        np.ones((8, 2), dtype=np.float32),
    )


def test_iterator_continuously_supplies_device_batches() -> None:
    buffer = DataBuffer.from_trajectories([_trajectory(0, 3)], seed=1)
    iterator = buffer.get_iterator(sample_args={"batch_size": 2}, queue_size=2)

    first_batch = next(iterator)
    second_batch = next(iterator)

    assert first_batch["actions"].shape == (2, 2)
    assert second_batch["actions"].shape == (2, 2)
