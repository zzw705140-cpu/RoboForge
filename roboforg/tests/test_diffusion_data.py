import numpy as np
import pytest

from roboforg.data.data_buffer import DataBuffer
from roboforg.data.diffusion_policy_data import DiffusionPolicySampler, DiffusionPolicyBatchConverter


def trajectory(length=20, offset=0):
    steps = np.arange(length + 1) + offset
    return dict(observations=dict(state=steps[:, None],
                                 front=np.broadcast_to(steps[:, None, None, None], (length + 1, 2, 2, 3)).astype(np.uint8)),
                actions=steps[:-1, None], rewards=np.zeros(length), masks=np.ones(length),
                dones=np.arange(length) == length - 1)


def test_alignment_and_both_boundaries():
    sampler = DiffusionPolicySampler(DataBuffer.from_trajectories([trajectory()]))
    first = sampler.build_sample(0, start_step=-1)
    np.testing.assert_array_equal(first['observations']['state'][:, 0], [0, 0])
    np.testing.assert_array_equal(first['actions'][:, 0], [0, *range(15)])
    middle = sampler.build_sample(0, start_step=3)
    np.testing.assert_array_equal(middle['observations']['state'][:, 0], [3, 4])
    np.testing.assert_array_equal(middle['actions'][1:9, 0], np.arange(4, 12))
    last = sampler.build_sample(0, start_step=11)
    np.testing.assert_array_equal(last['actions'][:, 0], [*range(11, 20), *([19] * 7)])
    with pytest.raises(IndexError):
        sampler.build_sample(0, start_step=12)


def test_epoch_and_image_format():
    sampler = DiffusionPolicySampler(DataBuffer.from_trajectories([trajectory(), trajectory(offset=100)]))
    batches = list(sampler.get_epoch_iterator(batch_size=5, shuffle=False))
    assert sum(len(batch['actions']) for batch in batches) == 26
    for batch in batches:
        obs, actions = DiffusionPolicyBatchConverter(image_keys=('front',))(batch)
        assert obs['front'].shape == (len(actions), 2, 2, 2, 3)
        assert obs['front'].dtype == np.uint8
        assert obs['front'] is batch['observations']['front']
        # 每个样本只能属于一条轨迹。
        assert np.all((actions[:, :, 0] < 100).all(1) | (actions[:, :, 0] >= 100).all(1))
    assert sum(len(b['actions']) for b in sampler.get_epoch_iterator(batch_size=5, drop_last=True)) == 25


def test_padding_can_be_replaced():
    def zero_before(values, count):
        return np.concatenate((np.zeros((count, *values.shape[1:]), dtype=values.dtype), values))
    sampler = DiffusionPolicySampler(DataBuffer.from_trajectories([trajectory(offset=10)]), before_padding=zero_before)
    sample = sampler.build_sample(0, start_step=-1)
    np.testing.assert_array_equal(sample['observations']['state'][:, 0], [0, 10])
    assert sample['actions'][0, 0] == 0


def test_short_empty_and_invalid_configuration():
    sampler = DiffusionPolicySampler(DataBuffer.from_trajectories([trajectory(length=2)]))
    with pytest.raises(ValueError, match='No valid'):
        sampler.sample(1)
    with pytest.raises(ValueError):
        DiffusionPolicySampler(DataBuffer(10), n_obs_steps=10)
    with pytest.raises(ValueError):
        DiffusionPolicySampler(DataBuffer(10), horizon=16.5)
    with pytest.raises(ValueError):
        DiffusionPolicySampler(DataBuffer(10)).sample(0)


def test_random_sampling_is_reproducible():
    buffer = DataBuffer.from_trajectories([trajectory()])
    left, right = (DiffusionPolicySampler(buffer, seed=5) for _ in range(2))
    np.testing.assert_array_equal(left.sample(12)['actions'], right.sample(12)['actions'])
