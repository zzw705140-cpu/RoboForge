import jax
import numpy as np
import pytest
from roboforg.agent.diffusion_policy import DataNormalizer


def test_action_identity_and_inverse():
    n = DataNormalizer()
    a = np.linspace(-1, 1, 42, dtype=np.float32).reshape(2, 3, 7)
    np.testing.assert_array_equal(n.normalize_action(a), a)
    np.testing.assert_array_equal(n.denormalize_action(a), a)
    n = DataNormalizer(action_scale=np.arange(1, 8), action_offset=np.arange(7))
    np.testing.assert_allclose(n.denormalize_action(n.normalize_action(a)), a, atol=1e-6)


def test_image_reference_and_jit():
    n = DataNormalizer()
    pixels = np.arange(24, dtype=np.uint8).reshape(1, 2, 2, 2, 3)
    for key in ('front', 'wrist'):
        mean, std = map(np.asarray, n.image_stats[key])
        expected = (pixels.astype(np.float32) / 255 * 2 - 1 - mean) / std
        output = jax.jit(lambda x: n.normalize_image(x, image_key=key))(pixels)
        np.testing.assert_allclose(output, expected, rtol=1e-6)
        assert output.shape == pixels.shape
        assert output.dtype == np.float32
    with pytest.raises(TypeError):
        n.normalize_image(output, image_key='front')


def test_invalid_inputs():
    with pytest.raises(ValueError):
        DataNormalizer(action_scale=np.zeros(7))
    n = DataNormalizer()
    with pytest.raises(ValueError):
        n.normalize_action(np.zeros(6))
    with pytest.raises(ValueError):
        n.normalize_image(np.zeros((2, 2, 3), dtype=np.uint8), image_key='front')
    with pytest.raises(KeyError):
        n.normalize_image(np.zeros((1, 1, 2, 2, 3), dtype=np.uint8), image_key='unknown')


def test_state_ranges_identity_and_no_clipping():
    import json
    from pathlib import Path
    from roboforg.agent.diffusion_policy import data_normalizer
    stats = json.loads(Path(data_normalizer.__file__).with_name('state_stats.json').read_text())
    n = DataNormalizer()
    low, high = np.array(stats['state_min']), np.array(stats['state_max'])
    states = np.stack([low, high, high + (high - low)])[None].astype(np.float32)
    result = np.asarray(jax.jit(n.normalize_state)(states))
    indices = stats['range_indices']
    np.testing.assert_allclose(result[0, 0, indices], -1, atol=2e-6)
    np.testing.assert_allclose(result[0, 1, indices], 1, atol=2e-6)
    np.testing.assert_allclose(result[0, 2, indices], 3, atol=2e-6)
    identity = [1, 2, 3, 4, *range(8, 17)]
    np.testing.assert_array_equal(result[..., identity], states[..., identity])
    np.testing.assert_allclose(result[..., 0], states[..., 0] * 2 - 1)
    assert result.shape == (1, 3, 23)
    with pytest.raises(ValueError):
        n.normalize_state(np.zeros((1, 2, 22)))


def test_state_constant_and_invalid_parameters(tmp_path):
    import json
    path = tmp_path / 'state.json'
    scale, offset = np.ones(23), np.zeros(23)
    offset[5] = -0.25
    path.write_text(json.dumps(dict(state_scale=scale.tolist(), state_offset=offset.tolist())))
    n = DataNormalizer(state_stats_path=path)
    state = np.zeros((1, 2, 23), dtype=np.float32)
    state[..., 5] = 0.25
    np.testing.assert_array_equal(n.normalize_state(state)[..., 5], 0)
    state[..., 5] += 1e-6
    np.testing.assert_allclose(n.normalize_state(state)[..., 5], 1e-6, atol=2e-8)
    scale[5] = 0
    path.write_text(json.dumps(dict(state_scale=scale.tolist(), state_offset=offset.tolist())))
    with pytest.raises(ValueError):
        DataNormalizer(state_stats_path=path)
