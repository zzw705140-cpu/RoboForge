import jax
import jax.numpy as jnp
import numpy as np
import pytest

from roboforg.agent.diffusion_policy import DataNormalizer
from roboforg.networks.diffusion_policy import ObservationEncoder, load_pretrained_resnet10


def _observations():
    normalizer = DataNormalizer()
    image = np.arange(32 * 32 * 3, dtype=np.uint32).reshape(32, 32, 3).astype(np.uint8)
    front = np.broadcast_to(image, (1, 2, 32, 32, 3))
    wrist = np.broadcast_to(np.flip(image, axis=1), (1, 2, 32, 32, 3))
    return {
        'front': normalizer.normalize_image(front, image_key='front'),
        'wrist': normalizer.normalize_image(wrist, image_key='wrist'),
        'state': normalizer.normalize_state(np.zeros((1, 2, 23), np.float32)),
    }


def test_shared_backbone_and_pretrained_checkpoint():
    obs = _observations()
    model = ObservationEncoder()
    params = model.init(jax.random.PRNGKey(0), obs)['params']
    assert set(params) == {'backbone'}
    output = model.apply({'params': params}, obs)
    assert output.shape == (1, 2, 1047)
    np.testing.assert_array_equal(output[..., -23:], obs['state'])

    # 同一图像送入两路相机时，ResNet 特征完全一致，说明两路使用同一组参数。
    same = {**obs, 'wrist': obs['front']}
    same_output = model.apply({'params': params}, same)
    np.testing.assert_allclose(same_output[..., :512], same_output[..., 512:1024])

    loaded = load_pretrained_resnet10(params, '.resnet_params/resnet10_params.pkl')
    pretrained_output = model.apply({'params': loaded}, obs)
    assert pretrained_output.shape == output.shape
    assert np.isfinite(np.asarray(pretrained_output)).all()
    assert not np.array_equal(np.asarray(output[..., :1024]), np.asarray(pretrained_output[..., :1024]))


def test_configurable_inputs_and_validation():
    model = ObservationEncoder(camera_keys=('front',), state_dim=4, n_obs_steps=1)
    obs = {'front': jnp.zeros((1, 1, 32, 32, 3)), 'state': jnp.ones((1, 1, 4))}
    params = model.init(jax.random.PRNGKey(1), obs)
    assert model.apply(params, obs).shape == (1, 1, 516)
    with pytest.raises(ValueError, match='state'):
        model.apply(params, {**obs, 'state': jnp.zeros((1, 1, 5))})
    with pytest.raises(TypeError, match='normalized'):
        model.apply(params, {**obs, 'front': jnp.zeros((1, 1, 32, 32, 3), dtype=jnp.uint8)})
