import jax
import jax.numpy as jnp
import numpy as np

from roboforg.networks.diffusion_policy import ConditionedResBlock, DiffusionUNet, SequenceConvBlock, diffusion_step_embedding


def test_diffusion_step_embedding_uses_fixed_formula():
    embedding = diffusion_step_embedding(jnp.array([0.0, 1.0]), 8)
    assert embedding.shape == (2, 8)
    np.testing.assert_allclose(embedding[0, :4], 0.0, atol=1e-6)
    np.testing.assert_allclose(embedding[0, 4:], 1.0, atol=1e-6)


def test_reusable_blocks_preserve_sequence_length():
    x = jnp.ones((2, 16, 4), dtype=jnp.float32)
    conv = SequenceConvBlock(out_channels=8, kernel_size=5, n_groups=4)
    conv_params = conv.init(jax.random.PRNGKey(0), x)
    assert conv.apply(conv_params, x).shape == (2, 16, 8)

    condition = jnp.ones((2, 6), dtype=jnp.float32)
    block = ConditionedResBlock(out_channels=8, condition_dim=6, n_groups=4)
    block_params = block.init(jax.random.PRNGKey(1), x, condition)
    assert block.apply(block_params, x, condition).shape == (2, 16, 8)


def test_unet_shape_and_conditions_affect_output():
    model = DiffusionUNet(action_dim=3, global_condition_dim=6, diffusion_step_embedding_dim=8, down_channels=(8, 16, 32), kernel_size=5, n_groups=4)
    sample = jnp.ones((2, 16, 3), dtype=jnp.float32)
    condition = jnp.zeros((2, 6), dtype=jnp.float32)
    variables = model.init(jax.random.PRNGKey(2), sample, jnp.array([1, 1]), condition)

    output = model.apply(variables, sample, jnp.array([1, 1]), condition)
    changed_time = model.apply(variables, sample, jnp.array([2, 2]), condition)
    changed_condition = model.apply(variables, sample, jnp.array([1, 1]), condition.at[:, 0].set(1.0))

    assert output.shape == sample.shape
    assert not np.allclose(output, changed_time)
    assert not np.allclose(output, changed_condition)
