"""条件 MLP 的输入契约、条件信号和训练模式测试。"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from roboforg_1.networks.mlp import MLPWithCondition


@pytest.mark.parametrize("batched", [False, True])
def test_condition_controls_output_and_receives_gradients(batched) -> None:
    """指定线性权重，确认主输入、条件的拼接顺序和梯度传播均正确。"""
    network = MLPWithCondition(hidden_dims=(1,))
    x = jnp.array([1.0, 2.0])
    condition = jnp.array([3.0])
    if batched:
        x = jnp.stack([x, x * 2])
        condition = jnp.stack([condition, condition * 2])
    variables = network.init(jax.random.PRNGKey(0), x, condition)
    variables["params"]["MLP_0"]["Dense_0"]["kernel"] = jnp.array([[1.0], [10.0], [100.0]])
    variables["params"]["MLP_0"]["Dense_0"]["bias"] = jnp.zeros(1)
    output = network.apply(variables, x, condition)
    np.testing.assert_allclose(output, [[321.0], [642.0]] if batched else [321.0])
    gradient = jax.grad(lambda value: network.apply(variables, x, value).sum())(condition)
    np.testing.assert_allclose(gradient, jnp.full_like(condition, 100))


def test_layer_configuration_and_dropout_train_mode() -> None:
    network = MLPWithCondition(hidden_dims=(16, 3), activations="gelu", use_layer_norm=True, dropout_rate=0.5)
    x, condition = jnp.ones((8, 2)), jnp.ones((8, 4))
    variables = network.init(jax.random.PRNGKey(1), x, condition)
    evaluated = network.apply(variables, x, condition, train=False)
    assert evaluated.shape == (8, 3)
    assert np.isfinite(evaluated).all()
    np.testing.assert_array_equal(evaluated, network.apply(variables, x, condition, train=False))
    trained = network.apply(variables, x, condition, train=True, rngs={"dropout": jax.random.PRNGKey(2)})
    assert trained.shape == evaluated.shape
    assert np.isfinite(trained).all()
    assert not np.allclose(trained, evaluated)
    compiled = jax.jit(lambda a, b: network.apply(variables, a, b))(x, condition)
    np.testing.assert_allclose(compiled, evaluated, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shapes", [((2,), (1, 3)), ((2, 3), (1, 4)), ((2, 3, 4), (2, 3, 5)), ((), ())])
def test_incompatible_input_shapes_are_rejected(shapes) -> None:
    network = MLPWithCondition(hidden_dims=(4, 2), activations=nn.relu)
    with pytest.raises(ValueError):
        network.init(jax.random.PRNGKey(0), jnp.zeros(shapes[0]), jnp.zeros(shapes[1]))
