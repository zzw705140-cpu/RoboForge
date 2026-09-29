"""临时检查视觉编码器、条件 U-Net 及关键基础计算。"""
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict

from roboforg.networks.diffusion_policy.diffusion_unet1d import DiffusionUNet, SequenceConvBlock
from roboforg.networks.diffusion_policy.observation_encoder import ObservationEncoder


# 串联视觉编码器和 U-Net，模拟算法层之后采用的真实数据流。
class CombinedDiffusionNetwork(nn.Module):
    @nn.compact
    def __call__(self, observations, noisy_actions, timestep):
        observation_features = ObservationEncoder(name="observation_encoder")(observations)
        global_condition = observation_features.reshape(observation_features.shape[0], -1)
        return DiffusionUNet(action_dim=7, global_condition_dim=2094, diffusion_step_embedding_dim=8,
                             down_channels=(8, 16, 32), n_groups=4, name="diffusion_unet")(noisy_actions, timestep, global_condition)


# 单独建立与 U-Net 相同的上下采样层，便于读取参数并进行数值对照。
class SamplingLayers(nn.Module):
    @nn.compact
    def __call__(self, x):
        down = nn.Conv(4, (3,), strides=(2,), padding=((1, 1),), name="downsample")(x)
        up = nn.ConvTranspose(4, (4,), strides=(2,), padding="SAME", name="upsample")(down)
        return down, up


# 构造与实际接口一致的两帧、双相机观测。
def make_observations():
    image = jnp.linspace(-1.0, 1.0, 32 * 32 * 3, dtype=jnp.float32).reshape(1, 1, 32, 32, 3)
    return {"front": jnp.repeat(image, 2, axis=1), "wrist": jnp.repeat(image[..., ::-1, :], 2, axis=1),
            "state": jnp.linspace(-1.0, 1.0, 46, dtype=jnp.float32).reshape(1, 2, 23)}


# 使用直接循环计算普通一维卷积，作为 Flax 卷积的独立参考。
def conv1d_reference(x, kernel, bias, stride=1, padding=0):
    x = np.pad(x, ((0, 0), (padding, padding), (0, 0)))
    output_length = (x.shape[1] - kernel.shape[0]) // stride + 1
    output = np.empty((x.shape[0], output_length, kernel.shape[-1]), dtype=np.float32)
    for step in range(output_length):
        window = x[:, step * stride:step * stride + kernel.shape[0], :]
        output[:, step, :] = np.einsum("bki,kio->bo", window, kernel) + bias
    return output


# 按 JAX ConvTranspose 默认卷积核约定直接计算转置卷积。
def conv_transpose1d_reference(x, kernel, bias, stride=2, padding=1):
    output_length = (x.shape[1] - 1) * stride - 2 * padding + kernel.shape[0]
    output = np.broadcast_to(bias, (x.shape[0], output_length, bias.shape[0])).copy()
    for step in range(x.shape[1]):
        for offset in range(kernel.shape[0]):
            output_step = step * stride - padding + kernel.shape[0] - 1 - offset
            if 0 <= output_step < output_length:
                output[:, output_step, :] += np.einsum("bi,io->bo", x[:, step, :], kernel[offset])
    return output


# 检查真实观测特征能够送入 U-Net，并保持动作序列形状。
def test_joint_forward():
    model = CombinedDiffusionNetwork()
    observations = make_observations()
    noisy_actions = jnp.ones((1, 16, 7), dtype=jnp.float32)
    variables = model.init(jax.random.PRNGKey(0), observations, noisy_actions, jnp.array([10]))
    output = model.apply(variables, observations, noisy_actions, jnp.array([10]))
    assert output.shape == (1, 16, 7)
    assert np.isfinite(np.asarray(output)).all()


# 检查损失梯度能够同时到达视觉骨干、时间映射层和 U-Net 输出层。
def test_joint_gradient():
    model = CombinedDiffusionNetwork()
    observations = make_observations()
    noisy_actions = jnp.ones((1, 16, 7), dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(1), observations, noisy_actions, jnp.array([10]))["params"]
    loss = lambda current: jnp.mean(model.apply({"params": current}, observations, noisy_actions, jnp.array([10])) ** 2)
    gradients = flatten_dict(jax.grad(loss)(params))

    backbone_norm = sum(float(jnp.linalg.norm(value)) for key, value in gradients.items() if key[:2] == ("observation_encoder", "backbone"))
    time_norm = float(jnp.linalg.norm(gradients[("diffusion_unet", "time_projection_1", "kernel")]))
    output_norm = float(jnp.linalg.norm(gradients[("diffusion_unet", "output_projection", "kernel")]))
    assert backbone_norm > 0 and time_norm > 0 and output_norm > 0


# 对照直接公式检查卷积填充、GroupNorm 和 Mish 的组合计算。
def test_conv_group_norm_matches_reference():
    x = jnp.linspace(-1.0, 1.0, 2 * 9 * 4, dtype=jnp.float32).reshape(2, 9, 4)
    block = SequenceConvBlock(out_channels=8, kernel_size=5, n_groups=4)
    variables = block.init(jax.random.PRNGKey(2), x)
    output = np.asarray(block.apply(variables, x))
    params = variables["params"]

    convolved = conv1d_reference(np.asarray(x), np.asarray(params["conv"]["kernel"]), np.asarray(params["conv"]["bias"]), padding=2)
    grouped = convolved.reshape(2, 9, 4, 2)
    mean = grouped.mean(axis=(1, 3), keepdims=True)
    variance = grouped.var(axis=(1, 3), keepdims=True)
    normalized = ((grouped - mean) / np.sqrt(variance + 1e-5)).reshape(2, 9, 8)
    normalized = normalized * np.asarray(params["group_norm"]["scale"]) + np.asarray(params["group_norm"]["bias"])
    reference = normalized * np.tanh(np.logaddexp(0.0, normalized))
    np.testing.assert_allclose(output, reference, rtol=2e-5, atol=2e-5)


# 对照直接公式检查下采样和上采样的数值及时间长度。
def test_sampling_matches_reference():
    x = jnp.linspace(-1.0, 1.0, 2 * 16 * 4, dtype=jnp.float32).reshape(2, 16, 4)
    layers = SamplingLayers()
    variables = layers.init(jax.random.PRNGKey(3), x)
    down, up = layers.apply(variables, x)
    params = variables["params"]

    down_reference = conv1d_reference(np.asarray(x), np.asarray(params["downsample"]["kernel"]), np.asarray(params["downsample"]["bias"]), stride=2, padding=1)
    up_reference = conv_transpose1d_reference(down_reference, np.asarray(params["upsample"]["kernel"]), np.asarray(params["upsample"]["bias"]))
    assert down.shape == (2, 8, 4) and up.shape == (2, 16, 4)
    np.testing.assert_allclose(down, down_reference, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(up, up_reference, rtol=1e-5, atol=1e-5)
