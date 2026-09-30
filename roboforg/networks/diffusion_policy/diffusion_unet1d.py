"""Diffusion Policy 的一维条件 U-Net。

输入和输出均采用 [batch, time, channels]，卷积沿动作序列的时间维进行。
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


# 使用固定正弦公式将扩散步编码为向量；该函数没有可训练参数。
def diffusion_step_embedding(timestep: jax.Array, embedding_dim: int) -> jax.Array:

    # 检查编码维度至少为4且为偶数
    if embedding_dim < 4 or embedding_dim % 2 != 0:
        raise ValueError("embedding_dim must be an even integer greater than or equal to 4.")
    timestep = jnp.asarray(timestep, dtype=jnp.float32)
    # 检查步数数组是否为B
    if timestep.ndim != 1:
        raise ValueError("timestep must have shape [B].")

    # 将编码维度分为两半，一半用于正弦编码，一半用于余弦编码。
    half_dim = embedding_dim // 2
    # 生成一组不同频率，用于让步数编码包含不同尺度的信息
    frequencies = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * (-math.log(10000.0) / (half_dim - 1)))
    angles = timestep[:, None] * frequencies[None, :]

    # 计算角度的正余弦值，并拼接成最终编码
    return jnp.concatenate((jnp.sin(angles), jnp.cos(angles)), axis=-1)


# 定义Mish激活值，返回变换后的结果（输入输出都是数组）
def _mish(x: jax.Array) -> jax.Array:
    return x * jnp.tanh(jax.nn.softplus(x))


"""
训练数据
→ 动作加噪
→ DiffusionUNet
    → 下采样阶段
        → ConditionedResBlock
            → SequenceConvBlock
            → 注入观测与扩散步条件
            → SequenceConvBlock
    → 中间阶段
        → ConditionedResBlock
            → SequenceConvBlock
    → 上采样阶段
        → ConditionedResBlock
            → SequenceConvBlock
→ 预测噪声
→ 计算损失
"""


# 对动作序列依次执行一维卷积、分组归一化和 Mish 激活。
# 在 U-Net 每次前向传播时，提取当前带噪动作序列的内部特征中相邻时间步之间的局部关系
class SequenceConvBlock(nn.Module):
    out_channels: int         #指定卷积层输出的通道数
    kernel_size: int = 5      #卷积核大小
    n_groups: int = 8         #归一化时分为8组

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        if x.ndim != 3:
            raise ValueError("x must have shape [B, T, C].")
        # 检查输出通道数和卷积核大小是否有效，并要求卷积核大小为奇数
        if self.out_channels <= 0 or self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("out_channels and kernel_size must be positive; kernel_size must be odd.")
        # 检查分组是否为正数，确认输出通道数是否能被分组数整除
        if self.n_groups <= 0 or self.out_channels % self.n_groups != 0:
            raise ValueError("out_channels must be divisible by n_groups.")

        # 设置卷积填充以保持时间序列长度，然后执行一维卷积，分组归一化和Mish激活
        padding = ((self.kernel_size // 2, self.kernel_size // 2),)
        x = nn.Conv(self.out_channels, (self.kernel_size,), padding=padding, name="conv")(x)
        x = nn.GroupNorm(num_groups=self.n_groups, epsilon=1e-5, name="group_norm")(x)
        return _mish(x)


# 将扩散步和观测条件注入两层卷积，并添加残差连接。
class ConditionedResBlock(nn.Module):
    out_channels:  int            #卷积核输出通道数
    condition_dim: int            #指定条件向量的特征维度
    kernel_size: int = 5
    n_groups:    int = 8
    predict_scale: bool = True    #是否让条件同时生成缩放和偏移参数
 
    @nn.compact
    def __call__(self, x: jax.Array, condition: jax.Array) -> jax.Array:

        # 检查数据数组的格式维度
        if x.ndim != 3 or condition.ndim != 2 or x.shape[0] != condition.shape[0]:
            raise ValueError("x and condition must have shapes [B, T, C] and [B, condition_dim].")
        if condition.shape[-1] != self.condition_dim:
            raise ValueError("condition has an incompatible feature dimension.")

        # 保存原始输入，然后用于残差连接
        residual = x
        # 对序列特征做一维卷积，分组归一化和Mish，然后调整通道数
        x = SequenceConvBlock(self.out_channels, self.kernel_size, self.n_groups, name="conv_block_1")(x)

        # 条件网络为每个输出通道生成缩放和偏移。
        condition_channels = self.out_channels * 2 if self.predict_scale else self.out_channels
        embedding = nn.Dense(condition_channels, name="condition_projection")(_mish(condition))
        embedding = embedding[:, None, :]
        if self.predict_scale:
            # 沿通道维把条件投影结果分成缩放量scale和偏移量bias
            scale, bias = jnp.split(embedding, 2, axis=-1)
            x = scale * x + bias
        else:
            # 将条件向量加到序列特征上，让条件影响每个时间位置
            x = x + embedding

        # 第二次卷积与残差连接
        x = SequenceConvBlock(self.out_channels, self.kernel_size, self.n_groups, name="conv_block_2")(x)
        if residual.shape[-1] != self.out_channels:
            residual = nn.Conv(self.out_channels, (1,), name="residual_projection")(residual)

        # 返回残差连接结果
        return x + residual


# 通过下采样、中间残差块、上采样和跳跃连接预测动作序列中的噪声。
class DiffusionUNet(nn.Module):
    action_dim: int
    global_condition_dim: int
    diffusion_step_embedding_dim: int = 128
    down_channels: Sequence[int] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    predict_scale: bool = True

    @nn.compact
    def __call__(self, sample: jax.Array, timestep: jax.Array | int | float, global_condition: jax.Array) -> jax.Array:
        if sample.ndim != 3 or sample.shape[-1] != self.action_dim:
            raise ValueError("sample must have shape [B, T, action_dim].")
        if global_condition.ndim != 2 or global_condition.shape != (sample.shape[0], self.global_condition_dim):
            raise ValueError("global_condition must have shape [B, global_condition_dim].")
        if len(self.down_channels) < 2 or any(channel <= 0 for channel in self.down_channels):
            raise ValueError("down_channels must contain at least two positive channel sizes.")
        if any(channel % self.n_groups != 0 for channel in self.down_channels):
            raise ValueError("every down channel size must be divisible by n_groups.")
        divisor = 2 ** (len(self.down_channels) - 1)
        if sample.shape[1] % divisor != 0:
            raise ValueError(f"sample time dimension must be divisible by {divisor}.")

        # 将标量扩散步扩展到整个 batch，再经过可训练的两层映射。
        timestep = jnp.asarray(timestep)
        if timestep.ndim == 0:
            timestep = jnp.broadcast_to(timestep, (sample.shape[0],))
        elif timestep.ndim == 1 and timestep.shape == (1,) and sample.shape[0] != 1:
            timestep = jnp.broadcast_to(timestep, (sample.shape[0],))
        elif timestep.ndim != 1 or timestep.shape[0] != sample.shape[0]:
            raise ValueError("timestep must be a scalar or have shape [B].")

        time_features = diffusion_step_embedding(timestep, self.diffusion_step_embedding_dim)
        time_features = nn.Dense(self.diffusion_step_embedding_dim * 4, name="time_projection_1")(time_features)
        time_features = _mish(time_features)
        time_features = nn.Dense(self.diffusion_step_embedding_dim, name="time_projection_2")(time_features)
        condition     = jnp.concatenate((time_features, global_condition), axis=-1)
        condition_dim = self.diffusion_step_embedding_dim + self.global_condition_dim

        # 下采样路径保存各阶段输出，供上采样路径建立跳跃连接。
        x = sample
        skips = []
        for index, output_channels in enumerate(self.down_channels):
            x = ConditionedResBlock(output_channels, condition_dim, self.kernel_size, self.n_groups, self.predict_scale, name=f"down_res_block_{index}_1")(x, condition)
            x = ConditionedResBlock(output_channels, condition_dim, self.kernel_size, self.n_groups, self.predict_scale, name=f"down_res_block_{index}_2")(x, condition)
            skips.append(x)
            if index < len(self.down_channels) - 1:
                # 下采样
                x = nn.Conv(output_channels, (3,), strides=(2,), padding=((1, 1),), name=f"downsample_{index}")(x)

        # 在序列最短的位置继续处理条件特征。
        middle_channels = self.down_channels[-1]
        x = ConditionedResBlock(middle_channels, condition_dim, self.kernel_size, self.n_groups, self.predict_scale, name="middle_res_block_1")(x, condition)
        x = ConditionedResBlock(middle_channels, condition_dim, self.kernel_size, self.n_groups, self.predict_scale, name="middle_res_block_2")(x, condition)

        # 与对应的下采样特征拼接，逐级恢复动作序列长度。
        for index, output_channels in enumerate(reversed(self.down_channels[:-1])):
            x = jnp.concatenate((x, skips.pop()), axis=-1)
            x = ConditionedResBlock(output_channels, condition_dim, self.kernel_size, self.n_groups, self.predict_scale, name=f"up_res_block_{index}_1")(x, condition)
            x = ConditionedResBlock(output_channels, condition_dim, self.kernel_size, self.n_groups, self.predict_scale, name=f"up_res_block_{index}_2")(x, condition)
            # 上采样
            x = nn.ConvTranspose(output_channels, (4,), strides=(2,), padding="SAME", name=f"upsample_{index}")(x)

        # 将最终特征映射回每一步动作对应的噪声维度。
        x = SequenceConvBlock(self.down_channels[0], self.kernel_size, self.n_groups, name="final_conv_block")(x)
        return nn.Conv(self.action_dim, (1,), name="output_projection")(x)
