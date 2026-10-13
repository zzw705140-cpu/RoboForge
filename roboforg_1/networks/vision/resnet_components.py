"""Shared ResNet blocks, conditioning, coordinates, and normalization."""

from typing import Any, Callable, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

ModuleDef = Any

# ------------------------------------------------------------------------- #
#                    基础空间处理与归一化组件                               #
# ------------------------------------------------------------------------- #

class FilmConditioning(nn.Module):
    """
    使用外部条件对卷积特征图执行逐通道 FiLM 调制

    输入: 
        conv_filters: [batch, height, width, channels]
        conditioning: [batch, conditioning_dim]

    输出: 
        与 conv_filters 形状相同的调制特征图

    两个条件投影均使用零初始化, 因此初始化时该模块是恒等映射
    """

    @nn.compact
    def __call__(
        self,
        conv_filters: jax.Array,
        conditioning: jax.Array,
    ) -> jax.Array:
        """使用条件生成逐通道调制量, 调整卷积特征."""

        projected_cond_add = nn.Dense(
            features=conv_filters.shape[-1],
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(conditioning)
        projected_cond_mult = nn.Dense(
            features=conv_filters.shape[-1],
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(conditioning)

        projected_cond_add = projected_cond_add[..., None, None, :]
        projected_cond_mult = projected_cond_mult[..., None, None, :]

        return conv_filters * (1.0 + projected_cond_add) + projected_cond_mult


class AddSpatialCoordinates(nn.Module):
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        """在视觉特征末维拼接归一化的空间坐标."""

        grid = jnp.array(
            np.stack(
                np.meshgrid(*[np.arange(s) / (s - 1) * 2 - 1 for s in x.shape[-3:-1]]),
                axis=-1,
            ),
            dtype=self.dtype,
        ).transpose(1, 0, 2)

        if x.ndim == 4:
            grid = jnp.broadcast_to(grid, [x.shape[0], *grid.shape])

        return jnp.concatenate([x, grid], axis=-1)


class MyGroupNorm(nn.GroupNorm):
    def __call__(self, x):
        """对单样本或批量特征执行 GroupNorm."""

        if x.ndim == 3:
            x = x[jnp.newaxis]
            x = super().__call__(x)
            return x[0]
        else:
            return super().__call__(x)


# ------------------------------------------------------------------------- #
#                         ResNet 残差块                                     #
# ------------------------------------------------------------------------- #

class ResNetBlock(nn.Module):
    """ResNet block"""

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        """执行基础残差块, 在形状变化时投影残差分支."""

        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm()(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1), self.strides, name="conv_proj")(residual)
            residual = self.norm(name="norm_proj")(residual)

        return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
    """Bottleneck ResNet block"""

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        """执行瓶颈残差块并融合残差分支."""

        residual = x
        y = self.conv(self.filters, (1, 1))(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3), self.strides)(y)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters * 4, (1, 1))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            residual = self.conv(
                self.filters * 4, (1, 1), self.strides, name="conv_proj"
            )(residual)
            residual = self.norm(name="norm_proj")(residual)

        return self.act(residual + y)
