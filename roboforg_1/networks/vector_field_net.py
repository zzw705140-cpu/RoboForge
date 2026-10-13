"""根据观测特征, 当前流状态和流时间预测条件速度."""

import flax.linen as nn
import jax
import jax.numpy as jnp


class VectorFieldNet(nn.Module):
    """根据编码后的观测, 当前流状态和时间预测速度.

    Args:
        time_embedding: 将标量流时间 t 编码为特征向量的 Flax 模块.
        network: 条件主干, 接口为 network(x_t, condition, train=train).

    输入: 
        单样本: features [feature_dim], x_t [flow_dim], t [].
        批量样本: features [B, feature_dim], x_t [B, flow_dim], t [B].

    输出: 
        与 x_t 形状相同的速度: [flow_dim] 或 [B, flow_dim].
    """

    time_embedding: nn.Module
    network: nn.Module

    @nn.compact
    def __call__(self, features: jax.Array, x_t: jax.Array, t: jax.Array, train: bool = False) -> jax.Array:
        """检查输入布局, 组合条件, 并调用主干预测速度."""

        """
        t: [B]
            ↓ SinusoidalTimeEmbedding
        时间特征: [B,E] ─┐
        观测特征: [B,F] ─┘
                ↓ Concatenate
        条件特征:       [B,F+E] ─┐
        当前流状态 x_t:  [B,A]  ─┘
                ↓ Concatenate(MLPWithCondition 内部)
            [B,A+F+E]
                ↓ Dense → 激活函数
                隐藏层
                ↓ Dense → 激活函数
                隐藏层
                ↓ Dense(A)
        速度 velocity: [B,A]
        """

        # 观测特征和流状态均为单个样本, 或均为批量样本.
        if features.ndim not in (1, 2):
            raise ValueError(f"features must have shape [feature_dim] or [batch, feature_dim], got {features.shape}")
        if x_t.ndim != features.ndim:
            raise ValueError(f"features and x_t must either both be unbatched or both be batched, got {features.shape} and {x_t.shape}")
        if features.shape[:-1] != x_t.shape[:-1]:
            raise ValueError(f"features and x_t must have matching leading dimensions, got {features.shape} and {x_t.shape}")

        # 每个样本对应一个流时间; 批量输入需要显式提供 [B] 的时间数组.
        t = jnp.asarray(t)
        if features.ndim == 2:
            batch_size = features.shape[0]
            if t.shape != (batch_size,):
                raise ValueError(f"For batched inputs, t must have shape [batch], got {t.shape} for batch_size={batch_size}")
        elif t.ndim != 0:
            raise ValueError(f"For an unbatched input, t must be a scalar, got {t.shape}")

        # 使用已有的时间编码模块
        time_features = self.time_embedding(t)

        # 将观测与时间沿特征维拼接
        condition = jnp.concatenate([features, time_features], axis=-1)

        # 主干以 x_t 为主输入, condition 为条件输入, 输出对应的速度向量.
        velocity = self.network(x_t, condition, train=train)
        
        # 检查预测速度的形状是否和x_t相同
        if velocity.shape != x_t.shape:
            raise ValueError(f"network must output velocity with the same shape as x_t, got {velocity.shape} and {x_t.shape}")

        return velocity
