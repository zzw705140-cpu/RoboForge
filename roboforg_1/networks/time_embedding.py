"""用于 Flow Matching 的正余弦时间编码."""

import math

import flax.linen as nn
import jax.numpy as jnp


class SinusoidalTimeEmbedding(nn.Module):
    """基于正弦曲线的时间编码"""
    
    embed_dim: int = 32                         # 输出时间特征的维度，必须为不小于 4 的偶数
    max_freq_log: float = math.log(10000)        # 控制频率范围，与 GCP 的默认配置一致

    @nn.compact
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        将标量流时间转换为时间特征, 供速度网络作为条件使用
        编码时间; 只整理时间特征, 不负责预测速度或执行积分.
        输入支持单个时间 [] 或批量时间 [B], 通常 t 位于 [0, 1].
        正余弦变换没有可学习参数, 后面两层 Dense 的参数随整个网络一起训练.

        流时间 t: [B]
            ↓ 与不同频率相乘
        angles: [B,16]
            ├→ Sin → [B,16] ─┐
            └→ Cos → [B,16] ─┘
                    ↓ Concatenate
                正余弦特征: [B,32]
                    ↓ Dense(64)
                        [B,64]
                    ↓ GELU
                        [B,64]
                    ↓ Dense(32)
                时间特征 emb: [B,32]        

        """

        t = jnp.asarray(t)
        if t.ndim not in (0, 1):
            raise ValueError(f"t must be a scalar or have shape [batch], got {t.shape}")
        if self.embed_dim < 4 or self.embed_dim % 2 != 0:
            raise ValueError("embed_dim must be an even integer greater than or equal to 4")

        # 一半维度用于 Sin，一半用于 Cos；频率默认从 1 按几何级数递减到 1/10000。
        half_dim = self.embed_dim // 2
        freqs = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32)* -(self.max_freq_log / (half_dim - 1))
        )

        # [] → [E/2]，或 [B] → [B, E/2]；每个时间与所有频率相乘。
        angles = t[..., None] * freqs

        # 沿最后的特征维拼接，得到 [E] 或 [B, E]。
        emb = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)

        # 小型 MLP：E → 2E → E，两层 Dense 均包含可学习的权重和偏置。
        emb = nn.Dense(self.embed_dim * 2)(emb)
        emb = nn.gelu(emb)
        emb = nn.Dense(self.embed_dim)(emb)

        return emb
