"""ACT 训练期 CVAE 后验编码器：由真实动作学习潜变量 z。"""

from __future__ import annotations

from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from roboforg.networks.act.tokenizer import (
    ActionTokenizer,
    CLSToken,
    CVAEInputAssembler,
    StateTokenizer,
)
from roboforg.networks.act.transformer import TransformerEncoder


# 后验编码器的一次前向结果；保留 mu/logvar 以供 Agent 计算 KL loss。
class LatentEncoderOutput(NamedTuple):
    z: jax.Array
    mu: jax.Array
    logvar: jax.Array


# 通过重参数技巧从 N(mu, diag(exp(logvar))) 采样，同时保持对 mu/logvar 可导。
def sample_latent(mu: jax.Array, logvar: jax.Array, rng: jax.Array) -> jax.Array:
    """Sample z = mu + exp(0.5 * logvar) * epsilon, epsilon ~ N(0, I)."""
    if mu.ndim != 2 or mu.shape[0] == 0 or mu.shape[1] == 0:
        raise ValueError("mu must have non-empty shape [B,Z].")
    if logvar.shape != mu.shape:
        raise ValueError("logvar must have exactly the same shape as mu.")

    # 随机噪声与分布参数逐元素结合；梯度不会穿过 epsilon，但会流向 mu/logvar。
    epsilon = jax.random.normal(rng, mu.shape, dtype=mu.dtype)
    return mu + jnp.exp(0.5 * logvar) * epsilon


# 计算 q(z|state, action) 相对于标准正态先验 N(0, I) 的 KL 散度。
def kl_loss(mu: jax.Array, logvar: jax.Array) -> jax.Array:
    """Return mean_B(sum_Z KL(q(z|x) || N(0,I))), matching original ACT reduction."""
    if mu.ndim != 2 or mu.shape[0] == 0 or mu.shape[1] == 0:
        raise ValueError("mu must have non-empty shape [B,Z].")
    if logvar.shape != mu.shape:
        raise ValueError("logvar must have exactly the same shape as mu.")

    # 先对每个样本的所有潜变量维求和，再对 batch 求平均，不能把 Z 维直接平均掉。
    per_sample = -0.5 * jnp.sum(1.0 + logvar - jnp.square(mu) - jnp.exp(logvar), axis=-1)
    return jnp.mean(per_sample)


# CVAE 后验网络：训练时用 state 和真实 action chunk 推断 z 的高斯分布参数。
class LatentEncoder(nn.Module):
    """Encode normalized state/action demonstrations into a sampled latent vector."""

    token_dim: int = 256
    latent_dim: int = 32
    num_heads: int = 8
    feedforward_dim: int = 2048
    num_layers: int = 4
    dropout_rate: float = 0.1

    # 输入 state [B,S]、actions [B,k,A]，返回 z、mu、logvar，三者均为 [B,Z]。
    @nn.compact
    def __call__(
        self,
        state: jax.Array,
        actions: jax.Array,
        *,
        rng: jax.Array,
        train: bool = False,
    ) -> LatentEncoderOutput:
        if self.token_dim <= 0 or self.latent_dim <= 0:
            raise ValueError("token_dim and latent_dim must be positive.")
        if state.ndim != 2 or state.shape[0] == 0 or state.shape[1] == 0:
            raise ValueError("state must have non-empty shape [B,S].")
        if actions.ndim != 3 or min(actions.shape) <= 0 or actions.shape[0] != state.shape[0]:
            raise ValueError("actions must have non-empty shape [B,k,A] and match the state batch size.")

        # 当前状态和真实未来动作分别投影为 token；两种 token 不共享投影权重。
        state_token = StateTokenizer(self.token_dim, name="state_tokenizer")(state)
        action_tokens = ActionTokenizer(self.token_dim, name="action_tokenizer")(actions)
        cls_token = CLSToken(self.token_dim, name="cls_token")(state.shape[0])

        # 固定排列 [CLS, state, action_0, ..., action_(k-1)]，并生成对应的一维位置编码。
        tokens, position = CVAEInputAssembler(self.token_dim, name="input_assembler")(
            cls_token, state_token, action_tokens
        )

        # Transformer 让 CLS 汇总状态与整段真实动作；其余 token 不直接作为后验输出。
        encoded = TransformerEncoder(
            token_dim=self.token_dim,
            num_heads=self.num_heads,
            feedforward_dim=self.feedforward_dim,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
            name="transformer_encoder",
        )(tokens, position, train=train)
        cls_feature = encoded[:, 0, :]

        # 两个独立线性头参数化对角高斯后验的均值与对数方差。
        mu = nn.Dense(
            self.latent_dim, kernel_init=nn.initializers.xavier_uniform(), name="mu_head"
        )(cls_feature)
        logvar = nn.Dense(
            self.latent_dim, kernel_init=nn.initializers.xavier_uniform(), name="logvar_head"
        )(cls_feature)
        z = sample_latent(mu, logvar, rng)
        return LatentEncoderOutput(z=z, mu=mu, logvar=logvar)
