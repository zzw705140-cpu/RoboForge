"""Diffusion Policy 的训练状态初始化、单步训练和验证；配置由调用方传入。"""
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax.core import freeze, unfreeze
from flax.training.train_state import TrainState

from roboforg.networks.diffusion_policy import load_pretrained_resnet10


# 保存普通参数、优化器状态、累计更新次数和 EMA 参数。
class DiffusionTrainState(TrainState):
    ema_params: object


# 初始化网络，先加载视觉预训练参数，再建立优化器与同起点的 EMA。
def create_train_state(policy, rng, observations, *, learning_rate, beta1, beta2, eps, weight_decay,
                       pretrained_path: str | Path | None = None):
    if not callable(learning_rate) and learning_rate <= 0:
        raise ValueError("learning_rate must be positive or an Optax schedule.")
    if not 0 <= beta1 < 1 or not 0 <= beta2 < 1 or eps <= 0 or weight_decay < 0:
        raise ValueError("Invalid AdamW parameters.")

    # None 表示随机初始化；正式训练由外部传入本地预训练权重路径。
    params = unfreeze(policy.init_parameters(rng, observations))
    if pretrained_path is not None:
        params["observation_encoder"] = load_pretrained_resnet10(params["observation_encoder"], pretrained_path)
    params = freeze(params)

    # 学习率可以是标量或调度函数；TrainState 内部保存 AdamW 的动量和步数。
    optimizer = optax.adamw(learning_rate=learning_rate, b1=beta1, b2=beta2, eps=eps, weight_decay=weight_decay)
    return DiffusionTrainState.create(apply_fn=policy.network.apply, params=params, tx=optimizer, ema_params=params)


# 根据更新前的累计步数计算 EMA 衰减，沿用原项目的 warmup 计数约定。
def ema_decay(step, *, update_after_step, inv_gamma, power, min_decay, max_decay):
    if update_after_step < 0 or inv_gamma <= 0 or power <= 0 or not 0 <= min_decay <= max_decay < 1:
        raise ValueError("Invalid EMA parameters.")
    age = jnp.maximum(0, step - update_after_step - 1)
    decay = 1.0 - (1.0 + age / inv_gamma) ** (-power)
    return jnp.where(age > 0, jnp.clip(decay, min_decay, max_decay), 0.0)


# 计算一个 batch 的梯度，先更新普通参数，再用新参数更新 EMA。
@partial(jax.jit, static_argnames=("policy", "update_after_step", "inv_gamma", "power", "min_decay", "max_decay"))
def train_step(state, observations, actions, rng, *, policy, update_after_step, inv_gamma, power, min_decay, max_decay):
    (_, metrics), gradients = jax.value_and_grad(policy.compute_loss, has_aux=True)(state.params, observations, actions, rng)
    updated = state.apply_gradients(grads=gradients)

    # 原项目先用旧 optimization_step 计算衰减，完成 EMA 后才递增计数。
    decay = ema_decay(state.step, update_after_step=update_after_step, inv_gamma=inv_gamma,
                      power=power, min_decay=min_decay, max_decay=max_decay)
    averaged = jax.tree_util.tree_map(lambda old, new: decay * old + (1.0 - decay) * new,
                                     state.ema_params, updated.params)
    updated = updated.replace(ema_params=averaged)
    metrics = {**metrics, "gradient_norm": optax.global_norm(gradients), "ema_decay": decay}
    return updated, metrics


# 使用 EMA 参数计算验证损失；调用方传入固定随机键可复现加噪结果。
@partial(jax.jit, static_argnames=("policy",))
def evaluate_step(state, observations, actions, rng, *, policy):
    _, metrics = policy.compute_loss(state.ema_params, observations, actions, rng)
    return metrics
