"""JAX 实现的 Diffusion Policy 余弦加噪与确定性 DDIM 去噪。"""
from __future__ import annotations

import math
from numbers import Integral

import jax
import jax.numpy as jnp

##########################################################
#             Diffusion Policy 的扩散数学工具              #
##########################################################
# 用于扩散过程加速


"""
推理流程

随机动作噪声 [B,16,7]
→ DiffusionScheduler.get_inference_timesteps()
→ 得到 [84,72,60,48,36,24,12,0]
→ 循环 8 次：
    → DiffusionUNet 预测当前噪声
    → DiffusionScheduler.ddim_step()
    → 得到噪声更少的动作序列
→ 最终动作序列 [B,16,7]
"""

# 根据 diffusers squaredcos_cap_v2 公式构造每个训练步的 beta。
def squared_cosine_betas(num_train_timesteps: int, max_beta: float = 0.999) -> jax.Array:
    if isinstance(num_train_timesteps, bool) or not isinstance(num_train_timesteps, Integral) or num_train_timesteps <= 0:
        raise ValueError("num_train_timesteps must be a positive integer.")
    if not 0.0 < max_beta <= 1.0:
        raise ValueError("max_beta must be in (0, 1].")

    def alpha_bar(time):
        return math.cos((time + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = [min(1 - alpha_bar((step + 1) / num_train_timesteps) / alpha_bar(step / num_train_timesteps), max_beta)
             for step in range(num_train_timesteps)]
    return jnp.asarray(betas, dtype=jnp.float32)


class DiffusionScheduler:
    """保存固定噪声表，并提供训练加噪和确定性 DDIM 单步更新。"""

    # 构造固定的余弦噪声表；Scheduler 不包含可训练参数和运行时可变状态。
    def __init__(self, 
                 num_train_timesteps: int = 100, 
                 beta_start:          float = 0.0001, 
                 beta_end:            float = 0.02,
                 beta_schedule:       str = "squaredcos_cap_v2",
                 clip_sample:         bool = True, 
                 set_alpha_to_one:    bool = True, 
                 steps_offset:        int = 0,
                 prediction_type:     str = "epsilon"
                 ) -> None:
        
        if beta_schedule != "squaredcos_cap_v2":
            raise ValueError("Only beta_schedule='squaredcos_cap_v2' is supported.")
        if prediction_type != "epsilon":
            raise ValueError("Only prediction_type='epsilon' is supported.")
        if isinstance(steps_offset, bool) or not isinstance(steps_offset, Integral) or steps_offset < 0:
            raise ValueError("steps_offset must be a nonnegative integer.")
        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError("beta_start and beta_end must satisfy 0 < beta_start < beta_end < 1.")

        self.num_train_timesteps = int(num_train_timesteps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.beta_schedule = beta_schedule
        self.clip_sample = bool(clip_sample)
        self.set_alpha_to_one = bool(set_alpha_to_one)
        self.steps_offset = int(steps_offset)
        self.prediction_type = prediction_type

        # 余弦调度直接由 alpha_bar 生成 beta；beta_start、beta_end 仅保留原配置接口。
        self.betas = squared_cosine_betas(self.num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = jnp.cumprod(self.alphas)
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)
        self.final_alpha_cumprod = jnp.array(1.0, dtype=jnp.float32) if self.set_alpha_to_one else self.alphas_cumprod[0]

    # 为 batch 中每条干净动作序列直接构造指定扩散步的带噪动作。
    def add_noise(self, 
                  clean_actions: jax.Array, 
                  noise: jax.Array, 
                  timesteps: jax.Array
                  ) -> jax.Array:
        clean_actions, noise, timesteps = jnp.asarray(clean_actions), jnp.asarray(noise), jnp.asarray(timesteps)
        
        if clean_actions.shape != noise.shape or clean_actions.ndim < 2:
            raise ValueError("clean_actions and noise must have the same shape [B, ...].")
        if timesteps.ndim != 1 or timesteps.shape[0] != clean_actions.shape[0]:
            raise ValueError("timesteps must have shape [B].")
        if not jnp.issubdtype(clean_actions.dtype, jnp.floating) or not jnp.issubdtype(noise.dtype, jnp.floating):
            raise TypeError("clean_actions and noise must use floating-point dtypes.")
        if not jnp.issubdtype(timesteps.dtype, jnp.integer):
            raise TypeError("timesteps must use an integer dtype.")

        coefficient_shape = (timesteps.shape[0],) + (1,) * (clean_actions.ndim - 1)
        clean_coefficient = self.sqrt_alphas_cumprod[timesteps].reshape(coefficient_shape)
        noise_coefficient = self.sqrt_one_minus_alphas_cumprod[timesteps].reshape(coefficient_shape)
        return clean_coefficient * clean_actions + noise_coefficient * noise

    # 按 diffusers 0.11.1 的 leading 规则返回确定性的 DDIM 推理时间步。
    def get_inference_timesteps(self, num_inference_steps: int = 8) -> jax.Array:

        if isinstance(num_inference_steps, bool) or not isinstance(num_inference_steps, Integral):
            raise ValueError("num_inference_steps must be an integer.")
        if not 0 < num_inference_steps <= self.num_train_timesteps:
            raise ValueError("num_inference_steps must be in [1, num_train_timesteps].")

        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = jnp.arange(num_inference_steps, dtype=jnp.int32) * step_ratio
        timesteps = timesteps[::-1] + self.steps_offset
        if (num_inference_steps - 1) * step_ratio + self.steps_offset >= self.num_train_timesteps:
            raise ValueError("steps_offset produces a timestep outside the training range.")
        return timesteps

    # 根据预测噪声执行一次 eta=0 的确定性 DDIM 更新，返回噪声更少的动作序列。
    def ddim_step(self, 
                  predicted_noise: jax.Array, 
                  timestep: jax.Array | int, 
                  sample: jax.Array,
                  num_inference_steps: int = 8
                  ) -> jax.Array:
        
        predicted_noise, sample, timestep = jnp.asarray(predicted_noise), jnp.asarray(sample), jnp.asarray(timestep)
      
        if predicted_noise.shape != sample.shape or sample.ndim < 2:
            raise ValueError("predicted_noise and sample must have the same shape [B, ...].")
        if timestep.ndim != 0 or not jnp.issubdtype(timestep.dtype, jnp.integer):
            raise TypeError("timestep must be an integer scalar.")
        if isinstance(num_inference_steps, bool) or not isinstance(num_inference_steps, Integral):
            raise ValueError("num_inference_steps must be an integer.")
        if not 0 < num_inference_steps <= self.num_train_timesteps:
            raise ValueError("num_inference_steps must be in [1, num_train_timesteps].")

        # 当前步与前一步的累计 alpha；最后一步按照 set_alpha_to_one 使用 alpha=1。
        step_ratio = self.num_train_timesteps // num_inference_steps
        previous_timestep = timestep - step_ratio
        alpha_t = self.alphas_cumprod[timestep]
        safe_previous_timestep = jnp.maximum(previous_timestep, 0)
        alpha_previous = jnp.where(previous_timestep >= 0, self.alphas_cumprod[safe_previous_timestep], self.final_alpha_cumprod)
        beta_t = 1.0 - alpha_t

        # epsilon 预测先恢复干净动作，再按照配置裁剪到归一化动作范围。
        predicted_original = (sample - jnp.sqrt(beta_t) * predicted_noise) / jnp.sqrt(alpha_t)
        if self.clip_sample:
            predicted_original = jnp.clip(predicted_original, -1.0, 1.0)

        # eta=0 时没有附加随机噪声，DDIM 更新完全由当前样本和预测噪声决定。
        direction = jnp.sqrt(jnp.maximum(1.0 - alpha_previous, 0.0)) * predicted_noise
        return jnp.sqrt(alpha_previous) * predicted_original + direction
