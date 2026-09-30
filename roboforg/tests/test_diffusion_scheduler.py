import jax
import jax.numpy as jnp
import numpy as np
import pytest

from roboforg.agent.diffusion_policy import DiffusionScheduler, squared_cosine_betas


# 检查余弦 beta 表的长度、范围和固定首尾数值。
def test_squared_cosine_schedule():
    betas = squared_cosine_betas(100)
    assert betas.shape == (100,)
    betas = np.asarray(betas)
    np.testing.assert_allclose(betas[[0, -1]], np.array([0.00063128, 0.999], np.float32), rtol=2e-5)
    assert np.all(np.diff(betas) > 0)


# 检查训练加噪对每个样本使用各自的整数扩散步。
def test_add_noise_matches_formula_and_jit():
    scheduler = DiffusionScheduler()
    clean = jnp.arange(2 * 4 * 3, dtype=jnp.float32).reshape(2, 4, 3) / 10
    noise = jnp.ones_like(clean)
    timesteps = jnp.array([0, 99], dtype=jnp.int32)
    output = jax.jit(scheduler.add_noise)(clean, noise, timesteps)

    clean_scale = np.asarray(scheduler.sqrt_alphas_cumprod)[[0, 99]].reshape(2, 1, 1)
    noise_scale = np.asarray(scheduler.sqrt_one_minus_alphas_cumprod)[[0, 99]].reshape(2, 1, 1)
    reference = clean_scale * np.asarray(clean) + noise_scale * np.asarray(noise)
    np.testing.assert_allclose(output, reference, rtol=1e-6, atol=1e-6)


# 检查 diffusers 0.11.1 在100个训练步和8个推理步下的固定时间表。
def test_inference_timesteps():
    scheduler = DiffusionScheduler(num_train_timesteps=100, steps_offset=0)
    np.testing.assert_array_equal(scheduler.get_inference_timesteps(8), [84, 72, 60, 48, 36, 24, 12, 0])


# 检查中间去噪步和最后一步都符合确定性 DDIM 公式。
@pytest.mark.parametrize("timestep", [84, 0])
def test_ddim_step_matches_formula(timestep):
    scheduler = DiffusionScheduler()
    sample = jnp.linspace(-0.5, 0.5, 2 * 4 * 3, dtype=jnp.float32).reshape(2, 4, 3)
    predicted_noise = jnp.full_like(sample, 0.1)
    output = jax.jit(lambda x, e: scheduler.ddim_step(e, jnp.int32(timestep), x, 8))(sample, predicted_noise)

    previous_timestep = timestep - 12
    alpha_t = float(scheduler.alphas_cumprod[timestep])
    alpha_previous = float(scheduler.alphas_cumprod[previous_timestep]) if previous_timestep >= 0 else 1.0
    predicted_original = (np.asarray(sample) - np.sqrt(1 - alpha_t) * np.asarray(predicted_noise)) / np.sqrt(alpha_t)
    predicted_original = np.clip(predicted_original, -1.0, 1.0)
    reference = np.sqrt(alpha_previous) * predicted_original + np.sqrt(1 - alpha_previous) * np.asarray(predicted_noise)
    np.testing.assert_allclose(output, reference, rtol=1e-5, atol=1e-5)
