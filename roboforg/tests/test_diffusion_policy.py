import jax
import jax.numpy as jnp
import numpy as np

from roboforg.agent.diffusion_policy import DiffusionPolicy


# 构造算法入口使用的原始双相机观测和动作序列。
def make_batch():
    image = np.arange(32 * 32 * 3, dtype=np.uint32).reshape(32, 32, 3).astype(np.uint8)
    observations = {"front": np.broadcast_to(image, (1, 2, 32, 32, 3)),
                    "wrist": np.broadcast_to(np.flip(image, axis=1), (1, 2, 32, 32, 3)),
                    "state": np.zeros((1, 2, 23), dtype=np.float32)}
    return observations, jnp.zeros((1, 16, 7), dtype=jnp.float32)


# 使用缩小通道验证完整参数初始化、训练损失及梯度。
def test_training_loss_and_gradient():
    policy = DiffusionPolicy(diffusion_step_embedding_dim=8, down_channels=(8, 16, 32), n_groups=4)
    observations, actions = make_batch()
    params = policy.init_parameters(jax.random.PRNGKey(0), observations)
    (loss, metrics), gradients = jax.value_and_grad(policy.compute_loss, has_aux=True)(params, observations, actions, jax.random.PRNGKey(1))

    assert loss.shape == () and np.isfinite(float(loss))
    assert set(metrics) == {"loss", "predicted_noise_mean", "target_noise_mean", "timestep_mean"}
    gradient_norm = sum(float(jnp.linalg.norm(value)) for value in jax.tree_util.tree_leaves(gradients))
    assert gradient_norm > 0


# 检查观测独立编码接口、确定性 DDIM 采样和最终动作切片。
def test_inference_reuses_condition_and_slices_actions():
    policy = DiffusionPolicy(num_inference_steps=2, diffusion_step_embedding_dim=8,
                             down_channels=(8, 16, 32), n_groups=4)
    observations, _ = make_batch()
    params = policy.init_parameters(jax.random.PRNGKey(2), observations)
    normalized = policy.normalize_observations(observations)
    condition = policy.network.apply({"params": params}, normalized, method=policy.network.encode_observation)
    assert condition.shape == (1, 2094)

    first = policy.predict_action(params, observations, jax.random.PRNGKey(3))
    second = policy.predict_action(params, observations, jax.random.PRNGKey(3))
    assert first["action_pred"].shape == (1, 16, 7)
    assert first["action"].shape == (1, 8, 7)
    np.testing.assert_allclose(first["action"], first["action_pred"][:, 1:9])
    np.testing.assert_allclose(first["action_pred"], second["action_pred"])
