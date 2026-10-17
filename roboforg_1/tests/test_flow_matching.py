"""验证 Flow Matching 的模型组装, 参数更新与观测布局契约."""

from functools import partial
import pickle

import flax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.traverse_util import flatten_dict

from roboforg_1.agent.policy import flow_matching as flow_module
from roboforg_1.agent.policy.flow_matching import FlowMatchingAgent
from roboforg_1.networks.vision.resnet_v1 import resnetv1_configs


@pytest.fixture(scope="module")
def local_backbone(tmp_path_factory):
    """用真实的小型 ResNet 结构创建本地测试权重, 不访问网络."""
    factory = partial(resnetv1_configs["resnetv1-10"], image_size=(32, 32), num_filters=8)
    params = factory().init(jax.random.PRNGKey(9), jnp.ones((1, 32, 32, 3)), train=False)["params"]
    checkpoint = tmp_path_factory.mktemp("flow_weights") / "resnet10_params.pkl"
    with checkpoint.open("wb") as file:
        pickle.dump(jax.device_get(params), file)
    return factory, checkpoint, params


@pytest.fixture
def create_agent(monkeypatch, local_backbone):
    """仅缩小测试 Backbone, 其余编码器, 加载器和更新逻辑均使用真实实现."""
    factory, checkpoint, _ = local_backbone
    monkeypatch.setattr(flow_module, "resnetv1_configs", {"resnetv1-10": factory})

    def create(**kwargs):
        """构造带两帧观测和两步动作块的测试 Agent."""
        observations = {
            "images": {"front": jax.random.randint(jax.random.PRNGKey(1), (2, 32, 32, 3), 0, 255, dtype=jnp.uint8)},
            "state": jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            "task_condition": {"task": jnp.array([[1., 0., 0.], [1., 0., 0.]])},
        }
        options = dict(seed=0, sample_observations=observations, sample_actions=jnp.zeros(14), observation_horizon=2, action_horizon=2, hidden_dim=16, num_blocks=1, time_embed_dim=4, dropout_rate=0.1, resnet_checkpoint_path=checkpoint)
        options.update(kwargs)
        return FlowMatchingAgent.create(**options), options["sample_observations"]

    return create


def _batch(observations):
    """扩展单样本为两条批量数据, 提供浮点示范动作块."""
    return {"observations": jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), observations), "actions": jnp.linspace(-0.8, 0.8, 28).reshape(2, 14)}


def _flat(params):
    """将 Flax 参数整理成路径到数组的映射, 用于检查参数分组."""
    return flatten_dict(flax.serialization.to_state_dict(params))


def _different(before, after):
    """判断两个参数树是否至少有一个数组发生改变."""
    return any(not np.array_equal(a, b) for a, b in zip(jax.tree_util.tree_leaves(before), jax.tree_util.tree_leaves(after)))


@pytest.mark.parametrize("train_backbone", [False, True])
def test_joint_update_respects_backbone_freezing(create_agent, local_backbone, train_backbone):
    """真实联合训练时头部和速度网络更新, Backbone 仅在启用微调时更新."""
    agent, observations = create_agent(train_backbone=train_backbone)
    initial = _flat(agent.encoder.params)
    checkpoint_params = _flat(local_backbone[2])
    backbone = {path: value for path, value in initial.items() if "pretrained_encoder" in path}
    assert backbone
    # 确认加载器实际替换了共享 Backbone, 而非保留创建时的随机参数.
    roots = {path[:path.index("pretrained_encoder") + 1] for path in backbone}
    assert len(roots) == 1
    for path, value in backbone.items():
        suffix = path[path.index("pretrained_encoder") + 1:]
        np.testing.assert_array_equal(value, checkpoint_params[suffix])

    updated, info = agent.update(_batch(observations))
    assert int(updated.encoder.step) == 1 and int(updated.vector_field.step) == 1
    assert all(np.isfinite(value).all() for value in info.values())
    assert _different(agent.vector_field.params, updated.vector_field.params)
    after = _flat(updated.encoder.params)
    changed_backbone = any(not np.array_equal(initial[path], after[path]) for path in backbone)
    changed_head = any(not np.array_equal(initial[path], after[path]) for path in initial if path not in backbone)
    assert changed_backbone == train_backbone
    assert changed_head
    assert not np.array_equal(agent.rng, updated.rng)
    assert int(agent.encoder.step) == 0

    # 通过同一个梯度输入, 确认微调参数组实际采用较小的学习率.
    if train_backbone:
        ones = jax.tree_util.tree_map(jnp.ones_like, agent.encoder.params)
        updates, _ = agent.encoder.tx.update(ones, agent.encoder.opt_state, agent.encoder.params)
        flat_updates = _flat(updates)
        backbone_update = next(float(np.abs(value).mean()) for path, value in flat_updates.items() if "pretrained_encoder" in path)
        head_update = next(float(np.abs(value).mean()) for path, value in flat_updates.items() if "pretrained_encoder" not in path)
        assert backbone_update / head_update == pytest.approx(agent.config["backbone_lr"] / agent.config["encoder_lr"], rel=1e-4)


@pytest.mark.parametrize("pooling", ["avg", "spatial_learned_embeddings", "task_conditioned_attention"])
def test_multicamera_raw_and_cached_features_agree(create_agent, pooling):
    """多相机共享 Backbone, 外围参数独立, 原始图像和缓存编码结果一致."""
    _, observations = create_agent(pooling_method=pooling)
    observations["images"]["wrist"] = jnp.flip(observations["images"]["front"], axis=2)
    agent, _ = create_agent(sample_observations=observations, pooling_method=pooling)
    batch = _batch(observations)
    raw = agent.encoder(batch["observations"], train=False)
    cached = agent.encoder.apply_fn({"params": agent.encoder.params}, batch["observations"]["images"], method="encode_backbone")
    cached_observations = {key: value for key, value in batch["observations"].items() if key != "images"}
    cached_observations["visual_features"] = cached
    np.testing.assert_allclose(raw, agent.encoder(cached_observations, train=False), rtol=1e-5, atol=1e-5)
    assert raw.shape == (2, 2 * 64 + 2 * 2 * 256 + 64)
    np.testing.assert_allclose(raw[0], agent.encoder(observations, train=False), rtol=1e-5, atol=1e-5)
    if pooling == "task_conditioned_attention":
        assert isinstance(cached["front"], dict)
    else:
        assert not isinstance(cached["front"], dict)
    flat_params = _flat(agent.encoder.params)
    assert len({path[:path.index("pretrained_encoder") + 1] for path in flat_params if "pretrained_encoder" in path}) == 1
    # 两个相机各有自己的瓶颈 Dense, 对共享 Backbone 的引用不产生第二套参数.
    assert any("front" in str(path) and "Dense_0" in path for path in flat_params)
    assert any("wrist" in str(path) and "Dense_0" in path for path in flat_params)
    updated, info = agent.update({"observations": cached_observations, "actions": batch["actions"]})
    assert int(updated.encoder.step) == 1
    assert np.isfinite(info["flow_matching_loss"])


def test_previous_chunk_condition_training_and_reset(create_agent):
    """上一动作块使用数据批中的输入训练, 运行状态重置不改变参数."""
    agent, observations = create_agent(condition_on_previous_chunk=True, pooling_method="avg")
    batch = _batch(observations)
    batch["previous_chunk"] = jnp.linspace(-0.5, 0.5, 28).reshape(2, 14)
    updated, _ = agent.update(batch)
    assert int(updated.previous_chunk_encoder.step) == 1
    assert _different(agent.previous_chunk_encoder.params, updated.previous_chunk_encoder.params)
    np.testing.assert_array_equal(agent.previous_chunk, updated.previous_chunk)
    batched_agent = updated.reset_previous_chunk(batch_size=2)
    assert batched_agent.previous_chunk.shape == (2, 14)
    conditioned = batched_agent._condition_features(batch["observations"])
    base = batched_agent.encoder(batch["observations"], train=False)
    assert conditioned.shape == (2, base.shape[-1] + 64)
    np.testing.assert_allclose(conditioned[:, :base.shape[-1]], base)
    assert batched_agent.encoder is updated.encoder
    with pytest.raises(ValueError, match="sampling interface"):
        agent._require_stateless_sampling()
    with pytest.raises(KeyError, match="previous_chunk"):
        agent.update(_batch(observations))


def test_non_temporal_images_and_optional_observation_branches(create_agent):
    """关闭历史窗口和状态任务分支时, 保持单样本及批量接口一致."""
    observations = {"images": {"front": jnp.ones((32, 32, 3), dtype=jnp.uint8)}}
    agent, _ = create_agent(sample_observations=observations, pooling_method="avg", use_temporal_history=False, observation_horizon=1)
    assert agent.encoder(observations, train=False).shape == (256,)
    updated, info = agent.update(_batch(observations))
    assert int(updated.vector_field.step) == 1
    assert np.isfinite(info["flow_matching_loss"])
    assert agent.reset_previous_chunk() is agent


def test_target_path_contract():
    """通过路径关系检查中间状态和监督速度的一致性, 而非完整执行积分."""
    actions = jnp.arange(28, dtype=jnp.float32).reshape(2, 14) / 28
    x_t, velocity, t = FlowMatchingAgent._build_flow_matching_targets(actions, jax.random.PRNGKey(0))
    assert t.shape == (2,) and np.all((t >= 0) & (t < 1))
    np.testing.assert_allclose(x_t + (1 - t[:, None]) * velocity, actions, atol=1e-6)
    with pytest.raises(TypeError, match="floating"):
        FlowMatchingAgent._build_flow_matching_targets(jnp.ones((2, 14), dtype=jnp.int32), jax.random.PRNGKey(0))


@pytest.mark.parametrize("options", [{"network_type": "film"}, {"time_embed_dim": 3}, {"action_horizon": 0}, {"pooling_method": "task_conditioned_attention", "attention_stages": None}])
def test_invalid_configuration_fails_before_loading(create_agent, options):
    """不支持或互相矛盾的配置应在创建网络和读取权重前明确报错."""
    with pytest.raises(ValueError):
        create_agent(**options)


def test_finetuning_rejects_cached_features(create_agent):
    """微调训练必须经过 Backbone, 防止缓存输入悄悄跳过视觉参数更新."""
    agent, observations = create_agent(train_backbone=True, pooling_method="avg")
    batch = _batch(observations)
    with pytest.raises(ValueError, match="raw images"):
        agent.update({"observations": {"visual_features": {}}, "actions": batch["actions"]})


def test_replay_buffer_action_layout_and_padding_mask(create_agent):
    """验证原生动作块布局等价于展平布局, 填充标记按有效步计算损失."""
    agent, observations = create_agent(pooling_method="avg", sample_actions=jnp.zeros((2, 7)))
    flat_batch = _batch(observations)
    chunk_batch = {**flat_batch, "actions": flat_batch["actions"].reshape(2, 2, 7)}
    updated_flat, flat_info = agent.update(flat_batch)
    updated_chunk, chunk_info = agent.update(chunk_batch)
    assert not _different(updated_flat.vector_field.params, updated_chunk.vector_field.params)
    np.testing.assert_allclose(flat_info["flow_matching_loss"], chunk_info["flow_matching_loss"])

    # 使用相同随机数和原参数重算逐步误差, 确认第二条样本的填充步不进入损失.
    valid = jnp.array([[True, True], [True, False]])
    _, loss_rng = jax.random.split(agent.rng)
    target_rng, encoder_rng, vector_rng = jax.random.split(loss_rng, 3)
    x_t, target, t = agent._build_flow_matching_targets(flat_batch["actions"], target_rng)
    features = agent.encoder(flat_batch["observations"], train=True, rngs={"dropout": encoder_rng})
    predicted = agent.vector_field(features, x_t, t, train=True, rngs={"dropout": vector_rng})
    errors = np.asarray((predicted - target) ** 2).reshape(2, 2, 7).mean(axis=-1)
    expected = (errors[0].mean() + errors[1, 0]) / 2
    _, masked_info = agent.update({**chunk_batch, "action_valid": valid})
    np.testing.assert_allclose(masked_info["flow_matching_loss"], expected, rtol=1e-5)
    with pytest.raises(ValueError, match="action_valid"):
        agent.update({**chunk_batch, "action_valid": jnp.ones((2, 3))})


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("action_dim", [6, 7])
def test_ode_euler_trace_and_final_projection(batched, action_dim):
    """用可解析的速度场检查左端点积分, 完整轨迹及仅最终动作投影."""
    def velocity(features, x_t, t, train=False):
        """返回 x_t + t, 其 Euler 递推有独立的闭式解."""
        assert not train
        return x_t + (t[:, None] if x_t.ndim == 2 else t)

    steps, horizon = 4, 2
    noise = jnp.linspace(-2.0, 3.0, horizon * action_dim)
    features = jnp.ones(3)
    if batched:
        noise, features = jnp.stack([noise, noise * 0.5]), jnp.ones((2, 3))
    actions, trace = FlowMatchingAgent.get_action_and_latent_trace_with_feat_and_noise_by_ode(velocity, features, noise, num_ode_steps=steps, action_horizon=horizon, action_dim=action_dim)
    k = np.arange(steps + 1, dtype=np.float32)
    expected = (1 + 1 / steps) ** k[:, None] * (np.asarray(noise)[..., None, :] + 1) - 1 - k[:, None] / steps
    np.testing.assert_allclose(trace, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(trace[..., 0, :], noise)

    # 独立按 NumPy 公式约束最终动作, 原始轨迹不应被裁剪.
    projected = expected[..., -1, :].reshape((*noise.shape[:-1], horizon, action_dim)).copy()
    for start in (0, 3):
        vector = projected[..., start:start + 3]
        projected[..., start:start + 3] = vector / np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), 1)
    if action_dim == 7:
        projected[..., 6] = np.clip(projected[..., 6], -1, 1)
    np.testing.assert_allclose(actions, projected.reshape(noise.shape), rtol=1e-6, atol=1e-6)
    assert np.any(np.abs(np.asarray(trace)) > 1)
    direct = FlowMatchingAgent.get_action_with_feat_and_noise_by_ode(velocity, features, noise, num_ode_steps=steps, action_horizon=horizon, action_dim=action_dim)
    np.testing.assert_array_equal(actions, direct)
    with pytest.raises(ValueError, match="positive"):
        FlowMatchingAgent.get_action_with_feat_and_noise_by_ode(velocity, features, noise, num_ode_steps=0, action_horizon=horizon, action_dim=action_dim)
    with pytest.raises(TypeError, match="floating"):
        FlowMatchingAgent.get_action_with_feat_and_noise_by_ode(velocity, features, noise.astype(jnp.int32), num_ode_steps=steps, action_horizon=horizon, action_dim=action_dim)


def test_ode_observation_interfaces_and_stateless_guard(create_agent):
    """用真实编码器验证给定噪声, RNG, 批量与缓存接口, 以及无状态限制."""
    agent, observations = create_agent(pooling_method="avg", num_ode_steps=3)
    key = jax.random.PRNGKey(20)
    features = agent.encoder(observations, train=False)
    noise = jax.random.normal(key, (14,), dtype=features.dtype)
    expected = agent.get_action_with_feat_and_noise_by_ode(agent.vector_field, features, noise, num_ode_steps=3, action_horizon=2, action_dim=7)
    actions, trace = agent.get_action_and_latent_trace_with_obs_and_noise_by_ode(observations, noise)
    assert trace.shape == (4, 14)
    np.testing.assert_allclose(actions, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(agent.get_action_with_obs_and_noise_by_ode(observations, noise), expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(agent.get_action_with_obs_by_ode(observations, key), expected, rtol=1e-5, atol=1e-6)
    cached = agent.encoder.apply_fn({"params": agent.encoder.params}, observations["images"], method="encode_backbone")
    cached_observations = {**observations, "visual_features": cached}
    del cached_observations["images"]
    np.testing.assert_allclose(agent.get_action_with_obs_by_ode(cached_observations, key), expected, rtol=1e-5, atol=1e-6)
    batch = _batch(observations)["observations"]
    batch_actions, batch_trace = agent.get_action_and_latent_trace_with_obs_and_noise_by_ode(batch, jnp.stack([noise, noise]))
    assert batch_actions.shape == (2, 14) and batch_trace.shape == (2, 4, 14)
    np.testing.assert_allclose(batch_actions, jnp.stack([expected, expected]), rtol=1e-5, atol=1e-6)
    assert agent.get_action_with_obs_by_ode(batch, key).shape == (2, 14)
    assert int(agent.encoder.step) == 0 and int(agent.vector_field.step) == 0
    guarded = agent.replace(config=agent.config.copy({"condition_on_previous_chunk": True}))
    with pytest.raises(ValueError, match="Previous-chunk"):
        guarded.get_action_with_obs_by_ode(observations, key)


@pytest.mark.parametrize("batched", [False, True])
def test_sde_sampling_and_probability_interfaces(create_agent, batched):
    """验证 SDE 接口一致性, 高斯概率及随机性, 单样本和批量均可运行."""
    from scipy.stats import norm

    agent, observations = create_agent(pooling_method="avg", num_sde_steps=3, sde_noise_scale=0.2)
    if batched:
        observations = _batch(observations)["observations"]
    features = agent.encoder(observations, train=False)
    key = jax.random.PRNGKey(30)
    actions, trace, step_probs, chain_prob = agent.get_action_and_latent_trace_with_obs_by_sde(observations, key)
    leading = features.shape[:-1]
    assert actions.shape == (*leading, 14)
    assert trace.shape == (*leading, 4, 14)
    assert step_probs.shape == (*leading, 3) and chain_prob.shape == leading
    assert all(np.isfinite(value).all() for value in (actions, trace, step_probs, chain_prob))
    np.testing.assert_allclose(chain_prob, step_probs.sum(axis=-1), rtol=1e-6)
    np.testing.assert_allclose(agent.get_action_with_obs_by_sde(observations, key), actions, rtol=1e-5, atol=1e-6)
    recomputed = agent.get_latent_trace_log_prob_with_obs_by_sde(observations, trace)
    np.testing.assert_allclose(recomputed[0], step_probs, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(recomputed[1], chain_prob, rtol=1e-5, atol=1e-5)

    # 指定初始噪声时复现相同轨迹, 初始 RNG 与各步转移 RNG 分开管理.
    transition_key, initial_key = jax.random.split(key)
    initial_noise = jax.random.normal(initial_key, shape=actions.shape, dtype=features.dtype)
    replayed = agent.get_action_and_latent_trace_with_obs_and_noise_by_sde(observations, initial_noise, transition_key)
    for actual, expected in zip(replayed, (actions, trace, step_probs, chain_prob)):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    changed = agent.get_action_and_latent_trace_with_feat_and_noise_by_sde(features, initial_noise, jax.random.PRNGKey(31))
    assert not np.allclose(changed[1], trace)

    # 用 SciPy 的高斯密度核对每步概率, 概率对应未投影轨迹.
    dt = 1 / 3
    for step in range(3):
        t = jnp.full(leading, step / 3, dtype=features.dtype)
        x_t, x_next = trace[..., step, :], trace[..., step + 1, :]
        velocity = np.asarray(agent.vector_field(features, x_t, t, train=False))
        expanded_t = np.asarray(t)[..., None]
        expected_mean = np.asarray(x_t) + (velocity + 0.5 * 0.2 ** 2 * (expanded_t * velocity - np.asarray(x_t))) * dt
        expected_std = 0.2 * np.sqrt(1 - expanded_t) * np.sqrt(dt)
        expected_log_prob = norm.logpdf(np.asarray(x_next), loc=expected_mean, scale=expected_std).sum(axis=-1)
        np.testing.assert_allclose(step_probs[..., step], expected_log_prob, rtol=1e-5, atol=2e-4)
    chunks = np.asarray(actions).reshape((*leading, 2, 7))
    assert np.all(np.linalg.norm(chunks[..., :3], axis=-1) <= 1 + 1e-6)
    assert np.all(np.linalg.norm(chunks[..., 3:6], axis=-1) <= 1 + 1e-6)
    assert np.all(np.abs(chunks[..., 6]) <= 1)

    # 重新评估轨迹概率可对当前速度网络求导, 采样本身不修改模型状态.
    def likelihood(params):
        """固定采样轨迹, 对当前速度网络的条件转移概率求和."""
        evaluating = agent.replace(vector_field=agent.vector_field.replace(params=params))
        return evaluating.get_latent_trace_log_prob_with_feat_by_sde(features, jax.lax.stop_gradient(trace))[1].sum()

    gradients = jax.grad(likelihood)(agent.vector_field.params)
    assert all(np.isfinite(value).all() for value in jax.tree_util.tree_leaves(gradients))
    assert any(np.any(np.asarray(value) != 0) for value in jax.tree_util.tree_leaves(gradients))
    assert int(agent.encoder.step) == 0 and int(agent.vector_field.step) == 0
    guarded = agent.replace(config=agent.config.copy({"condition_on_previous_chunk": True}))
    with pytest.raises(ValueError, match="Previous-chunk"):
        guarded.get_action_with_obs_by_sde(observations, key)
    with pytest.raises(ValueError, match="length"):
        agent.get_latent_trace_log_prob_with_feat_by_sde(features, trace[..., :-1, :])


def test_sde_zero_noise_actions_and_probability_guard(create_agent):
    """零扩散时动作生成退化为同一步数 ODE, 概率接口拒绝退化高斯."""
    agent, observations = create_agent(pooling_method="avg", num_ode_steps=3, num_sde_steps=3, sde_noise_scale=0.0)
    key = jax.random.PRNGKey(40)
    features = agent.encoder(observations, train=False)
    _, initial_key = jax.random.split(key)
    initial_noise = jax.random.normal(initial_key, (14,), dtype=features.dtype)
    expected = agent.get_action_with_obs_and_noise_by_ode(observations, initial_noise)
    np.testing.assert_allclose(agent.get_action_with_obs_by_sde(observations, key), expected, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="sde_noise_scale > 0"):
        agent.get_action_and_latent_trace_with_obs_by_sde(observations, key)
    with pytest.raises(ValueError, match="sde_noise_scale > 0"):
        agent.get_latent_trace_log_prob_with_feat_by_sde(features, jnp.zeros((4, 14)))
