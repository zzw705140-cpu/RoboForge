"""Flow Matching 策略的模型创建, 条件组织与速度回归训练."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from roboforg_1.agent.policy.action_utils import project_action_chunks_to_unit_balls

from roboforg_1.model.flax_model import Model
from roboforg_1.model.resnet_loader import load_resnet10_params
from roboforg_1.networks.common import PRNGKey, Params
from roboforg_1.networks.encoder_net import EncoderNet        # 导入编码器类
from roboforg_1.networks.mlp import MLP, MLPWithCondition
from roboforg_1.networks.time_embedding import SinusoidalTimeEmbedding
from roboforg_1.networks.vector_field_net import VectorFieldNet
from roboforg_1.networks.vision.pretrained_encoder import PreTrainedResNetEncoder
from roboforg_1.networks.vision.resnet_v1 import resnetv1_configs


class FlowMatchingAgent(flax.struct.PyTreeNode):
    """保存观测编码器, 速度网络, 优化器状态和可选的上一动作块条件."""

    rng: PRNGKey
    encoder: Model
    vector_field: Model
    config: Any = flax.struct.field(pytree_node=False)

    # 可选encoder
    previous_chunk_encoder: Model | None = None
    previous_chunk: jax.Array | None = None

    @classmethod
    def create(
        cls, *, seed: int, sample_observations: Mapping[str, Any], sample_actions: jax.Array,
        observation_horizon: int = 1, action_horizon: int = 4, use_temporal_history: bool = True,
        pooling_method: str = "task_conditioned_attention", attention_stages: tuple[int, int, int, int] | None = (0, 0, 1, 0),
        train_backbone: bool = False, backbone_lr: float = 1e-5,
        network_type: str = "condition", hidden_dim: int = 256, num_blocks: int = 4, time_embed_dim: int = 32, dropout_rate: float = 0.0,
        encoder_lr: float = 3e-4, vector_field_lr: float = 3e-4,
        num_ode_steps: int = 10, num_sde_steps: int = 10, sde_noise_scale: float = 0.1,
        condition_on_previous_chunk: bool = False, resnet_checkpoint_path: str | Path | None = None,
    ) -> FlowMatchingAgent:
        """按示例输入初始化模型, 加载视觉预训练权重, 配置优化器并返回 Agent.

        示例观测是不含 batch 维的单样本, 动作支持 [H,A] 或展平的 [H*A].
        当前速度主干使用 MLPWithCondition; ODE/SDE 参数先保存供后续采样接口使用.
        resnet_checkpoint_path 可指定本地权重文件, 未指定时采用已有加载器的默认缓存.
        """



        """--------------------------- 输入和超参数检查 ---------------------------"""
        
        for name, value in (("observation_horizon", observation_horizon), ("action_horizon", action_horizon), ("hidden_dim", hidden_dim), ("num_blocks", num_blocks), ("num_ode_steps", num_ode_steps), ("num_sde_steps", num_sde_steps)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not use_temporal_history and observation_horizon != 1:
            raise ValueError("observation_horizon must equal 1 when use_temporal_history=False.")
        if pooling_method not in {"avg", "spatial_learned_embeddings", "task_conditioned_attention"}:
            raise ValueError("pooling_method must be 'avg', 'spatial_learned_embeddings', or 'task_conditioned_attention'.")
        if attention_stages is not None:
            if len(attention_stages) != 4 or any(type(value) not in (bool, int) or value not in (0, 1) for value in attention_stages):
                raise ValueError("attention_stages must contain four 0/1 values.")
            attention_stages = tuple(bool(value) for value in attention_stages)
        if network_type != "condition":
            raise ValueError("Only network_type='condition' is implemented in this project.")
        if not isinstance(time_embed_dim, int) or isinstance(time_embed_dim, bool) or time_embed_dim < 4 or time_embed_dim % 2 != 0:
            raise ValueError("time_embed_dim must be an even integer greater than or equal to 4.")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must lie in [0, 1).")
        for name, value in (("encoder_lr", encoder_lr), ("backbone_lr", backbone_lr), ("vector_field_lr", vector_field_lr)):
            if not 0.0 < value < float("inf"):
                raise ValueError(f"{name} must be finite and positive.")
        if not 0.0 <= sde_noise_scale < float("inf"):
            raise ValueError("sde_noise_scale must be finite and non-negative.")

        # 动作样本用于确定展平动作维度, 每步动作支持六维或带夹爪的七维.
        sample_actions = jnp.asarray(sample_actions)
        if sample_actions.ndim == 2:
            if sample_actions.shape[0] != action_horizon or sample_actions.shape[1] not in (6, 7):
                raise ValueError(f"An unflattened sample action chunk must have shape [{action_horizon},6 or 7], got {sample_actions.shape}.")
            sample_actions = sample_actions.reshape(-1)
        if sample_actions.ndim != 1 or sample_actions.shape[0] == 0:
            raise ValueError(f"sample_actions must be a non-empty flattened action chunk, got {sample_actions.shape}.")
        if not jnp.issubdtype(sample_actions.dtype, jnp.floating):
            raise TypeError("sample_actions must have a floating dtype.")
        flow_dim = sample_actions.shape[0]
        if flow_dim % action_horizon != 0:
            raise ValueError("The sample action length must be divisible by action_horizon.")
        action_dim = flow_dim // action_horizon
        if action_dim not in (6, 7):
            raise ValueError(f"Each action must have 6 or 7 dimensions, got {action_dim}.")

        # 创建阶段使用原始图像初始化共享 Backbone; 训练可使用后续缓存的视觉特征.
        if "images" not in sample_observations:
            raise KeyError("sample_observations must contain 'images'.")
        images = sample_observations["images"]
        if not isinstance(images, Mapping) or not images:
            raise ValueError("sample_observations['images'] must contain at least one camera.")
        image_keys = tuple(images.keys())
        for image_key in image_keys:
            image = jnp.asarray(images[image_key])
            expected_rank = 4 if use_temporal_history else 3
            if image.ndim != expected_rank or image.shape[-1] != 3:
                raise ValueError(f"Camera {image_key!r} must be an unbatched RGB image or history, got {image.shape}.")
            if use_temporal_history and image.shape[0] != observation_horizon:
                raise ValueError(f"Camera {image_key!r} history must match observation_horizon={observation_horizon}.")
        if "state" in sample_observations:
            state = jnp.asarray(sample_observations["state"])
            if use_temporal_history and (state.ndim != 2 or state.shape[0] != observation_horizon):
                raise ValueError("Temporal state must have shape [observation_horizon, state_dim].")
            if not use_temporal_history and state.ndim != 1:
                raise ValueError("Non-temporal state must have shape [state_dim].")

        # 注意力池化接收多阶段字典, 普通池化接收最终一张特征图.
        if pooling_method == "task_conditioned_attention":
            if attention_stages is None or not any(attention_stages):
                raise ValueError("Task-conditioned attention requires at least one selected ResNet stage.")
            if "task_condition" not in sample_observations:
                raise KeyError("Task-conditioned attention requires sample_observations['task_condition'].")
        else:
            attention_stages = None



        """--------------------------- 创建随机数与观测编码器 ---------------------------"""

        rng, encoder_rng, vector_field_rng = jax.random.split(jax.random.PRNGKey(seed), 3)
        pretrained_encoder = resnetv1_configs["resnetv1-10"](attention_stages=attention_stages, freeze_backbone=not train_backbone, name="pretrained_encoder")
        image_encoders = {
            image_key: PreTrainedResNetEncoder(pooling_method=pooling_method, train_backbone=train_backbone, num_spatial_blocks=8, bottleneck_dim=256, pretrained_encoder=pretrained_encoder, name=f"encoder_{image_key}")
            for image_key in image_keys
        }
        encoder_def = EncoderNet(image_encoders=image_encoders, use_proprio="state" in sample_observations, proprio_latent_dim=64, use_temporal_history=use_temporal_history, image_keys=image_keys, pooling_method=pooling_method)

        # 微调时按参数路径分组, 共享 Backbone 使用较小的学习率.
        if train_backbone:
            def parameter_labels(params):
                """标记共享 Backbone 与其余观测编码参数, 供优化器分别更新."""
                return jax.tree_util.tree_map_with_path(lambda path, _: "backbone" if any(getattr(entry, "key", None) == "pretrained_encoder" for entry in path) else "encoder", params)

            encoder_tx = optax.multi_transform({"backbone": optax.adam(backbone_lr), "encoder": optax.adam(encoder_lr)}, parameter_labels)
        else:
            encoder_tx = optax.adam(encoder_lr)

        # 创建 encoder model, 并完成初始化
        encoder = Model.create(network_def=encoder_def, inputs=(sample_observations,), rng=encoder_rng, tx=encoder_tx)
        
        # 加载预训练 resnet backbone 参数
        encoder = load_resnet10_params(encoder, checkpoint_path=resnet_checkpoint_path)

        # 单个初始化样本经 Encoder 得到 [feature_dim]
        sample_features = encoder(sample_observations, train=False)



        """--------------------------- 可选的上一动作块编码器 ---------------------------"""

        previous_chunk_encoder, previous_chunk = None, None
        if condition_on_previous_chunk:
            rng, previous_chunk_rng = jax.random.split(rng)
            previous_chunk = jnp.zeros((flow_dim,), dtype=sample_features.dtype)
            previous_chunk_encoder = Model.create(network_def=MLP(hidden_dims=(128, 64), use_layer_norm=True), inputs=(previous_chunk,), rng=previous_chunk_rng, tx=optax.adam(vector_field_lr))
            sample_features = jnp.concatenate([sample_features, previous_chunk_encoder(previous_chunk, train=False)], axis=-1)



        """--------------------------- 创建条件速度网络 ---------------------------"""

        network = MLPWithCondition(hidden_dims=(*([hidden_dim] * num_blocks), flow_dim), activations=nn.gelu, activate_final=False, use_layer_norm=True, dropout_rate=dropout_rate)
        vector_field_def = VectorFieldNet(time_embedding=SinusoidalTimeEmbedding(embed_dim=time_embed_dim), network=network)
        sample_time = jnp.asarray(0.0, dtype=jnp.float32)
        vector_field = Model.create(network_def=vector_field_def, inputs=(sample_features, sample_actions, sample_time), rng=vector_field_rng, tx=optax.adam(vector_field_lr))

        # 保存网络配置和后续采样所需的设置, 不在创建时执行积分.
        config = flax.core.FrozenDict({
            "observation_horizon": observation_horizon, "action_horizon": action_horizon, "action_dim": action_dim, "flow_dim": flow_dim,
            "use_temporal_history": use_temporal_history, "pooling_method": pooling_method, "attention_stages": attention_stages,
            "train_backbone": train_backbone, "backbone_lr": backbone_lr, "encoder_lr": encoder_lr, "vector_field_lr": vector_field_lr,
            "network_type": network_type, "hidden_dim": hidden_dim, "num_blocks": num_blocks, "time_embed_dim": time_embed_dim, "dropout_rate": dropout_rate,
            "num_ode_steps": num_ode_steps, "num_sde_steps": num_sde_steps, "sde_noise_scale": sde_noise_scale, "image_keys": image_keys,
            "condition_on_previous_chunk": condition_on_previous_chunk,
        })
        return cls(rng=rng, encoder=encoder, vector_field=vector_field, config=config, previous_chunk_encoder=previous_chunk_encoder, previous_chunk=previous_chunk)

    def reset_previous_chunk(self, batch_size: int | None = None) -> FlowMatchingAgent:
        """清空上一动作块的运行状态, 不修改网络参数或优化器状态."""
        if not self.config["condition_on_previous_chunk"]:
            return self
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        shape = (self.config["flow_dim"],) if batch_size is None else (batch_size, self.config["flow_dim"])
        return self.replace(previous_chunk=jnp.zeros(shape, dtype=self.previous_chunk.dtype))

    def _condition_features(self, observations: Mapping[str, Any]) -> jax.Array:
        """编码当前观测, 按配置拼接 Agent 保存的上一动作块特征."""
        features = self.encoder(observations, train=False)
        if not self.config["condition_on_previous_chunk"]:
            return features
        if features.ndim not in (1, 2) or self.previous_chunk.shape != (*features.shape[:-1], self.config["flow_dim"]):
            raise ValueError("previous_chunk must match the observation batch and flow_dim; call reset_previous_chunk with the matching batch_size.")
        chunk_features = self.previous_chunk_encoder(self.previous_chunk, train=False)
        return jnp.concatenate([features, chunk_features], axis=-1)

    def _require_stateless_sampling(self) -> None:
        """检查后续无状态采样接口是否允许忽略上一动作块的运行状态."""
        if self.config["condition_on_previous_chunk"]:
            raise ValueError("Previous-chunk conditioning requires a sampling interface that updates previous_chunk.")


    """--------------------------- Flow Matching 训练 ---------------------------"""

    @staticmethod
    def _build_flow_matching_targets(actions: jax.Array, rng: PRNGKey) -> tuple[jax.Array, jax.Array, jax.Array]:
        """由示范动作构造直线插值状态 x_t, 目标速度和随机流时间 t."""
        actions = jnp.asarray(actions)
        if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] == 0:
            raise ValueError(f"Training actions must have non-empty shape [B, flow_dim], got {actions.shape}.")
        if not jnp.issubdtype(actions.dtype, jnp.floating):
            raise TypeError("Training actions must have a floating dtype.")
        noise_rng, time_rng = jax.random.split(rng)
        x_0 = jax.random.normal(noise_rng, shape=actions.shape, dtype=actions.dtype)
        t = jax.random.uniform(time_rng, shape=(actions.shape[0],), dtype=actions.dtype)
        x_t = (1.0 - t[:, None]) * x_0 + t[:, None] * actions
        target_velocity = actions - x_0
        return x_t, target_velocity, t

    @jax.jit
    def update(self, batch: Mapping[str, Any]) -> tuple[FlowMatchingAgent, dict[str, jax.Array]]:
        """计算速度 MSE, 联合更新相关网络, 返回新 Agent 和训练指标.

        动作支持数据层的 [B,H,A] 或已展平的 [B,H*A].
        可选 action_valid: [B,H], 屏蔽尾部填充动作的训练损失.
        """
        # 本次采样与 Dropout 使用独立随机数, 下一次更新保留新的 key.
        new_rng, loss_rng = jax.random.split(self.rng)
        target_rng, encoder_dropout_rng, vector_field_dropout_rng = jax.random.split(loss_rng, 3)

        def loss_fn(encoder_params: Params, vector_field_params: Params, previous_chunk_encoder_params: Params | None = None) -> tuple[jax.Array, dict[str, jax.Array]]:
            """用显式参数完成前向计算, 为所有启用网络提供同一个速度回归损失."""
            observations, actions = batch["observations"], jnp.asarray(batch["actions"])
            if actions.ndim == 3:
                if actions.shape[1:] != (self.config["action_horizon"], self.config["action_dim"]):
                    raise ValueError("Unflattened training actions must have shape [B, action_horizon, action_dim].")
                actions = actions.reshape((actions.shape[0], self.config["flow_dim"]))
            if actions.ndim != 2 or actions.shape[-1] != self.config["flow_dim"]:
                raise ValueError(f"batch['actions'] must have shape [B,{self.config['flow_dim']}], got {actions.shape}.")
            if self.config["train_backbone"] and "visual_features" in observations:
                raise ValueError("Backbone fine-tuning requires raw images, not cached visual_features.")

            # 示范动作构造监督目标, 当前观测编码为条件特征.
            x_t, target_velocity, t = self._build_flow_matching_targets(actions, target_rng)
            features = self.encoder.apply(encoder_params, observations, train=True, rngs={"dropout": encoder_dropout_rng})
            if self.config["condition_on_previous_chunk"]:
                if "previous_chunk" not in batch:
                    raise KeyError("Previous-chunk conditioning requires batch['previous_chunk'].")
                previous_chunk = jnp.asarray(batch["previous_chunk"])
                if previous_chunk.ndim == 3:
                    if previous_chunk.shape[1:] != (self.config["action_horizon"], self.config["action_dim"]):
                        raise ValueError("Unflattened previous_chunk must have shape [B, action_horizon, action_dim].")
                    previous_chunk = previous_chunk.reshape((previous_chunk.shape[0], self.config["flow_dim"]))
                if previous_chunk.shape != actions.shape:
                    raise ValueError(f"previous_chunk must match actions shape {actions.shape}, got {previous_chunk.shape}.")
                chunk_features = self.previous_chunk_encoder.apply(previous_chunk_encoder_params, previous_chunk, train=True)
                features = jnp.concatenate([features, chunk_features], axis=-1)

            # 在随机时间点回归速度, 训练阶段不执行完整积分和最终动作投影.
            predicted_velocity = self.vector_field.apply(vector_field_params, features, x_t, t, train=True, rngs={"dropout": vector_field_dropout_rng})
            squared_error = jnp.square(predicted_velocity - target_velocity)
            if "action_valid" in batch:
                valid = jnp.asarray(batch["action_valid"], dtype=squared_error.dtype)
                if valid.shape != (actions.shape[0], self.config["action_horizon"]):
                    raise ValueError("action_valid must have shape [B, action_horizon].")
                per_step_error = squared_error.reshape((actions.shape[0], self.config["action_horizon"], self.config["action_dim"])).mean(axis=-1)
                loss = jnp.mean(jnp.sum(per_step_error * valid, axis=-1) / jnp.maximum(jnp.sum(valid, axis=-1), 1.0))
            else:
                loss = jnp.mean(squared_error)
            info = {
                "flow_matching_loss": loss,
                "predicted_velocity_norm": jnp.mean(jnp.linalg.norm(predicted_velocity, axis=-1)),
                "target_velocity_norm": jnp.mean(jnp.linalg.norm(target_velocity, axis=-1)),
                "time_mean": jnp.mean(t),
            }
            return loss, info

        # 对同一个损失求导, 可选的上一动作块编码器也参与更新.
        if self.config["condition_on_previous_chunk"]:
            (_, info), (encoder_grads, vector_field_grads, previous_grads) = jax.value_and_grad(loss_fn, argnums=(0, 1, 2), has_aux=True)(self.encoder.params, self.vector_field.params, self.previous_chunk_encoder.params)
            new_previous_chunk_encoder = self.previous_chunk_encoder.apply_gradient(previous_grads)
        else:
            (_, info), (encoder_grads, vector_field_grads) = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(self.encoder.params, self.vector_field.params)
            new_previous_chunk_encoder = None

        # 各模型使用创建时配置的优化器, 返回更新后的不可变 Agent.
        new_encoder = self.encoder.apply_gradient(encoder_grads)
        new_vector_field = self.vector_field.apply_gradient(vector_field_grads)
        new_agent = self.replace(rng=new_rng, encoder=new_encoder, vector_field=new_vector_field, previous_chunk_encoder=new_previous_chunk_encoder)
        return new_agent, info

    ####################################################
    #                     ode采样                      #
    ####################################################  

    @staticmethod
    def get_action_and_latent_trace_with_feat_and_noise_by_ode(vector_field: Model, features: jax.Array, initial_noise: jax.Array, *, num_ode_steps: int, action_horizon: int, action_dim: int) -> tuple[jax.Array, jax.Array]:
        """从给定噪声完成 Flow ODE 积分, 并返回投影动作和未投影积分轨迹"""

        # 检查观测特征与初始噪声的布局, 以及积分配置和动作维度.
        if features.ndim not in (1, 2):
            raise ValueError(f"features must have shape [feature_dim] or [batch, feature_dim], got {features.shape}.")
        if initial_noise.ndim != features.ndim:
            raise ValueError(f"features and initial_noise must either both be unbatched or both be batched, got {features.shape} and {initial_noise.shape}.")
        if features.shape[:-1] != initial_noise.shape[:-1]:
            raise ValueError(f"features and initial_noise must have matching leading dimensions, got {features.shape} and {initial_noise.shape}.")
        if num_ode_steps <= 0:
            raise ValueError("num_ode_steps must be positive.")
        if initial_noise.shape[-1] != action_horizon * action_dim:
            raise ValueError(f"The final initial_noise dimension must equal action_horizon * action_dim = {action_horizon * action_dim}, got {initial_noise.shape[-1]}.")

        if not jnp.issubdtype(initial_noise.dtype, jnp.floating):
            raise TypeError("initial_noise must have a floating dtype.")

        # 从 t=0 出发, 用固定步长积分, 轨迹包含初始噪声和每步结果.
        input_is_batched = (features.ndim == 2)
        x_t = initial_noise
        dt = jnp.asarray(1.0 / num_ode_steps, dtype=x_t.dtype)
        latent_states = [x_t]

        # 左端点显式 Euler: x_(i+1) = x_i + v(x_i, t_i) * dt
        for step in range(num_ode_steps):
            time_value = jnp.asarray(step / num_ode_steps, dtype=x_t.dtype)
            if input_is_batched:
                t = jnp.full((features.shape[0],), time_value, dtype=x_t.dtype)
            else:
                t = time_value

            velocity = vector_field(features, x_t, t, train=False)
            x_t = x_t + velocity * dt
            latent_states.append(x_t)

        # 只约束积分完成后的动作, 不限制中间流状态或向量场速度.
        actions = project_action_chunks_to_unit_balls(x_t, action_horizon=action_horizon, action_dim=action_dim)
        latent_chain = jnp.stack(latent_states, axis=-2)
        return actions, latent_chain

    @staticmethod
    def get_action_with_feat_and_noise_by_ode(vector_field: Model, features: jax.Array, initial_noise: jax.Array, *, num_ode_steps: int, action_horizon: int, action_dim: int) -> jax.Array:
        """从给定初始噪声出发, 使用已编码的观测特征确定性完成 Flow ODE 积分."""

        actions, _ = FlowMatchingAgent.get_action_and_latent_trace_with_feat_and_noise_by_ode(vector_field, features, initial_noise, num_ode_steps=num_ode_steps, action_horizon=action_horizon, action_dim=action_dim)
        return actions

    @jax.jit
    def get_action_and_latent_trace_with_obs_and_noise_by_ode(self, observations: Mapping[str, Any], initial_noise: jax.Array) -> tuple[jax.Array, jax.Array]:
        """编码观测, 从指定噪声生成动作, 并返回完整 ODE 积分轨迹"""

        self._require_stateless_sampling()
        features = self.encoder(observations, train=False)
        return self.get_action_and_latent_trace_with_feat_and_noise_by_ode(self.vector_field, features, initial_noise, num_ode_steps=self.config["num_ode_steps"], action_horizon=self.config["action_horizon"], action_dim=self.config["action_dim"])

    @jax.jit
    def get_action_with_obs_and_noise_by_ode(self, observations: Mapping[str, Any], initial_noise: jax.Array) -> jax.Array:
        """编码观测, 并从调用方给定的初始噪声确定性生成 action chunk"""

        self._require_stateless_sampling()
        features = self.encoder(observations, train=False)
        return self.get_action_with_feat_and_noise_by_ode(self.vector_field, features, initial_noise, num_ode_steps=self.config["num_ode_steps"], action_horizon=self.config["action_horizon"], action_dim=self.config["action_dim"])

    @jax.jit
    def get_action_with_obs_by_ode(self, observations: Mapping[str, Any], rng: PRNGKey) -> jax.Array:
        """
        使用 Euler ODE 积分生成动作 chunk, 从 Normal(0, I) 出发, 在 [0,1] 之间调用向量场 ``num_ode_steps`` 次

        Returns:
            单个观测返回 ``[flow_dim]``
            批量观测返回 ``[B, flow_dim]``
        """

        self._require_stateless_sampling()
        # 观测条件与积分时间无关, 只需编码一次并在全部 Euler 步中复用
        features = self.encoder(observations, train=False)
        if features.ndim not in (1, 2):
            raise ValueError(f"The encoder must return [feature_dim] or [batch, feature_dim], got {features.shape}.")

        leading_shape = features.shape[:-1]
        flow_shape = (*leading_shape, self.config["flow_dim"])

        # 采用的 Flow Matching 源分布 N(0, I)
        initial_noise = jax.random.normal(rng, shape=flow_shape, dtype=features.dtype)

        return self.get_action_with_feat_and_noise_by_ode(self.vector_field, features, initial_noise, num_ode_steps=self.config["num_ode_steps"], action_horizon=self.config["action_horizon"], action_dim=self.config["action_dim"])


    ##################################################
    #                     sde采样                    #
    ##################################################

    def _get_transition_dist_params_by_sde(self, features: jax.Array, x_t: jax.Array, t: jax.Array, dt: jax.Array) -> tuple[jax.Array, jax.Array]:
        """
        计算从 ``x_t`` 到 ``x_(t+dt)`` 的 Euler-Maruyama 单步 SDE 高斯转移分布 ``N(mean, std^2 I)`` 的 mean 和 std

        * 单个输入时 t 和 std 为标量
        * 批量输入时 t 和 std 尺寸分别为 [B], [B, 1]
        """

        input_is_batched = (features.ndim == 2)

        # Flow Matching 向量场给出 probability-flow ODE 速度
        velocity = self.vector_field(features, x_t, t, train=False)

        # 单样本时 t 为标量, 批量输入时 t 为 [B]; 只有批量输入需要扩展为 [B, 1]
        if input_is_batched:
            t_expanded = t[:, None]
        else:
            t_expanded = t

        # tapered diffusion 系数: sigma_t = sigma_init * sqrt(1 - t)
        noise_scale = jnp.asarray(self.config["sde_noise_scale"], dtype=x_t.dtype)
        sigma_t     = noise_scale * jnp.sqrt(jnp.maximum(1.0 - t_expanded, 0.0))

        # 使用 GCP 的 ODE 到 SDE 漂移修正, 与随时间衰减的扩散系数配套
        drift_correction = 0.5 * jnp.square(noise_scale) * (t_expanded * velocity - x_t)
        corrected_drift  = velocity + drift_correction

        # Euler-Maruyama 转移的高斯均值和标准差
        mean = x_t + corrected_drift * dt
        std  = sigma_t * jnp.sqrt(dt)

        return mean, std

    @jax.jit
    def get_action_and_latent_trace_with_obs_by_sde(self, observations: Mapping[str, Any], rng: PRNGKey) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        通过 SDE 积分生成动作, 并返回完整的隐变量采样轨迹及其对数概率

        Returns:
            单个观测: 
                actions:        ``[flow_dim]``
                latent_chain:   ``[num_sde_steps + 1, flow_dim]``
                step_log_probs: ``[num_sde_steps]``
                chain_log_prob: ``[]``

            批量观测: 
                actions:        ``[B, flow_dim]``
                latent_chain:   ``[B, num_sde_steps + 1, flow_dim]``
                step_log_probs: ``[B, num_sde_steps]``
                chain_log_prob: ``[B]``
        """

        self._require_stateless_sampling()
        features = self.encoder(observations, train=False)
        return self.get_action_and_latent_trace_with_feat_by_sde(features, rng)

    @jax.jit
    def get_action_and_latent_trace_with_feat_by_sde(self, features: jax.Array, rng: PRNGKey) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """从标准高斯源分布采样初始噪声, 再基于已编码特征生成完整 SDE 轨迹."""

        if features.ndim not in (1, 2):
            raise ValueError(f"features must have shape [feature_dim] or [batch, feature_dim], got {features.shape}.")

        flow_shape = (*features.shape[:-1], self.config["flow_dim"])
        rng, initial_rng = jax.random.split(rng)
        initial_noise = jax.random.normal(initial_rng, shape=flow_shape, dtype=features.dtype)
        return self.get_action_and_latent_trace_with_feat_and_noise_by_sde(features, initial_noise, rng)

    def get_action_and_latent_trace_with_feat_and_noise_by_sde(self, features: jax.Array, initial_noise: jax.Array, rng: PRNGKey) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """从调用方给定的初始噪声采样 SDE 轨迹; ``rng`` 只用于转移噪声."""

        # 验证输入布局, 概率计算要求正的扩散强度.
        if features.ndim not in (1, 2) or initial_noise.ndim != features.ndim or initial_noise.shape[:-1] != features.shape[:-1]:
            raise ValueError(f"features and initial_noise must have matching single-sample or batch layouts, got {features.shape} and {initial_noise.shape}.")
        if initial_noise.shape[-1] != self.config["flow_dim"]:
            raise ValueError("initial_noise must end with flow_dim.")
        if not jnp.issubdtype(initial_noise.dtype, jnp.floating):
            raise TypeError("initial_noise must have a floating dtype.")
        if self.config["sde_noise_scale"] <= 0:
            raise ValueError("SDE log probabilities require sde_noise_scale > 0; use the action-only interface for zero noise.")

        input_is_batched = (features.ndim == 2)
        flow_shape = initial_noise.shape
        x_t = initial_noise

        num_sde_steps = self.config["num_sde_steps"]
        dt = jnp.asarray(1.0 / num_sde_steps, dtype=x_t.dtype)
        log_two_pi = jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=x_t.dtype))

        latent_states        = [x_t]    # 潜状态集合
        transition_log_probs = []       # 状态转移对数概率

        # 左端点 Euler-Maruyama: x_(i+1) = x_i + (velocity + 偏移修正系数) * dt + 扩散系数 * sqrt(dt) * epsilon
        for step in range(num_sde_steps):
            # 计算时间
            time_value = jnp.asarray(step / num_sde_steps, dtype=x_t.dtype)
            if input_is_batched:
                t = jnp.full((features.shape[0],), time_value, dtype=x_t.dtype)
            else:
                t = time_value

            # 计算从 x_k 到 x_(k+1) 的对角高斯分布参数
            mean, std = self._get_transition_dist_params_by_sde(features, x_t, t, dt)

            # 采样噪声
            rng, noise_rng = jax.random.split(rng)
            epsilon = jax.random.normal(noise_rng, shape=flow_shape, dtype=x_t.dtype)

            # 从当前步的高斯转移分布中采样
            x_next = mean + std * epsilon

            # 对 flow 维度求和, 得到该步高斯条件转移的 log probability
            normalized_residual = (x_next - mean) / std
            step_log_prob = -0.5 * jnp.sum(jnp.square(normalized_residual) + 2.0 * jnp.log(std) + log_two_pi, axis=-1)

            latent_states.append(x_next)
            transition_log_probs.append(step_log_prob)
            x_t = x_next

        # axis=-2 同时兼容 [D] 和 [B,D]
        latent_chain   = jnp.stack(latent_states, axis=-2)          # 单样本, 生成 [K+1, D],    batch 生成 [B, K+1, D]
        step_log_probs = jnp.stack(transition_log_probs, axis=-1)   # 单样本, 生成 [K],         batch 生成 [B, K]
        # 条件转移 log probability 之和, 不含初始噪声密度, 也不是投影动作的密度.
        chain_log_prob = jnp.sum(step_log_probs, axis=-1)           # 单样本, 生成 [],          batch 生成 [B]

        # 概率始终对未投影的 latent_chain 计算; 只在交给环境前投影最终动作
        actions = project_action_chunks_to_unit_balls(x_t, action_horizon=self.config["action_horizon"], action_dim=self.config["action_dim"])

        return actions, latent_chain, step_log_probs, chain_log_prob

    @jax.jit
    def get_action_and_latent_trace_with_obs_and_noise_by_sde(self, observations: Mapping[str, Any], initial_noise: jax.Array, rng: PRNGKey) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """编码观测, 从指定初始噪声采样 SDE, 并返回动作, 轨迹和转移概率."""

        self._require_stateless_sampling()
        features = self.encoder(observations, train=False)
        return self.get_action_and_latent_trace_with_feat_and_noise_by_sde(features, initial_noise, rng)

    @jax.jit
    def get_latent_trace_log_prob_with_obs_by_sde(self, observations: Mapping[str, Any], latent_chain: jax.Array) -> tuple[jax.Array, jax.Array]:
        """
        使用当前网络参数重新计算一条已有 SDE 隐变量轨迹的对数概率

        Args:
            observations: 产生该隐变量轨迹时使用的观测
            latent_chain: 单样本为 ``[K + 1, flow_dim]``, 批量为 ``[B, K + 1, flow_dim]``

        Returns:
            单样本返回: 每一步条件转移 log probability [K],     整条链的 log probability []
            批量返回:   每一步条件转移 log probability [B, K],  整条链的 log probability [B]
        """

        self._require_stateless_sampling()
        features = self.encoder(observations, train=False)
        return self.get_latent_trace_log_prob_with_feat_by_sde(features, latent_chain)

    @jax.jit
    def get_latent_trace_log_prob_with_feat_by_sde(self, features: jax.Array, latent_chain: jax.Array) -> tuple[jax.Array, jax.Array]:
        """基于已编码特征重新计算已有 SDE 隐变量轨迹的逐步及整链对数概率."""

        if features.ndim not in (1, 2):
            raise ValueError(f"The encoder must return [feature_dim] or [batch, feature_dim], got {features.shape}.")
        if self.config["sde_noise_scale"] <= 0:
            raise ValueError("SDE log probabilities require sde_noise_scale > 0.")
        if not jnp.issubdtype(latent_chain.dtype, jnp.floating):
            raise TypeError("latent_chain must have a floating dtype.")
        if latent_chain.ndim != features.ndim + 1:
            raise ValueError(f"latent_chain must have shape [K + 1, flow_dim] for one sample or [B, K + 1, flow_dim] for batched samples, got {latent_chain.shape}.")
        if latent_chain.shape[:-2] != features.shape[:-1]:
            raise ValueError(f"features and latent_chain must have matching leading dimensions, got {features.shape} and {latent_chain.shape}.")
        if latent_chain.shape[-2] != self.config["num_sde_steps"] + 1:
            raise ValueError(f"The latent chain length must equal num_sde_steps + 1 = {self.config['num_sde_steps'] + 1}, got {latent_chain.shape[-2]}.")
        if latent_chain.shape[-1] != self.config["flow_dim"]:
            raise ValueError(f"The final latent_chain dimension must equal flow_dim = {self.config['flow_dim']}, got {latent_chain.shape[-1]}.")

        input_is_batched = (features.ndim == 2)
        num_sde_steps = self.config["num_sde_steps"]
        dt = jnp.asarray(1.0 / num_sde_steps, dtype=latent_chain.dtype)
        log_two_pi = jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=latent_chain.dtype))

        transition_log_probs = []

        for step in range(num_sde_steps):
            time_value = jnp.asarray(step / num_sde_steps, dtype=latent_chain.dtype)
            if input_is_batched:
                t = jnp.full((features.shape[0],), time_value, dtype=latent_chain.dtype)
            else:
                t = time_value

            # 直接取出已保存的相邻隐状态, 不在概率重算时重新采样
            x_t    = latent_chain[..., step, :]
            x_next = latent_chain[..., step + 1, :]

            mean, std = self._get_transition_dist_params_by_sde(features, x_t, t, dt)

            # 对 flow 维度求和, 得到该步对角高斯条件转移的 log probability
            normalized_residual = (x_next - mean) / std
            step_log_prob = -0.5 * jnp.sum(jnp.square(normalized_residual) + 2.0 * jnp.log(std) + log_two_pi, axis=-1)
            transition_log_probs.append(step_log_prob)

        step_log_probs = jnp.stack(transition_log_probs, axis=-1)   # 单样本生成 [K], batch 生成 [B, K]
        # 条件转移 log probability 之和, 不含初始噪声密度, 也不是投影动作的密度.
        chain_log_prob = jnp.sum(step_log_probs, axis=-1)           # 单样本生成 [],  batch 生成 [B]

        return step_log_probs, chain_log_prob

    @jax.jit
    def get_action_with_obs_by_sde(self, observations: Mapping[str, Any], rng: PRNGKey) -> jax.Array:
        """
        通过 SDE 积分生成动作 chunk

        Returns:
            单个观测返回 ``[flow_dim]``
            批量观测返回 ``[B, flow_dim]``
        """

        self._require_stateless_sampling()
        # 观测条件与积分时间无关, 只需编码一次并在全部 SDE 步中复用
        features = self.encoder(observations, train=False)
        return self._get_action_with_feat_by_sde(features, rng)

    def _get_action_with_feat_by_sde(self, features: jax.Array, rng: PRNGKey) -> jax.Array:
        """从已编码条件采样; 供无状态及有状态入口共同使用"""

        if features.ndim not in (1, 2):
            raise ValueError(f"The encoder must return [feature_dim] or [batch, feature_dim], got {features.shape}.")

        input_is_batched = (features.ndim == 2)
        leading_shape = features.shape[:-1]
        flow_shape = (*leading_shape, self.config["flow_dim"])

        # 分离初始状态和后续每个 Euler-Maruyama 步骤使用的随机数
        rng, initial_rng = jax.random.split(rng)
        x_t = jax.random.normal(initial_rng, shape=flow_shape, dtype=features.dtype)

        num_sde_steps = self.config["num_sde_steps"]
        dt = jnp.asarray(1.0 / num_sde_steps, dtype=x_t.dtype)

        # 左端点 Euler-Maruyama: x_(i+1) ~ N(mean_i, std_i^2 I)
        for step in range(num_sde_steps):
            # 计算时间
            time_value = jnp.asarray(step / num_sde_steps, dtype=x_t.dtype)
            if input_is_batched:
                t = jnp.full((features.shape[0],), time_value, dtype=x_t.dtype)
            else:
                t = time_value

            # 获取单步 SDE 高斯转移分布的均值和标准差
            mean, std = self._get_transition_dist_params_by_sde(features, x_t, t, dt)

            # 采样噪声
            rng, noise_rng = jax.random.split(rng)
            epsilon = jax.random.normal(noise_rng, shape=flow_shape, dtype=x_t.dtype)

            # 从当前步的高斯转移分布中采样
            x_t = mean + std * epsilon

        # 只约束积分完成后的动作样本, 不限制中间流状态或向量场速度
        return project_action_chunks_to_unit_balls(x_t, action_horizon=self.config["action_horizon"], action_dim=self.config["action_dim"])
