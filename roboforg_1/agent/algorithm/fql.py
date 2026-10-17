"""基于动作块的 FQL 算法：行为流、单步 Actor 与 Q 价值学习。"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from roboforg_1.agent.policy.action_utils import project_action_chunks_to_unit_balls
from roboforg_1.agent.policy.flow_matching import FlowMatchingAgent
from roboforg_1.model.flax_model import Model
from roboforg_1.model.resnet_loader import load_resnet10_params
from roboforg_1.networks.actor_nets import OneStepActorNet
from roboforg_1.networks.common import PRNGKey, Params
from roboforg_1.networks.critic_nets import ActionValueCritic, ensemblize
from roboforg_1.networks.encoder_net import EncoderNet
from roboforg_1.networks.mlp import MLP, MLPWithCondition
from roboforg_1.networks.time_embedding import SinusoidalTimeEmbedding
from roboforg_1.networks.vector_field_net import VectorFieldNet
from roboforg_1.networks.vision.pretrained_encoder import PreTrainedResNetEncoder
from roboforg_1.networks.vision.resnet_v1 import resnetv1_configs


##############################################################################
#                                 FQL Agent                                  #
##############################################################################

class FQLAGENT(flax.struct.PyTreeNode):
    """保存共享观测编码器、行为流、单步 Actor 和在线及目标 Critic。"""

    rng: PRNGKey
    encoder: Model
    behavior_flow: Model
    one_step_actor: Model
    critic: Model
    target_critic: Model
    config: Any = flax.struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        sample_observations: dict[str, jax.Array],
        sample_actions: jax.Array,
        observation_horizon: int = 1,
        action_horizon: int = 4,
        use_temporal_history: bool = True,
        network_type: str = "condition",
        hidden_dim: int = 512,
        num_blocks: int = 4,
        critic_hidden_dims: tuple[int, ...] = (512, 512, 512, 512),
        time_embed_dim: int = 32,
        dropout_rate: float = 0.0,
        num_qs: int = 2,
        discount: float = 0.99,
        tau: float = 0.005,
        alpha: float = 100.0,
        q_aggregation: str = "mean",
        num_ode_steps: int = 10,
        encoder_lr: float = 3e-4,
        behavior_flow_lr: float = 3e-4,
        one_step_actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
    ) -> FQLAGENT:
        """检查输入、初始化各网络并返回 Agent；示例观测不含 batch 维，动作块须展平。"""

        # 输入和网络结构检查
        if not isinstance(use_temporal_history, bool):
            raise TypeError("use_temporal_history must be a bool.")
        if not isinstance(observation_horizon, int) or isinstance(observation_horizon, bool) or observation_horizon <= 0:
            raise ValueError("observation_horizon must be positive.")
        if not isinstance(action_horizon, int) or isinstance(action_horizon, bool) or action_horizon <= 0:
            raise ValueError("action_horizon must be positive.")
        if not use_temporal_history and observation_horizon != 1:
            raise ValueError("observation_horizon must equal 1 when use_temporal_history=False.")
        if network_type != "condition":
            raise ValueError("Only network_type='condition' is implemented in this project.")
        if not isinstance(hidden_dim, int) or isinstance(hidden_dim, bool) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if not isinstance(num_blocks, int) or isinstance(num_blocks, bool) or num_blocks <= 0:
            raise ValueError("num_blocks must be positive.")
        if not critic_hidden_dims or any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in critic_hidden_dims):
            raise ValueError("critic_hidden_dims must contain only positive dimensions.")
        if not isinstance(time_embed_dim, int) or isinstance(time_embed_dim, bool) or time_embed_dim < 4 or time_embed_dim % 2 != 0:
            raise ValueError("time_embed_dim must be an even integer greater than or equal to 4.")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must lie in [0, 1).")

        # FQL 超参数检查
        if not isinstance(num_qs, int) or isinstance(num_qs, bool) or num_qs <= 0:
            raise ValueError("num_qs must be positive.")
        if not 0.0 <= discount <= 1.0:
            raise ValueError("discount must lie in [0, 1].")
        if not 0.0 <= tau <= 1.0:
            raise ValueError("tau must lie in [0, 1].")
        if not 0.0 <= alpha < float("inf"):
            raise ValueError("alpha must be finite and non-negative.")
        if q_aggregation not in {"mean", "min"}:
            raise ValueError("q_aggregation must be either 'mean' or 'min'.")
        if not isinstance(num_ode_steps, int) or isinstance(num_ode_steps, bool) or num_ode_steps <= 0:
            raise ValueError("num_ode_steps must be positive.")

        # 优化器学习率检查
        if not 0.0 < encoder_lr < float("inf"):
            raise ValueError("encoder_lr must be finite and positive.")
        if not 0.0 < behavior_flow_lr < float("inf"):
            raise ValueError("behavior_flow_lr must be finite and positive.")
        if not 0.0 < one_step_actor_lr < float("inf"):
            raise ValueError("one_step_actor_lr must be finite and positive.")
        if not 0.0 < critic_lr < float("inf"):
            raise ValueError("critic_lr must be finite and positive.")

        # sample_actions 必须是已经展平的完整 action chunk
        sample_actions = jnp.asarray(sample_actions)
        if sample_actions.ndim != 1:
            raise ValueError(f"sample_actions must be a flattened action chunk with shape [action_horizon * action_dim], got {sample_actions.shape}.")
        flow_dim = int(sample_actions.shape[0])
        if flow_dim == 0:
            raise ValueError("sample_actions cannot be empty.")
        if flow_dim % action_horizon != 0:
            raise ValueError(f"The flattened sample_actions length must be divisible by action_horizon; got length={flow_dim}, action_horizon={action_horizon}.")
        action_dim = flow_dim // action_horizon
        if action_dim not in (6, 7):
            raise ValueError(f"Each single-arm action must have 6 or 7 dimensions, got action_dim={action_dim}.")
        if not jnp.issubdtype(sample_actions.dtype, jnp.floating):
            raise TypeError(f"sample_actions must have a floating dtype, got {sample_actions.dtype}.")

        # 检查单个初始化 observation 的相机和时间维度
        if not isinstance(sample_observations, Mapping):
            raise TypeError("sample_observations must be a mapping.")
        if "visual_features" in sample_observations:
            raise ValueError("Initialization requires raw images, not cached visual_features.")
        if "images" not in sample_observations:
            raise KeyError("sample_observations must contain an 'images' namespace.")
        images = sample_observations["images"]
        if not isinstance(images, Mapping):
            raise TypeError("sample_observations['images'] must be a camera mapping.")
        image_keys = tuple(images.keys())
        if not image_keys:
            raise ValueError("sample_observations['images'] must contain at least one image.")

        for image_key in image_keys:
            image = jnp.asarray(images[image_key])
            if use_temporal_history:
                if image.ndim != 4:
                    raise ValueError(f"Temporal image {image_key!r} must have shape [T,H,W,C], got {image.shape}.")
                if image.shape[0] != observation_horizon:
                    raise ValueError(f"Temporal image {image_key!r} has T={image.shape[0]}, but observation_horizon={observation_horizon}.")
            elif image.ndim != 3:
                raise ValueError(f"Image {image_key!r} must have shape [H,W,C] when use_temporal_history=False, got {image.shape}.")

            if image.shape[-1] != 3 or any(dim <= 0 for dim in image.shape):
                raise ValueError(f"Camera {image_key!r} must contain non-empty RGB images, got {image.shape}.")

        if "state" in sample_observations:
            state = jnp.asarray(sample_observations["state"])
            if use_temporal_history:
                if state.ndim != 2 or state.shape[0] != observation_horizon:
                    raise ValueError(f"Temporal state must have shape [T,state_dim] with T={observation_horizon}, got {state.shape}.")
            elif state.ndim != 1:
                raise ValueError(f"State must have shape [state_dim] when use_temporal_history=False, got {state.shape}.")

        # 各网络使用独立的初始化随机数，rng 保留供后续训练使用。
        rng = jax.random.PRNGKey(seed)
        rng, encoder_rng, behavior_flow_rng, one_step_actor_rng, critic_rng = jax.random.split(rng, 5)


        """------------------------- 创建 encoder model -------------------------"""

        # 各相机共享冻结的 ResNet-10，但拥有独立的空间池化和瓶颈投影。
        pretrained_encoder = resnetv1_configs["resnetv1-10"](
            freeze_backbone=True,
            name="pretrained_encoder",
        )
        image_encoders = {
            image_key: PreTrainedResNetEncoder(
                pooling_method="spatial_learned_embeddings",
                train_backbone=False,
                num_spatial_blocks=8,
                bottleneck_dim=256,
                pretrained_encoder=pretrained_encoder,
                name=f"encoder_{image_key}",
            )
            for image_key in image_keys
        }

        encoder_def = EncoderNet(
            image_encoders=image_encoders,
            use_proprio="state" in sample_observations,
            proprio_latent_dim=64,
            use_temporal_history=use_temporal_history,
            image_keys=image_keys,
            pooling_method="spatial_learned_embeddings",
        )
        encoder = Model.create(
            network_def=encoder_def,
            inputs=(sample_observations,),
            rng=encoder_rng,
            tx=optax.adam(learning_rate=encoder_lr),
        )
        encoder = load_resnet10_params(encoder)

        # 后续 Behavior Flow、Actor 和 Critic 根据此特征初始化输入维度。
        sample_features = encoder(sample_observations, train=False)


        """------------------------- 创建 vector field -------------------------"""

        # num_blocks 个隐藏层，最后用线性层输出 flow_dim 维速度。
        behavior_network = MLPWithCondition(
            hidden_dims=(*([hidden_dim] * num_blocks), flow_dim),
            activations=nn.gelu,
            activate_final=False,
            use_layer_norm=True,
            dropout_rate=dropout_rate,
        )
        behavior_flow_def = VectorFieldNet(
            time_embedding=SinusoidalTimeEmbedding(embed_dim=time_embed_dim),
            network=behavior_network,
        )

        # 使用单个观测特征、动作块与标量流时间初始化条件速度网络。
        sample_time = jnp.asarray(0.0, dtype=sample_actions.dtype)
        behavior_flow = Model.create(
            network_def=behavior_flow_def,
            inputs=(sample_features, sample_actions, sample_time),
            rng=behavior_flow_rng,
            tx=optax.adam(learning_rate=behavior_flow_lr),
        )

        """------------------------- 创建 one-step actor -------------------------"""

        # 一次前向输出完整动作块，沿用 gcp 的无 LayerNorm 配置。
        one_step_network = MLPWithCondition(
            hidden_dims=(*([hidden_dim] * num_blocks), flow_dim),
            activations=nn.gelu,
            activate_final=False,
            use_layer_norm=False,
            dropout_rate=dropout_rate,
        )


        one_step_actor_def = OneStepActorNet(network=one_step_network)
        sample_noise = jnp.zeros_like(sample_actions)
        one_step_actor = Model.create(
            network_def=one_step_actor_def,
            inputs=(sample_features, sample_noise),
            rng=one_step_actor_rng,
            tx=optax.adam(learning_rate=one_step_actor_lr),
        )

        """------------------------- 创建 critic -------------------------"""

        critic_cls = partial(
            ActionValueCritic,
            network=MLP(
                hidden_dims=(*critic_hidden_dims, 1),
                activations=nn.gelu,
                activate_final=False,
                use_layer_norm=True,
            ),
            output_activation=None,                  # FQL 的 Q 表示无界累计回报
        )
        critic_def = ensemblize(critic_cls, num_qs=num_qs, out_axes=0)()
        critic = Model.create(
            network_def=critic_def,
            inputs=(sample_features, sample_actions),
            rng=critic_rng,
            tx=optax.adam(learning_rate=critic_lr),
        )

        """------------------------- 创建 target critic -------------------------"""

        # Target Critic 与在线 Critic 参数完全相同，但不持有优化器和 Adam 状态
        target_critic = critic.replace(tx=None, opt_state=None)


        """------------------------- 创建 FQLAGENT 实例 -------------------------"""

        config = flax.core.FrozenDict(
            {
                "observation_horizon": observation_horizon,
                "action_horizon": action_horizon,
                "action_dim": action_dim,
                "flow_dim": flow_dim,
                "use_temporal_history": use_temporal_history,
                "network_type": network_type,
                "hidden_dim": hidden_dim,
                "num_blocks": num_blocks,
                "critic_hidden_dims": tuple(critic_hidden_dims),
                "time_embed_dim": time_embed_dim,
                "dropout_rate": dropout_rate,
                "num_qs": num_qs,
                "discount": discount,
                "tau": tau,
                "alpha": alpha,
                "q_aggregation": q_aggregation,
                "num_ode_steps": num_ode_steps,
                "image_keys": image_keys,
            }
        )

        return cls(
            rng=rng,
            encoder=encoder,
            behavior_flow=behavior_flow,
            one_step_actor=one_step_actor,
            critic=critic,
            target_critic=target_critic,
            config=config,
        )

    def _update_behavior_flow(self, batch: flax.core.FrozenDict) -> tuple[FQLAGENT, dict[str, jax.Array]]:
        """使用 Flow Matching loss 联合更新 Encoder 和 Behavior Flow"""

        # new_rng 保存到新 Agent，loss_rng 只供本次目标采样和 Dropout 使用
        new_rng, loss_rng = jax.random.split(self.rng)
        target_rng, encoder_dropout_rng, behavior_flow_dropout_rng = jax.random.split(loss_rng, 3)

        def loss_fn(encoder_params: Params, behavior_flow_params: Params) -> tuple[jax.Array, dict[str, jax.Array]]:
            """计算 Behavior Flow 的 Flow Matching 速度回归损失"""

            observations = batch["observations"]
            actions = jnp.asarray(batch["actions"])
            if actions.ndim == 3:
                if actions.shape[1:] != (self.config["action_horizon"], self.config["action_dim"]):
                    raise ValueError("Unflattened training actions must have shape [B, action_horizon, action_dim].")
                actions = actions.reshape((actions.shape[0], self.config["flow_dim"]))
            if actions.ndim != 2 or actions.shape[-1] != self.config["flow_dim"]:
                raise ValueError(f"Training actions must have shape [B,{self.config['flow_dim']}], got {actions.shape}.")

            # 创建 Flow Matching 向量场回归目标
            x_t, target_velocity, t = FlowMatchingAgent._build_flow_matching_targets(actions, target_rng)

            # 使用显式待求导参数，梯度会同时回传到 Encoder 和 Behavior Flow
            features = self.encoder.apply(
                encoder_params,
                observations,
                train=True,
                rngs={"dropout": encoder_dropout_rng},
            )
            predicted_velocity = self.behavior_flow.apply(
                behavior_flow_params,
                features,
                x_t,
                t,
                train=True,
                rngs={"dropout": behavior_flow_dropout_rng},
            )

            # 沿用 BC 的有效动作掩码，尾部填充不参与速度回归。
            squared_error = jnp.square(predicted_velocity - target_velocity)
            if "action_valid" in batch:
                valid = jnp.asarray(batch["action_valid"], dtype=squared_error.dtype)
                if valid.shape != (actions.shape[0], self.config["action_horizon"]):
                    raise ValueError("action_valid must have shape [B, action_horizon].")
                per_step_error = squared_error.reshape(
                    (actions.shape[0], self.config["action_horizon"], self.config["action_dim"])
                ).mean(axis=-1)
                flow_matching_loss = jnp.mean(
                    jnp.sum(per_step_error * valid, axis=-1)
                    / jnp.maximum(jnp.sum(valid, axis=-1), 1.0)
                )
            else:
                flow_matching_loss = jnp.mean(squared_error)

            info = {
                "flow_matching_loss": flow_matching_loss,
                "predicted_velocity_norm": jnp.mean(jnp.linalg.norm(predicted_velocity, axis=-1)),
                "target_velocity_norm": jnp.mean(jnp.linalg.norm(target_velocity, axis=-1)),
                "time_mean": jnp.mean(t),
            }

            return flow_matching_loss, info

        # 同一个 loss 同时对 Encoder 和 Behavior Flow 参数求导
        (_, info), (encoder_grads, behavior_flow_grads) = jax.value_and_grad(
            loss_fn,
            argnums=(0, 1),
            has_aux=True,
        )(self.encoder.params, self.behavior_flow.params)

        # 分别使用各自 Adam 优化器更新 Encoder 和 Behavior Flow
        new_encoder = self.encoder.apply_gradient(encoder_grads)
        new_behavior_flow = self.behavior_flow.apply_gradient(behavior_flow_grads)

        new_agent = self.replace(
            rng=new_rng,
            encoder=new_encoder,
            behavior_flow=new_behavior_flow,
        )

        return new_agent, info

    def _update_critic(
        self,
        batch: flax.core.FrozenDict,
        features: jax.Array,
        next_features: jax.Array,
    ) -> tuple[FQLAGENT, dict[str, jax.Array]]:
        """使用 chunk TD loss 更新 Critic。

        rewards 为 H 步折扣奖励和，masks 为是否允许 bootstrap 的 [B] 标记。
        允许 bootstrap 时，next_features 必须对应执行 H 步后的观测；
        真正终止时 masks=0，截断处不足 H 步的样本需由数据层另行处理。
        """

        actions = jnp.asarray(batch["actions"])
        rewards = jnp.asarray(batch["rewards"])
        masks   = jnp.asarray(batch["masks"])

        # 数据层可以提供 [B,H,A]，Critic 与 Actor 统一使用 [B,H*A]。
        if actions.ndim == 3:
            if actions.shape[1:] != (self.config["action_horizon"], self.config["action_dim"]):
                raise ValueError("Unflattened training actions must have shape [B, action_horizon, action_dim].")
            actions = actions.reshape((actions.shape[0], self.config["flow_dim"]))
        if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[-1] != self.config["flow_dim"]:
            raise ValueError(f"Training actions must have non-empty shape [B,{self.config['flow_dim']}], got {actions.shape}.")
        if not jnp.issubdtype(actions.dtype, jnp.floating):
            raise TypeError("Training actions must have a floating dtype.")
        batch_size = actions.shape[0]
        if rewards.shape != (batch_size,) or masks.shape != (batch_size,):
            raise ValueError("rewards and masks must have shape [B], one value per chunk transition.")
        if features.ndim != 2 or next_features.shape != features.shape or features.shape[0] != batch_size:
            raise ValueError("features and next_features must have matching shape [B, feature_dim].")

        # Critic 更新阶段只使用固定的观测特征，不将梯度传回 Encoder
        features      = jax.lax.stop_gradient(features)
        next_features = jax.lax.stop_gradient(next_features)

        # 每次更新使用新的高斯噪声生成下一 action chunk，并保存剩余 RNG 供下一阶段使用
        new_rng, noise_rng = jax.random.split(self.rng)
        next_noises = jax.random.normal(
            noise_rng,
            shape=actions.shape,
            dtype=next_features.dtype,
        )
        next_actions = self.one_step_actor(next_features, next_noises, train=False)

        # One-step Actor 的原始输出先投影到合法动作集合，再交给 Target Critic 计算 bootstrap value
        next_actions = project_action_chunks_to_unit_balls(
            next_actions,
            action_horizon=self.config["action_horizon"],
            action_dim=self.config["action_dim"],
        )
        next_actions = jax.lax.stop_gradient(next_actions)

        # Target Critic ensemble 默认取均值，也可以选择更保守的最小值
        next_qs = self.target_critic(next_features, next_actions, train=False)
        if self.config["q_aggregation"] == "min":
            next_q = jnp.min(next_qs, axis=0)
        else:
            next_q = jnp.mean(next_qs, axis=0)

        # 一个 chunk transition 跨越 action_horizon 个环境步，因此 bootstrap 使用 gamma**H
        chunk_discount = self.config["discount"] ** self.config["action_horizon"]
        target_q = rewards + chunk_discount * masks * next_q
        target_q = jax.lax.stop_gradient(target_q)

        def critic_loss_fn(critic_params: Params) -> tuple[jax.Array, dict[str, jax.Array]]:
            """基于 H 步 Bellman target 计算 Critic ensemble loss"""

            # Online Critic 的每个 Q head 都回归同一个 TD target
            q_values = self.critic.apply(
                critic_params,
                features,
                actions,
                train=True,
            )
            critic_loss = jnp.mean(jnp.square(q_values - target_q[None, :]))

            info = {
                "critic_loss": critic_loss,
                "q": jnp.mean(q_values),
                "q_min": jnp.min(q_values),
                "q_max": jnp.max(q_values),
                "target_q": jnp.mean(target_q),
            }

            return critic_loss, info

        # 反向传播只计算并更新 Online Critic 参数
        new_critic, info = self.critic.apply_loss_fn(critic_loss_fn)

        new_agent = self.replace(
            rng=new_rng,
            critic=new_critic,
        )

        return new_agent, info

    def _update_one_step_actor(self, features: jax.Array) -> tuple[FQLAGENT, dict[str, jax.Array]]:
        """使用蒸馏 loss 和 Q actor loss 更新 One-step Actor"""

        # One-step Actor 更新阶段只使用固定的观测特征，不将梯度传回 Encoder
        features = jax.lax.stop_gradient(features)
        if features.ndim not in (1, 2):
            raise ValueError(f"features must have shape [feature_dim] or [batch, feature_dim], got {features.shape}.")

        # 同一份高斯噪声同时提供给 Behavior Flow teacher 和 One-step Actor student
        new_rng, loss_rng = jax.random.split(self.rng)
        noise_rng, actor_dropout_rng = jax.random.split(loss_rng)
        noise_shape = (*features.shape[:-1], self.config["flow_dim"])
        noises = jax.random.normal(
            noise_rng,
            shape=noise_shape,
            dtype=features.dtype,
        )

        # Behavior Flow 通过多步 Euler ODE 生成蒸馏目标，并停止 teacher 方向的梯度
        target_flow_actions = FlowMatchingAgent.get_action_with_feat_and_noise_by_ode(
            self.behavior_flow,
            features,
            noises,
            num_ode_steps=self.config["num_ode_steps"],
            action_horizon=self.config["action_horizon"],
            action_dim=self.config["action_dim"],
        )
        target_flow_actions = jax.lax.stop_gradient(target_flow_actions)

        def actor_loss_fn(one_step_actor_params: Params) -> tuple[jax.Array, dict[str, jax.Array]]:
            """计算 One-step Actor 的蒸馏损失和 Q 引导损失"""

            # One-step Actor 从同一个噪声出发，一次前向直接预测最终 action chunk
            actor_actions = self.one_step_actor.apply(
                one_step_actor_params,
                features,
                noises,
                train=True,
                rngs={"dropout": actor_dropout_rng},
            )

            # 蒸馏损失约束 One-step Actor 接近 Behavior Flow 的动作分布
            distill_loss = jnp.mean(jnp.square(actor_actions - target_flow_actions))

            # Critic 只评估满足环境动作约束的 action chunk；投影运算仍允许梯度回传到 Actor
            projected_actor_actions = project_action_chunks_to_unit_balls(
                actor_actions,
                action_horizon=self.config["action_horizon"],
                action_dim=self.config["action_dim"],
            )

            # Critic 参数保持固定，但 Q 对 action 的梯度会继续回传到 One-step Actor
            q_values = self.critic(features, projected_actor_actions, train=False)
            if self.config["q_aggregation"] == "min":
                q = jnp.min(q_values, axis=0)
            else:
                q = jnp.mean(q_values, axis=0)
            q_actor_loss = -jnp.mean(q)

            # alpha 控制 Behavior Flow 蒸馏约束强度，Q actor loss 推动策略选择更高价值动作
            actor_loss = self.config["alpha"] * distill_loss + q_actor_loss

            info = {
                "actor_loss": actor_loss,
                "distill_loss": distill_loss,
                "q_actor_loss": q_actor_loss,
                "q": jnp.mean(q),
                "actor_action_norm": jnp.mean(jnp.linalg.norm(projected_actor_actions, axis=-1)),
                "teacher_action_norm": jnp.mean(jnp.linalg.norm(target_flow_actions, axis=-1)),
            }

            return actor_loss, info

        # 反向传播只计算并更新 One-step Actor 参数
        new_one_step_actor, info = self.one_step_actor.apply_loss_fn(actor_loss_fn)

        new_agent = self.replace(
            rng=new_rng,
            one_step_actor=new_one_step_actor,
        )

        return new_agent, info

    def _update_target_critic(self) -> FQLAGENT:
        """使用在线 Critic 参数对 Target Critic 进行 Polyak 软更新"""

        tau = self.config["tau"]
        new_target_params = jax.tree_util.tree_map(
            lambda critic_param, target_param: tau * critic_param + (1.0 - tau) * target_param,
            self.critic.params,
            self.target_critic.params,
        )
        # 只替换目标网络参数，保留其无优化器的状态。
        new_target_critic = self.target_critic.replace(params=new_target_params)
        return self.replace(target_critic=new_target_critic)


    #####################################################
    #                   创建更新流程                     #
    #####################################################

    @jax.jit
    def update(self, batch: flax.core.FrozenDict) -> tuple[FQLAGENT, dict[str, jax.Array]]:
        """依次执行 Behavior Flow、Critic、One-step Actor 和 Target Critic 更新"""

        """------------------------- 更新 Encoder 和 Behavior Flow -------------------------"""

        agent, behavior_flow_info = self._update_behavior_flow(batch)

        """------------------------- 共享特征计算 -------------------------"""

        features      = agent.encoder(batch["observations"], train=False)
        next_features = agent.encoder(batch["next_observations"], train=False)
        features      = jax.lax.stop_gradient(features)
        next_features = jax.lax.stop_gradient(next_features)

        """------------------------- 更新 critic, actor 和 target critic -------------------------"""

        # Critic 只更新 Online Q ensemble，随后 One-step Actor 使用更新后的 Critic 提供 Q 引导
        agent, critic_info = agent._update_critic(
            batch,
            features,
            next_features,
        )
        agent, actor_info = agent._update_one_step_actor(features)

        # 每次 Online Critic 更新后，对 Target Critic 执行一次 Polyak 软更新
        agent = agent._update_target_critic()

        # 使用前缀区分不同模块的训练指标，避免同名字段互相覆盖
        info = {
            **{f"behavior_flow/{key}": value for key, value in behavior_flow_info.items()},
            **{f"critic/{key}": value for key, value in critic_info.items()},
            **{f"actor/{key}": value for key, value in actor_info.items()},
        }

        return agent, info

    @jax.jit
    def sample_actions(self, observations: dict[str, jax.Array], rng: PRNGKey) -> jax.Array:
        """使用 One-step Actor 一次生成展平的 action chunk"""

        # 观测只编码一次，One-step Actor 不需要时间输入或 ODE 数值积分
        features = self.encoder(observations, train=False)
        if features.ndim not in (1, 2):
            raise ValueError(f"The encoder must return [feature_dim] or [batch, feature_dim], got {features.shape}.")

        leading_shape = features.shape[:-1]
        noise_shape = (*leading_shape, self.config["flow_dim"])

        # 从 Flow Matching 源分布 N(0, I) 采样 One-step Actor 的输入噪声
        noises = jax.random.normal(
            rng,
            shape=noise_shape,
            dtype=features.dtype,
        )
        actions = self.one_step_actor(features, noises, train=False)

        # 网络输出保持展平 action chunk 形状，只在最终动作上施加环境动作约束
        return project_action_chunks_to_unit_balls(
            actions,
            action_horizon=self.config["action_horizon"],
            action_dim=self.config["action_dim"],
        )
