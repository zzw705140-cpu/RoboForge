"""ACT 的训练、评估和推理接口。"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

import flax.struct as struct
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax

from roboforg.agent.act.act_policy import ACTPolicy
from roboforg.agent.act.data_normalizer import DataNormalizer
from roboforg.agent.act.latent_encoder import LatentEncoder, kl_loss


# 从 ACT 格式 batch 中取出 state、双相机图像和动作，并做基本形状检查。
def _unpack_batch(
    batch: Mapping[str, Any] | tuple[Mapping[str, Any], Any], *, camera_keys: tuple[str, ...], action_horizon: int, action_dim: int
) -> tuple[jax.Array, dict[str, jax.Array], jax.Array]:
    """Extract raw state/images/actions from either supported converted ACT batch format."""
    # 当前 ACTBatchConverter 返回 (observations, actions)；同时兼容未来 workflow 可能使用的字典格式。
    if isinstance(batch, tuple):
        if len(batch) != 2 or not isinstance(batch[0], Mapping):
            raise TypeError("Tuple ACT batch must be (observations, actions).")
        observations, actions = batch
    elif isinstance(batch, Mapping):
        if "observations" not in batch or "actions" not in batch:
            raise KeyError("Mapping ACT batch must contain observations and actions.")
        observations, actions = batch["observations"], batch["actions"]
        if not isinstance(observations, Mapping):
            raise TypeError("Mapping ACT batch field 'observations' must be a mapping.")
    else:
        raise TypeError("ACT batch must be (observations, actions) or a mapping with those fields.")
    if "state" not in observations:
        raise KeyError("ACT batch observations must contain state.")

    state = jnp.asarray(observations["state"], dtype=jnp.float32)
    actions = jnp.asarray(actions, dtype=jnp.float32)
    if state.ndim != 2 or state.shape[0] == 0 or state.shape[1] == 0:
        raise ValueError("batch observations['state'] must have non-empty shape [B,S].")
    if actions.shape != (state.shape[0], action_horizon, action_dim):
        raise ValueError(
            "batch actions must have shape "
            f"[{state.shape[0]},{action_horizon},{action_dim}], got {actions.shape}."
        )

    images: dict[str, jax.Array] = {}
    for key in camera_keys:
        if key not in observations:
            raise KeyError(f"ACT batch observations is missing camera {key!r}.")
        image = jnp.asarray(observations[key])
        if image.ndim != 4 or image.shape[0] != state.shape[0] or image.shape[-1] != 3:
            raise ValueError(f"Camera {key!r} must have shape [B,H,W,3] matching state batch size.")
        images[key] = image
    return state, images, actions


# 为 policy 参数创建两组优化器标签：ResNet 主干可使用比其余网络更小的学习率。
def _make_policy_optimizer(
    params: Any, *, learning_rate: float, backbone_learning_rate: float, weight_decay: float
) -> optax.GradientTransformation:
    """Build AdamW with a separate learning rate for the policy backbone subtree."""
    if learning_rate <= 0.0 or backbone_learning_rate <= 0.0:
        raise ValueError("learning_rate and backbone_learning_rate must be positive.")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative.")

    # ACTPolicy 将外部 ResNet 注册为顶层 backbone；其余 token/Transformer/action 参数都归 main。
    if "backbone" not in params:
        return optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    labels = {
        name: jax.tree_util.tree_map(
            lambda _: "backbone" if name == "backbone" else "main", subtree
        )
        for name, subtree in params.items()
    }
    return optax.multi_transform(
        {
            "main": optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
            "backbone": optax.adamw(
                learning_rate=backbone_learning_rate, weight_decay=weight_decay
            ),
        },
        labels,
    )


# 训练期前向、损失与两个参数树的梯度更新；Module 是静态网络定义，参数保持为 JAX 数组。
@partial(jax.jit, static_argnames=("policy", "latent_encoder", "kl_weight"))
def _update_states(
    policy_state: TrainState,
    latent_state: TrainState,
    state: jax.Array,
    images: Mapping[str, jax.Array],
    actions: jax.Array,
    rng: jax.Array,
    *,
    policy: ACTPolicy,
    latent_encoder: LatentEncoder,
    kl_weight: float,
) -> tuple[TrainState, TrainState, dict[str, jax.Array]]:
    """Differentiate BC+KL loss and return updated TrainStates plus scalar metrics."""
    latent_sample_key, latent_dropout_key, policy_dropout_key = jax.random.split(rng, 3)

    def loss_fn(policy_params: Any, latent_params: Any) -> tuple[jax.Array, dict[str, jax.Array]]:
        # 训练时 CVAE 使用真实动作推断 z；两个网络各自获得独立 Dropout 随机数。
        latent_output = latent_encoder.apply(
            {"params": latent_params},
            state,
            actions,
            rng=latent_sample_key,
            train=True,
            rngs={"dropout": latent_dropout_key},
        )
        predicted_actions = policy.apply(
            {"params": policy_params},
            state,
            images,
            latent_output.z,
            train=True,
            rngs={"dropout": policy_dropout_key},
        )

        # BC 使用标准化动作的逐元素 L1；KL 的归约方式在 latent_encoder.py 中与原 ACT 对齐。
        bc = jnp.mean(jnp.abs(predicted_actions - actions))
        kl = kl_loss(latent_output.mu, latent_output.logvar)
        total = bc + kl_weight * kl
        return total, {"loss": total, "bc_loss": bc, "kl_loss": kl}

    (_, metrics), (policy_grads, latent_grads) = jax.value_and_grad(
        loss_fn, argnums=(0, 1), has_aux=True
    )(policy_state.params, latent_state.params)
    return (
        policy_state.apply_gradients(grads=policy_grads),
        latent_state.apply_gradients(grads=latent_grads),
        metrics,
    )


# 验证期只计算 BC/KL，不使用 Dropout，也不会修改参数。
@partial(jax.jit, static_argnames=("policy", "latent_encoder", "kl_weight"))
def _evaluate_states(
    policy_state: TrainState,
    latent_state: TrainState,
    state: jax.Array,
    images: Mapping[str, jax.Array],
    actions: jax.Array,
    rng: jax.Array,
    *,
    policy: ACTPolicy,
    latent_encoder: LatentEncoder,
    kl_weight: float,
) -> dict[str, jax.Array]:
    """Evaluate one labeled batch without updating any parameters."""
    latent_output = latent_encoder.apply(
        {"params": latent_state.params}, state, actions, rng=rng, train=False
    )
    predicted_actions = policy.apply(
        {"params": policy_state.params}, state, images, latent_output.z, train=False
    )
    bc = jnp.mean(jnp.abs(predicted_actions - actions))
    kl = kl_loss(latent_output.mu, latent_output.logvar)
    return {"loss": bc + kl_weight * kl, "bc_loss": bc, "kl_loss": kl}


# ACT 算法对象：持有两个网络的 TrainState、固定数据统计量和训练超参数。
@struct.dataclass
class ACTAgent:
    """Functional ACT trainer; update returns a new agent rather than mutating this one."""

    policy_state: TrainState
    latent_state: TrainState
    policy: ACTPolicy = struct.field(pytree_node=False)
    latent_encoder: LatentEncoder = struct.field(pytree_node=False)
    normalizer: DataNormalizer = struct.field(pytree_node=False)
    kl_weight: float = struct.field(pytree_node=False)

    # 用一个真实格式的示例 batch 初始化两套网络参数与 AdamW 优化器。
    @classmethod
    def create(
        cls,
        *,
        policy: ACTPolicy,
        latent_encoder: LatentEncoder,
        normalizer: DataNormalizer,
        example_batch: Mapping[str, Any] | tuple[Mapping[str, Any], Any],
        rng: jax.Array,
        learning_rate: float,
        backbone_learning_rate: float | None = None,
        weight_decay: float = 0.0,
        kl_weight: float = 10.0,
    ) -> "ACTAgent":
        """Initialize ACT parameters from one batch and configure its optimizers."""
        if kl_weight < 0.0:
            raise ValueError("kl_weight must be non-negative.")
        backbone_learning_rate = learning_rate if backbone_learning_rate is None else backbone_learning_rate
        state, images, actions = _unpack_batch(
            example_batch,
            camera_keys=policy.camera_keys,
            action_horizon=policy.action_horizon,
            action_dim=policy.action_dim,
        )
        normalized_state = normalizer.normalize_state(state)
        normalized_actions = normalizer.normalize_action(actions)
        latent_init_key, latent_dropout_key, latent_sample_key, policy_init_key, policy_dropout_key = (
            jax.random.split(rng, 5)
        )

        # 后验网络在初始化时走完整训练路径，确保 Transformer、两个参数头和 Dropout 都被创建。
        latent_variables = latent_encoder.init(
            {"params": latent_init_key, "dropout": latent_dropout_key},
            normalized_state,
            normalized_actions,
            rng=latent_sample_key,
            train=True,
        )
        zero_z = jnp.zeros((state.shape[0], latent_encoder.latent_dim), dtype=jnp.float32)
        policy_variables = policy.init(
            {"params": policy_init_key, "dropout": policy_dropout_key},
            normalized_state,
            images,
            zero_z,
            train=True,
        )

        # policy 的 ResNet 可单独使用较小学习率；CVAE 后验网络使用主学习率。
        policy_tx = _make_policy_optimizer(
            policy_variables["params"],
            learning_rate=learning_rate,
            backbone_learning_rate=backbone_learning_rate,
            weight_decay=weight_decay,
        )
        latent_tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
        return cls(
            policy_state=TrainState.create(
                apply_fn=policy.apply, params=policy_variables["params"], tx=policy_tx
            ),
            latent_state=TrainState.create(
                apply_fn=latent_encoder.apply, params=latent_variables["params"], tx=latent_tx
            ),
            policy=policy,
            latent_encoder=latent_encoder,
            normalizer=normalizer,
            kl_weight=kl_weight,
        )

    # 执行一次训练更新，返回新参数状态、未修改旧 agent，以及三项 loss。
    def update(
        self, batch: Mapping[str, Any] | tuple[Mapping[str, Any], Any], rng: jax.Array
    ) -> tuple["ACTAgent", dict[str, jax.Array]]:
        """Run one BC+KL gradient update using a raw converted ACT batch."""
        state, images, actions = _unpack_batch(
            batch,
            camera_keys=self.policy.camera_keys,
            action_horizon=self.policy.action_horizon,
            action_dim=self.policy.action_dim,
        )
        policy_state, latent_state, metrics = _update_states(
            self.policy_state,
            self.latent_state,
            self.normalizer.normalize_state(state),
            images,
            self.normalizer.normalize_action(actions),
            rng,
            policy=self.policy,
            latent_encoder=self.latent_encoder,
            kl_weight=self.kl_weight,
        )
        return self.replace(policy_state=policy_state, latent_state=latent_state), metrics

    # 在带真实动作标签的数据上评估 BC/KL，不更新网络，用于 eval split 和 checkpoint 选择。
    def evaluate(
        self, batch: Mapping[str, Any] | tuple[Mapping[str, Any], Any], rng: jax.Array
    ) -> dict[str, jax.Array]:
        """Compute labeled evaluation metrics without any parameter update."""
        state, images, actions = _unpack_batch(
            batch,
            camera_keys=self.policy.camera_keys,
            action_horizon=self.policy.action_horizon,
            action_dim=self.policy.action_dim,
        )
        return _evaluate_states(
            self.policy_state,
            self.latent_state,
            self.normalizer.normalize_state(state),
            images,
            self.normalizer.normalize_action(actions),
            rng,
            policy=self.policy,
            latent_encoder=self.latent_encoder,
            kl_weight=self.kl_weight,
        )

    # 推理不使用真实动作和后验编码器，固定 z=0 并把输出恢复为环境动作尺度。
    def predict(self, observations: Mapping[str, Any]) -> jax.Array:
        """Predict one raw-scale action chunk from batched observations using z = 0."""
        if "state" not in observations:
            raise KeyError("observations must contain state.")
        state = jnp.asarray(observations["state"], dtype=jnp.float32)
        if state.ndim != 2 or state.shape[0] == 0:
            raise ValueError("observations['state'] must have non-empty shape [B,S].")
        images = {key: jnp.asarray(observations[key]) for key in self.policy.camera_keys if key in observations}
        if len(images) != len(self.policy.camera_keys):
            missing = [key for key in self.policy.camera_keys if key not in images]
            raise KeyError(f"observations is missing cameras: {missing}.")

        # z=0 是标准正态先验均值；推理输出恢复到 RoboForge 的原始七维动作尺度。
        zero_z = jnp.zeros((state.shape[0], self.latent_encoder.latent_dim), dtype=jnp.float32)
        normalized_actions = self.policy.apply(
            {"params": self.policy_state.params},
            self.normalizer.normalize_state(state),
            images,
            zero_z,
            train=False,
        )
        return self.normalizer.denormalize_action(normalized_actions)
