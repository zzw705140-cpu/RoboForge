from dataclasses import dataclass
from typing import Iterable, Mapping

import flax.linen as nn
import jax
import jax.numpy as jnp

##############################################################################
#               Image and Proprioception Encoder Network                     #
##############################################################################


@dataclass(frozen=True)
class _ObservationLayout:
    """规范化后的 observation 批量与时间布局"""

    input_was_batched: bool
    batch_size: int
    observation_horizon: int


def _select_visual_namespace(observations: Mapping[str, jax.Array]) -> tuple[str, Mapping[str, jax.Array], bool]:
    """选择原始图像或预计算视觉特征，并返回是否需要执行 Backbone"""

    has_images = "images" in observations
    has_visual_features = "visual_features" in observations

    if has_images == has_visual_features:
        raise ValueError("EncoderNet requires exactly one visual namespace: 'images' or 'visual_features'.")

    namespace = "images" if has_images else "visual_features"

    return namespace, observations[namespace], has_images


def _normalize_to_batched_temporal(
    value: jax.Array,
    *,
    feature_rank: int,
    use_temporal_history: bool,
    name: str,
) -> tuple[jax.Array, _ObservationLayout]:
    """
    将单样本/批量、单帧/历史窗口统一为 [B,T,...]

    Args:
        feature_rank: 表示单个时刻数据本身的维数
    
            * 视觉为 [H,W,C], 因此是 3
            * 状态为 [D],     因此是 1
    """

    value = jnp.asarray(value)

    if use_temporal_history:
        if value.ndim == feature_rank + 1:
            # [T,...] → [1,T,...]
            normalized = value[None, ...]
            input_was_batched = False
        elif value.ndim == feature_rank + 2:
            # [B,T,...] 保持不变
            normalized = value
            input_was_batched = True
        else:
            raise ValueError(f"Expected temporal {name} with {feature_rank + 1} or {feature_rank + 2} dimensions, got shape {value.shape}.")
    else:
        if value.ndim == feature_rank:
            # [...] → [1,1,...]
            normalized = value[None, None, ...]
            input_was_batched = False
        elif value.ndim == feature_rank + 1:
            # [B,...] → [B,1,...]
            normalized = value[:, None, ...]
            input_was_batched = True
        else:
            raise ValueError(f"Expected non-temporal {name} with {feature_rank} or {feature_rank + 1} dimensions, got shape {value.shape}.")

    layout = _ObservationLayout(
        input_was_batched=input_was_batched,
        batch_size=normalized.shape[0],
        observation_horizon=normalized.shape[1],
    )
    return normalized, layout


def _validate_layout(
    layout: _ObservationLayout,
    expected_layout: _ObservationLayout,
    *,
    name: str,
) -> None:
    """检查某个 observation 字段是否与第一个相机的 B、T 布局一致"""

    if layout != expected_layout:
        raise ValueError(f"{name} must use layout {expected_layout}, got {layout}.")


def _prepare_task_condition_leaf(
    value: jax.Array,
    *,
    layout: _ObservationLayout,
    use_temporal_history: bool,
) -> jax.Array:
    """验证静态任务条件的布局，取最后时刻并统一输出 [B,D]"""

    value = jnp.asarray(value)
    batch_size = layout.batch_size
    observation_horizon = layout.observation_horizon

    if use_temporal_history:
        if layout.input_was_batched:
            if (value.ndim < 2 or value.shape[:2] != (batch_size, observation_horizon)):
                raise ValueError(f"Every batched task_condition leaf must have leading shape [B,T]=[{batch_size},{observation_horizon}], got {value.shape}.")
            value = value[:, -1, ...]
        else:
            if value.ndim < 1 or value.shape[0] != observation_horizon:
                raise ValueError(f"Every unbatched task_condition leaf must have leading shape [T]=[{observation_horizon}], got {value.shape}.")
            value = value[-1, ...][None, ...]
    elif layout.input_was_batched:
        if value.ndim < 1 or value.shape[0] != batch_size:
            raise ValueError(f"Every batched task_condition leaf must have leading shape [B]=[{batch_size}], got {value.shape}.")
    else:
        value = value[None, ...]

    return value.reshape((batch_size, -1))


class EncoderNet(nn.Module):
    """
    将 “多帧视觉输入”、“proprio observation” 和 “任务条件” 编码成统一特征向量

    ========================================================================

    视觉输入支持两个互斥的命名空间 (只能选择其一):

    - images:          原始图像，图像 Encoder 会先执行冻结 Backbone
    - visual_features: 预计算的 pre-pooling Backbone 特征，图像 Encoder 跳过 Backbone, 直接执行可训练的 pooling 和 bottleneck

    ========================================================================

    Encode 输入支持历史维度:

    - 格式 1: use_temporal_history = true

        推理阶段 (单个历史窗口):
            {
                "images": {
                    "front": [T, H, W, C],
                    "wrist": [T, H, W, C],
                },
                "state": [T, state_dim],
                "task_condition": {
                    "task_one_hot": [T, task_dim],
                    "goal":         [T, goal_dim],
                },
            }
        训练阶段（批量历史窗口）：
            {
                "images": {
                    "front": [B, T, H, W, C],
                    "wrist": [B, T, H, W, C],
                },
                "state": [B, T, state_dim],
                "task_condition": {
                    "task_one_hot": [B, T, task_dim],
                    "goal":         [B, T, goal_dim],
                },
            }

        使用预计算特征时，只需将 images 替换为：

            {
                "visual_features": {
                    "front": [B, T, Hf, Wf, Cf],
                    "wrist": [B, T, Hf, Wf, Cf],
                },
                ...
            }

        当前 ResNet-10 与 128 x 128 图像对应 [Hf,Wf,Cf] = [4,4,512]

    - 格式 2: use_temporal_history = false

        推理阶段 (单个历史窗口):
            {
                "images": {
                    "front": [H, W, C],
                    "wrist": [H, W, C],
                },
                "state": [state_dim],
                "task_condition": {
                    "task_one_hot": [task_dim],
                    "goal":         [goal_dim],
                },
            }
        训练阶段（批量历史窗口）：
            {
                "images": {
                    "front": [B, H, W, C],
                    "wrist": [B, H, W, C],
                },
                "state": [B, state_dim],
                "task_condition": {
                    "task_one_hot": [B, task_dim],
                    "goal":         [B, goal_dim],
                },
            }

    ========================================================================

    【images / visual_features】

    每个相机拥有独立的图像 Encoder, 但不同历史时刻共享同一组网络参数

    * images 先经过 ResNet
    * visual_features 则跳过 ResNet

    两条路径共享同一组可训练 pooling/bottleneck, 然后展平时间维:

    单样本: [T,H,W,C]   → ResNet → [T,image_dim]   → [T*image_dim]
    批量：  [B,T,H,W,C] → ResNet → [B,T,image_dim] → [B,T*image_dim]

    单样本: [T,Hf,Wf,Cf]   → Head → [T,image_dim]   → [T*image_dim]
    批量：  [B,T,Hf,Wf,Cf] → Head → [B,T,image_dim] → [B,T*image_dim]

    多个相机的编码结果最终沿特征维拼接

    ========================================================================

    【state】

    每一帧展平后的机器人本体状态共享同一个 proprio projection, 随后展平时间维:

    单样本: [T,state_dim]   → Projection → [T,proprio_dim]   → [T*proprio_dim]
    批量：  [B,T,state_dim] → Projection → [B,T,proprio_dim] → [B,T*proprio_dim]

    ========================================================================

    【task_condition】

    task_condition 是由一个或多个数组叶子组成的 PyTree, 表示当前阶段内保持不变的任务目标。Encoder 递归取得所有叶子。启用历史观测时，每个叶子只取最后一个时刻，再展平其余特征维，并将所有叶子沿特征维拼接：

    单样本: [T,...]   → 取最后时刻 [...]   → 展平 [D]
    批量：  [B,T,...] → 取最后时刻 [B,...] → 展平 [B,D]

    拼接后的原始条件再经过独立的可学习投影：

        raw task condition → Dense → LayerNorm → SiLU → task embedding

    这样任务条件以稳定的特征尺度进入后续 Flow Matching FiLM 网络，避免低维 one-hot 在高维视觉特征中被弱化。

    ========================================================================

    最终融合方式：
        proprio_features ───┐
        image_features ─────┼─ concatenate → features
        task_embedding ─────┘

    当 pooling_method="task_conditioned_attention" 时，task_embedding 还会进入
    每个相机、每个历史时刻的视觉空间池化；最终拼接仍保留 task_embedding。

    ========================================================================

    输出格式：
        推理阶段：[feature_dim]
        训练阶段：[B, feature_dim]
    """

    image_encoders: Mapping[str, nn.Module]         # 每个相机对应一个图像 Encoder
    use_proprio: bool                               # 是否编码并拼接机器人本体状态

    proprio_latent_dim: int = 64                    # 每一帧 proprio 投影后的维度
    task_condition_latent_dim: int = 64             # 整个 task condition 投影后的维度
    use_temporal_history: bool = True
    image_keys: Iterable[str] = ("image",)          # 需要编码的相机字段
    pooling_method: str = "spatial_learned_embeddings"

    def encode_backbone(self, images: Mapping[str, jax.Array]) -> dict[str, jax.Array]:
        """
        对所有相机只执行冻结 Backbone，并保持输入中的批量与时间前导维

        支持的单相机输入与输出布局包括：

        - [H,W,C]     → [Hf,Wf,Cf]
        - [T,H,W,C]   → [T,Hf,Wf,Cf]
        - [B,T,H,W,C] → [B,T,Hf,Wf,Cf]

        该接口供离线特征缓存和在线 Actor 共用；它不会执行可训练的 pooling/bottleneck，也不会编码 state 或 task_condition
        """

        if not isinstance(images, Mapping):
            raise TypeError("images must be a mapping from camera names to arrays.")

        visual_features: dict[str, jax.Array] = {}
        expected_leading_shape: tuple[int, ...] | None = None
        for image_key in self.image_keys:
            if image_key not in images:
                raise KeyError(f"Images have no camera key {image_key!r}.")

            image = jnp.asarray(images[image_key])
            if image.ndim < 3:
                raise ValueError(f"Camera {image_key!r} must end with [H,W,C], got shape {image.shape}.")
            leading_shape = tuple(image.shape[:-3])
            if expected_leading_shape is None:
                expected_leading_shape = leading_shape
            elif leading_shape != expected_leading_shape:
                raise ValueError("All camera images must use identical batch/time leading dimensions.")

            image_batch = image.reshape((-1, *image.shape[-3:]))
            image_encoder = self.image_encoders[image_key]
            if not hasattr(image_encoder, "encode_backbone"):
                raise TypeError(f"Image encoder {image_key!r} does not expose encode_backbone().")
            feature_batch = image_encoder.encode_backbone(image_batch)
            if feature_batch.ndim != 4 or feature_batch.shape[0] != image_batch.shape[0]:
                raise ValueError(f"Backbone for camera {image_key!r} must return [N,Hf,Wf,Cf], got {feature_batch.shape}.")
            visual_features[image_key] = feature_batch.reshape((*leading_shape, *feature_batch.shape[-3:]))

        if not visual_features:
            raise ValueError("EncoderNet requires at least one image key.")
        return visual_features

    @nn.compact
    def __call__(self, observations: Mapping[str, jax.Array], train: bool = False) -> jax.Array:
        """编码 observation, 并在特征层融合视觉、state 和任务条件"""

        visual_namespace, visual_observations, encode_backbone = _select_visual_namespace(observations)
        image_keys = tuple(self.image_keys)
        if not image_keys:
            raise ValueError("EncoderNet requires at least one image key.")

        first_image_key = image_keys[0]
        if first_image_key not in visual_observations:
            raise KeyError(f"Observation {visual_namespace!r} has no camera key {first_image_key!r}.")
        _, layout = _normalize_to_batched_temporal(
            visual_observations[first_image_key],
            feature_rank=3,
            use_temporal_history=self.use_temporal_history,
            name=f"{visual_namespace} camera {first_image_key!r}",
        )

        # ------------------------------------- 任务条件编码 ------------------------------------- #

        # 先编码静态任务条件，供条件视觉池化和最终特征融合共同使用
        task_features = None
        if "task_condition" in observations:
            if self.task_condition_latent_dim <= 0:
                raise ValueError("task_condition_latent_dim must be positive.")
            task_condition_leaves = jax.tree_util.tree_leaves(observations["task_condition"])
            if not task_condition_leaves:
                raise ValueError("Observation 'task_condition' must contain at least one array leaf.")
            raw_task_condition = jnp.concatenate(
                [
                    _prepare_task_condition_leaf(
                        condition,
                        layout=layout,
                        use_temporal_history=self.use_temporal_history,
                    ).astype(jnp.float32)
                    for condition in task_condition_leaves
                ],
                axis=-1,
            )
            task_features = nn.Dense(
                features=self.task_condition_latent_dim,
                use_bias=False,
                kernel_init=nn.initializers.xavier_uniform(),
                name="task_condition_projection",
            )(raw_task_condition)
            task_features = nn.LayerNorm(name="task_condition_layer_norm")(task_features)
            task_features = nn.silu(task_features)

        # ------------------------------------- 本体状态编码 ------------------------------------- #

        # 每帧状态既用于注意力条件，也照常进入最终策略特征。
        state_features = None
        if self.use_proprio:
            if "state" not in observations:
                raise KeyError("use_proprio=True, but observations has no 'state' key.")

            state, state_layout = _normalize_to_batched_temporal(
                observations["state"],
                feature_rank=1,
                use_temporal_history=self.use_temporal_history,
                name="proprio state",
            )
            _validate_layout(state_layout, layout, name="proprio state")
            state_batch = state.reshape((-1, state.shape[-1]))
            state_features = nn.Dense(
                features=self.proprio_latent_dim,
                kernel_init=nn.initializers.xavier_uniform(),
                name="proprio_projection",
            )(state_batch)
            state_features = nn.LayerNorm(name="proprio_layer_norm")(state_features)
            state_features = nn.tanh(state_features)

        # ------------------------------------- 输入图像编码 ------------------------------------- #

        if self.pooling_method == "task_conditioned_attention" and task_features is None:
            raise ValueError("pooling_method='task_conditioned_attention' requires task_condition.")

        temporal_attention_features = None
        if self.pooling_method == "task_conditioned_attention":
            temporal_attention_features = jnp.broadcast_to(
                task_features[:, None, :],
                (layout.batch_size, layout.observation_horizon, task_features.shape[-1]),
            ).reshape((-1, task_features.shape[-1]))
            if state_features is not None:
                temporal_attention_features = jnp.concatenate([temporal_attention_features, state_features], axis=-1)

        # 每个相机的历史帧分别池化，然后展平时间维
        encoded_visual_features = []
        for image_key in image_keys:
            if image_key not in visual_observations:
                raise KeyError(f"Observation {visual_namespace!r} has no camera key {image_key!r}.")

            visual_input, current_layout = _normalize_to_batched_temporal(
                visual_observations[image_key],
                feature_rank=3,
                use_temporal_history=self.use_temporal_history,
                name=f"{visual_namespace} camera {image_key!r}",
            )
            _validate_layout(current_layout, layout, name=f"{visual_namespace} camera {image_key!r}")

            visual_batch = visual_input.reshape((-1, *visual_input.shape[-3:]))
            encoder_kwargs = {"encode": encode_backbone, "train": train}
            if temporal_attention_features is not None:
                encoder_kwargs["task_features"] = temporal_attention_features
            image_features = self.image_encoders[image_key](visual_batch, **encoder_kwargs)

            if image_features.ndim != 2:
                raise ValueError(f"Image encoder {image_key!r} must return [N,D], got {image_features.shape}.")
            image_features = image_features.reshape(layout.batch_size, layout.observation_horizon, -1)
            encoded_visual_features.append(image_features.reshape(layout.batch_size, -1))

        image_features = jnp.concatenate(encoded_visual_features, axis=-1)
        feature_parts = []

        if state_features is not None:
            feature_parts.append(state_features.reshape(layout.batch_size, -1))

        # ------------------------------------- 编码合并 ------------------------------------- #

        feature_parts.append(image_features)
        if task_features is not None:
            feature_parts.append(task_features.astype(image_features.dtype))
        features = jnp.concatenate(feature_parts, axis=-1)

        # 推理输入在规范化时临时增加了 B=1，这里恢复原有接口
        if not layout.input_was_batched:
            return features[0]
        return features
