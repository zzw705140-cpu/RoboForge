"""Spatial pooling and task-conditioned attention for visual feature maps."""

from collections.abc import Mapping
from typing import Any, Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp


# ------------------------------------------------------------------------- #
#                         空间 Softmax 池化                                  #
# ------------------------------------------------------------------------- #

class SpatialSoftmax(nn.Module):
    height: int
    width: int
    channel: int
    pos_x: jnp.ndarray
    pos_y: jnp.ndarray
    temperature: None
    log_heatmap: bool = False

    @nn.compact
    def __call__(self, features):
        """对空间位置执行 Softmax, 输出每个通道的期望坐标."""

        if self.temperature == -1:
            from jax.nn import initializers

            temperature = self.param("softmax_temperature", initializers.ones, (1), jnp.float32)
        else:
            temperature = 1.0

        # add batch dim if missing
        no_batch_dim = len(features.shape) < 4
        if no_batch_dim:
            features = features[None]

        assert len(features.shape) == 4
        batch_size, num_featuremaps = features.shape[0], features.shape[3]
        features = features.transpose(0, 3, 1, 2).reshape(batch_size, num_featuremaps, self.height * self.width)

        softmax_attention = nn.softmax(features / temperature)
        expected_x = jnp.sum(self.pos_x * softmax_attention, axis=2, keepdims=True).reshape(batch_size, num_featuremaps)
        expected_y = jnp.sum(self.pos_y * softmax_attention, axis=2, keepdims=True).reshape(batch_size, num_featuremaps)
        expected_xy = jnp.concatenate([expected_x, expected_y], axis=1)

        expected_xy = jnp.reshape(expected_xy, [batch_size, 2 * num_featuremaps])

        if no_batch_dim:
            expected_xy = expected_xy[0]
        return expected_xy


# ------------------------------------------------------------------------- #
#                   可学习与任务条件化空间池化                                  #
# ------------------------------------------------------------------------- #

class SpatialLearnedEmbeddings(nn.Module):
    height: int
    width: int
    channel: int
    num_features: int = 5
    kernel_init: Callable = nn.initializers.lecun_normal()
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, features):
        """
        features is B x H x W X C
        """
        kernel = self.param(
            "kernel",
            self.kernel_init,
            (self.height, self.width, self.channel, self.num_features),
            self.param_dtype,
        )

        # add batch dim if missing
        no_batch_dim = len(features.shape) < 4
        if no_batch_dim:
            features = features[None]

        batch_size = features.shape[0]
        assert len(features.shape) == 4
        features = jnp.sum(jnp.expand_dims(features, -1) * jnp.expand_dims(kernel, 0), axis=(1, 2))
        features = jnp.reshape(features, [batch_size, -1])

        if no_batch_dim:
            features = features[0]

        return features


class TaskConditionedSpatialAttention(nn.Module):
    """将 [B,H,W,C] 视觉特征按 [B,D] 任务特征池化为 [B,C]"""

    @nn.compact
    def __call__(
        self,
        visual_features: jax.Array,
        task_features: jax.Array,
    ) -> jax.Array:
        """根据任务特征计算空间注意力, 汇聚视觉特征."""

        if visual_features.ndim != 4:
            raise ValueError(f"visual_features must have shape [B,H,W,C], got {visual_features.shape}.")
        if task_features.ndim != 2:
            raise ValueError(f"task_features must have shape [B,D], got {task_features.shape}.")
        if visual_features.shape[0] != task_features.shape[0]:
            raise ValueError(f"visual_features and task_features must use the same batch size, got {visual_features.shape[0]} and {task_features.shape[0]}.")

        channels = visual_features.shape[-1]
        if channels <= 0:
            raise ValueError("visual_features must have a non-empty channel dimension.")

        query = nn.Dense(
            features=channels,
            use_bias=False,
            kernel_init=nn.initializers.xavier_uniform(),
            name="task_query",
        )(task_features)

        # 统一 query 和各空间位置的特征尺度, 避免 softmax 过早饱和
        query = nn.LayerNorm(name="query_norm")(query)
        keys  = nn.LayerNorm(name="visual_key_norm")(visual_features)

        # scaled dot-product attention
        logits = jnp.einsum("bhwc,bc->bhw", keys, query) / jnp.sqrt(
            jnp.asarray(channels, dtype=visual_features.dtype)
        )
        batch_size, height, width = logits.shape
        attention = nn.softmax(logits.reshape(batch_size, height * width), axis=-1)
        attention = attention.reshape(batch_size, height, width)
        self.sow("intermediates", "attention_map", attention)

        # 使用原始视觉特征作为 value, 按空间注意力加权得到 [B,C]
        return jnp.einsum("bhw,bhwc->bc", attention, visual_features)


# ------------------------------------------------------------------------- #
#                     空间池化与瓶颈统一入口                                   #
# ------------------------------------------------------------------------- #

def _apply_spatial_pooling_and_bottleneck(
    x: jax.Array | Mapping[int, jax.Array],
    *,
    pooling_method: str,
    num_spatial_blocks: int,
    softmax_temperature: float,
    bottleneck_dim: Optional[int],
    train: bool,
    task_features: Optional[jax.Array],
) -> jax.Array:
    """对 ResNet 特征图执行空间汇聚和可选的瓶颈投影"""

    # ----------------------------------- spatial_learned_embeddings ----------------------------------- #
    if pooling_method == "spatial_learned_embeddings":
        height, width, channel = x.shape[-3:]
        x = SpatialLearnedEmbeddings(
            height=height,
            width=width,
            channel=channel,
            num_features=num_spatial_blocks,
        )(x)
        x = nn.Dropout(0.1, deterministic=not train)(x)

    # ----------------------------------- spatial_softmax ----------------------------------- #
    elif pooling_method == "spatial_softmax":
        height, width, channel = x.shape[-3:]
        pos_x, pos_y = jnp.meshgrid(
            jnp.linspace(-1.0, 1.0, height),
            jnp.linspace(-1.0, 1.0, width),
        )
        pos_x = pos_x.reshape(height * width)
        pos_y = pos_y.reshape(height * width)
        x = SpatialSoftmax(
            height,
            width,
            channel,
            pos_x,
            pos_y,
            softmax_temperature,
        )(x)

    # ----------------------------------- task_conditioned_attention ----------------------------------- #

    # 旧版 task_conditioned_attention: 恢复旧 checkpoint 时启用此分支并停用下方新版.
    elif pooling_method == "task_conditioned_attention":
        if task_features is None:
            raise ValueError("pooling_method='task_conditioned_attention' requires task_features.")
        if not isinstance(x, Mapping) or not x:
            raise ValueError("Task-conditioned attention requires a non-empty stage feature dictionary.")
        # 每一 stage 做注意力, 随后分别归一化并拼接
        pooled_stages = []
        for stage in sorted(x):
            pooled = TaskConditionedSpatialAttention(name=f"stage_{stage}_attention")(x[stage], task_features)
            pooled = nn.LayerNorm(name=f"stage_{stage}_norm")(pooled)
            pooled_stages.append(pooled)
        x = jnp.concatenate(pooled_stages, axis=-1)

    # 新版: 每个 stage 单独投影到 256 维并经过非线性映射, 再进行融合.
    # elif pooling_method == "task_conditioned_attention":
    #     if task_features is None:
    #         raise ValueError("pooling_method='task_conditioned_attention' requires task_features.")
    #     if not isinstance(x, Mapping) or not x:
    #         raise ValueError("Task-conditioned attention requires a non-empty stage feature dictionary.")
    #     pooled_stages = []
    #     for stage in sorted(x):
    #         pooled = TaskConditionedSpatialAttention(name=f"stage_{stage}_attention")(x[stage], task_features)
    #         pooled = nn.LayerNorm(name=f"stage_{stage}_norm")(pooled)
    #         pooled = nn.Dense(256, name=f"stage_{stage}_projection")(pooled)
    #         pooled_stages.append(nn.silu(pooled))
    #     x = jnp.concatenate(pooled_stages, axis=-1)

    # ----------------------------------- avg ----------------------------------- #
    elif pooling_method == "avg":
        x = jnp.mean(x, axis=(-3, -2))

    # ----------------------------------- max ----------------------------------- #
    elif pooling_method == "max":
        x = jnp.max(x, axis=(-3, -2))

    # ----------------------------------- none ----------------------------------- #
    elif pooling_method == "none":
        pass
    else:
        raise ValueError(f"Unknown pooling method: {pooling_method!r}.")

    # 将池化后的视觉特征投影到指定的低维表示
    if bottleneck_dim is not None:
        x = nn.Dense(bottleneck_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)

    return x
