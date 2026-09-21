"""ACT 策略主网络：观测与潜变量 z 共同预测未来动作 chunk。"""

from __future__ import annotations

from collections.abc import Mapping

import flax.linen as nn
import jax
import jax.numpy as jnp

from roboforg.networks.act.tokenizer import (
    ActionQueries,
    ImageTokenizer,
    ObservationInputAssembler,
    SinusoidalPositionEncoding2D,
    StateTokenizer,
)
from roboforg.networks.act.transformer import TransformerDecoder, TransformerEncoder


# ACT 右半部分：共享视觉主干、观测编码器、动作解码器和最终的动作投影。
class ACTPolicy(nn.Module):
    """Map normalized state, RGB observations, and z to a normalized action chunk."""

    backbone: nn.Module
    action_horizon: int
    action_dim: int = 7
    token_dim: int = 256
    num_heads: int = 8
    feedforward_dim: int = 2048
    encoder_num_layers: int = 4
    decoder_num_layers: int = 7
    dropout_rate: float = 0.1
    feature_height: int = 4
    feature_width: int = 4
    camera_keys: tuple[str, ...] = ("front", "wrist")
    backbone_trainable: bool = True

    # 输入 state [B,S]、images、z [B,Z]，输出标准化动作 [B,k,A]。
    @nn.compact
    def __call__(
        self,
        state: jax.Array,
        images: Mapping[str, jax.Array],
        z: jax.Array,
        *,
        train: bool = False,
    ) -> jax.Array:
        if self.action_horizon <= 0 or self.action_dim <= 0 or self.token_dim <= 0:
            raise ValueError("action_horizon, action_dim, and token_dim must be positive.")
        if self.feature_height <= 0 or self.feature_width <= 0:
            raise ValueError("feature_height and feature_width must be positive.")
        # 当前 ResNet 的 pre_pooling=True 会 stop_gradient；正式 ACT 微调时拒绝这种静默冻结配置。
        if self.backbone_trainable and getattr(self.backbone, "pre_pooling", False):
            raise ValueError(
                "A trainable ACT backbone requires pre_pooling=False. "
                "Use pooling_method='none' to retain the feature map without stop_gradient."
            )
        if not self.camera_keys or len(set(self.camera_keys)) != len(self.camera_keys):
            raise ValueError("camera_keys must be non-empty and contain no duplicates.")
        if state.ndim != 2 or state.shape[0] == 0 or state.shape[1] == 0:
            raise ValueError("state must have non-empty shape [B,S].")
        if z.ndim != 2 or z.shape[0] != state.shape[0] or z.shape[1] == 0:
            raise ValueError("z must have non-empty shape [B,Z] and match the state batch size.")

        batch_size = state.shape[0]
        # 当前 policy 的状态投影独立于训练期 CVAE 后验编码器中的状态投影。
        state_token = StateTokenizer(self.token_dim, name="state_tokenizer")(state)

        # 原版 ACT 的所有相机共用同一个视觉主干和 1×1 图像投影，不能为每个相机创建独立实例。
        image_tokenizer = ImageTokenizer(
            backbone=self.backbone, token_dim=self.token_dim, name="shared_image_tokenizer"
        )
        image_position = SinusoidalPositionEncoding2D(
            self.token_dim, name="image_position"
        )(self.feature_height, self.feature_width)
        expected_visual_tokens = self.feature_height * self.feature_width
        image_tokens: dict[str, jax.Array] = {}
        image_positions: dict[str, jax.Array] = {}

        # 依照 camera_keys 的固定顺序读取图像，保证 token 和二维位置编码的相机顺序稳定。
        for key in self.camera_keys:
            if key not in images:
                raise KeyError(f"images is missing camera {key!r}.")
            image = images[key]
            if image.ndim != 4 or image.shape[0] != batch_size or image.shape[-1] != 3:
                raise ValueError(f"images[{key!r}] must have shape [B,H,W,3] matching state batch size.")

            # 两次调用同一个模块对象，因此 front/wrist 共用完全相同的 ResNet 与特征投影参数。
            tokens = image_tokenizer(image, train=train)
            if tokens.shape[1] != expected_visual_tokens:
                raise ValueError(
                    f"Backbone for {key!r} returned {tokens.shape[1]} visual tokens, but "
                    f"feature_height*feature_width is {expected_visual_tokens}."
                )
            image_tokens[key] = tokens
            image_positions[key] = image_position

        # z、状态、各相机视觉 token 组成观测序列；其中 z/state 使用可学习位置标记。
        observation_tokens, observation_position = ObservationInputAssembler(
            token_dim=self.token_dim, camera_keys=self.camera_keys, name="observation_assembler"
        )(z, state_token, image_tokens, image_positions)
        memory = TransformerEncoder(
            token_dim=self.token_dim,
            num_heads=self.num_heads,
            feedforward_dim=self.feedforward_dim,
            num_layers=self.encoder_num_layers,
            dropout_rate=self.dropout_rate,
            name="observation_encoder",
        )(observation_tokens, observation_position, train=train)

        # k 个可学习 query 只作为位置信息；ACT 的 decoder 初始内容 tgt 为零向量。
        query_position = ActionQueries(
            num_queries=self.action_horizon, token_dim=self.token_dim, name="action_queries"
        )(batch_size)
        decoder_tokens = jnp.zeros_like(query_position)
        decoded = TransformerDecoder(
            token_dim=self.token_dim,
            num_heads=self.num_heads,
            feedforward_dim=self.feedforward_dim,
            num_layers=self.decoder_num_layers,
            dropout_rate=self.dropout_rate,
            name="action_decoder",
        )(
            decoder_tokens,
            memory,
            pos=observation_position,
            query_pos=query_position,
            train=train,
        )

        # 每一个 decoder 输出 token 独立投影为一时刻 A 维动作，保留原有 chunk 的时间顺序。
        return nn.Dense(
            self.action_dim, kernel_init=nn.initializers.xavier_uniform(), name="action_head"
        )(decoded)
