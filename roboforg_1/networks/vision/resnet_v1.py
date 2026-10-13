import functools as ft
from functools import partial
from typing import Any, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from roboforg_1.networks.vision.image_utils import resize
from roboforg_1.networks.vision.resnet_components import (
    AddSpatialCoordinates,
    BottleneckResNetBlock,
    FilmConditioning,
    MyGroupNorm,
    ResNetBlock,
)
from roboforg_1.networks.vision.pretrained_encoder import PreTrainedResNetEncoder
ModuleDef = Any

# ------------------------------------------------------------------------- #
#                         ResNet Backbone                                   #
# ------------------------------------------------------------------------- #

class ResNetEncoder(nn.Module):
    """
    ResNetV1 backbone.当前 Flow Matching 使用 ResNet-10 配置

    以 128*128 RGB 输入, stage_sizes=(1, 1, 1, 1) 为例: 

        图像 [B, 128, 128, 3]
            │
            │ 归一化 → 7*7 Conv, stride 2 → GroupNorm → ReLU
            │
            ▼
        [B, 64, 64, 64]
            │
            │ 3*3 MaxPool, stride 2
            │
            ▼
        [B, 32, 32, 64]
            │
            │ stage 1: ResNetBlock(64), stride 1
            │
            ▼
        [B, 32, 32, 64]
            │
            │ stage 2: ResNetBlock(128), stride 2
            │
            ▼
        [B, 16, 16, 128]
            │
            │ stage 3: ResNetBlock(256), stride 2
            │
            ▼
        [B, 8, 8, 256]
            │
            │ stage 4: ResNetBlock(512), stride 2
            │
            ▼
        [B, 4, 4, 512]  ← 默认返回空间特征图

    非零 attention_stages 选择返回的阶段; None 或全零时返回最终 stage 的特征图.
    stage 3 后的任务条件残差注意力实验代码目前已注释, 不参与前向计算.
    
    本类始终返回未池化的空间特征图; 空间池化由外围编码器负责.开启 FiLM 或乘性条件化时, 各残差块之后还会增加对应的条件模块
    """

    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: str = "relu"
    conv: ModuleDef = nn.Conv
    norm: str = "group"
    add_spatial_coordinates: bool = False
    use_multiplicative_cond: bool = False
    use_film: bool = False
    freeze_backbone: bool = False
    image_size: tuple = (128, 128)
    attention_stages: Optional[tuple[bool, bool, bool, bool]] = None

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        encode: bool = True,
        train: bool = True,
        cond_var=None,
        stop_gradient=False,
        task_features=None,
    ):
        # EncodingWrapper uses one common interface for regular and pretrained
        # encoders. If features were encoded upstream, pass them through.
        """预处理图像并执行 ResNet, 返回未池化的空间特征."""

        if not encode:
            return observations

        # put inputs in [-1, 1]
        # x = observations.astype(jnp.float32) / 127.5 - 1.0
        if observations.shape[-3:-1] != self.image_size:
            observations = resize(observations, self.image_size)

        # imagenet mean and std # TODO: add this back
        mean = jnp.array([0.485, 0.456, 0.406])
        std = jnp.array([0.229, 0.224, 0.225])
        x = (observations.astype(jnp.float32) / 255.0 - mean) / std

        if self.add_spatial_coordinates:
            x = AddSpatialCoordinates(dtype=self.dtype)(x)

        conv = partial(
            self.conv,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nn.initializers.kaiming_normal(),
        )
        if self.norm == "batch":
            raise NotImplementedError
        elif self.norm == "group":
            norm = partial(MyGroupNorm, num_groups=4, epsilon=1e-5, dtype=self.dtype)
        elif self.norm == "layer":
            norm = partial(nn.LayerNorm, epsilon=1e-5, dtype=self.dtype)
        else:
            raise ValueError("norm not found")

        act = getattr(nn, self.act)

        x = conv(
            self.num_filters,
            (7, 7),
            (2, 2),
            padding=[(3, 3), (3, 3)],
            name="conv_init",
        )(x)

        x = norm(name="norm_init")(x)
        x = act(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")

        # 决定 ResNet 最多运行到第几个 stage
        #
        # * 无注意力时运行全部 stage
        # * 有注意力时只运行到最深的选中阶段
        last_stage = len(self.stage_sizes)
        if self.attention_stages is not None:
            if len(self.attention_stages) != len(self.stage_sizes):
                raise ValueError("attention_stages must have one entry per ResNet stage.")
        use_attention = bool(self.attention_stages and any(self.attention_stages))
        if use_attention:
            last_stage = max(i + 1 for i, enabled in enumerate(self.attention_stages) if enabled)

        stage_features = {}
        for i, block_size in enumerate(self.stage_sizes[:last_stage]):
            for j in range(block_size):
                stride = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(
                    self.num_filters * 2**i,
                    strides=stride,
                    conv=conv,
                    norm=norm,
                    act=act,
                )(x)
                if self.use_film:
                    assert (cond_var is not None), "Cond var is None, nothing to condition on"
                    x = FilmConditioning()(x, cond_var)
                if self.use_multiplicative_cond:
                    assert (cond_var is not None), "Cond var is None, nothing to condition on"
                    cond_out = nn.Dense(x.shape[-1], kernel_init=nn.initializers.xavier_normal())(cond_var)
                    x_mult = jnp.expand_dims(jnp.expand_dims(cond_out, 1), 1)
                    x = x * x_mult

            # if self.attention_stages is None and task_features is not None and i == 2:
            #     query = nn.LayerNorm(name="interstage_query_norm")(nn.Dense(x.shape[-1], use_bias=False, name="interstage_task_query")(task_features))
            #     keys  = nn.LayerNorm(name="interstage_key_norm")(x)
            #     height, width = x.shape[1:3]
            #     logits = jnp.einsum("bhwc,bc->bhw", keys, query) / jnp.sqrt(x.shape[-1])
            #     attention = nn.softmax(logits.reshape(x.shape[0], height * width), axis=-1)
            #     attention = attention.reshape(x.shape[0], height, width)
            #     self.sow("intermediates", "interstage_attention_map", attention)
            #     strength = self.param("interstage_attention_strength", nn.initializers.zeros_init(), ())
            #     x = x + strength * (height * width * attention[..., None] - 1.0) * x

            if use_attention and self.attention_stages[i]:
                stage_features[i + 1] = x

        if self.freeze_backbone or stop_gradient:
            x = jax.lax.stop_gradient(x)
            stage_features = jax.tree_util.tree_map(jax.lax.stop_gradient, stage_features)

        return stage_features if use_attention else x

# ------------------------------------------------------------------------- #
#                         ResNet 配置注册表                                 #
# ------------------------------------------------------------------------- #

# 名称只描述基础 ResNet 结构; 输出阶段, 冻结, FiLM 和空间池化由模块参数控制
resnetv1_configs = {
    "resnetv1-10": ft.partial(
        ResNetEncoder, stage_sizes=(1, 1, 1, 1), block_cls=ResNetBlock
    ),
    "resnetv1-18": ft.partial(
        ResNetEncoder, stage_sizes=(2, 2, 2, 2), block_cls=ResNetBlock
    ),
    "resnetv1-34": ft.partial(
        ResNetEncoder, stage_sizes=(3, 4, 6, 3), block_cls=ResNetBlock
    ),
    "resnetv1-50": ft.partial(
        ResNetEncoder, stage_sizes=(3, 4, 6, 3), block_cls=BottleneckResNetBlock
    ),
    # 非标准布局直接写明每个 stage 的残差块数量, 避免沿用误导性的深度名称
    "resnetv1-basic-3-3-3-3": ft.partial(
        ResNetEncoder, stage_sizes=(3, 3, 3, 3), block_cls=ResNetBlock
    ),
    "resnetv1-basic-4-4-4-4": ft.partial(
        ResNetEncoder, stage_sizes=(4, 4, 4, 4), block_cls=ResNetBlock
    ),
}
