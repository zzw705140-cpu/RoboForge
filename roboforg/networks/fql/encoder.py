"""参考官方 FQL 的 IMPALA 图像编码器，并适配双相机与机器人状态。"""

from collections.abc import Mapping
from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from roboforg.networks.fql.network import MLP

#################################################################
#        卷积，下采样和残差块

class ResnetStack(nn.Module):
    """3*3 卷积 → 下采样 → 残差块，空间尺寸默认减半。"""

    num_features: int                  # 当前阶段的输出通道数，如 16 或 32。
    num_blocks:   int                  # 当前阶段包含几个残差块。
    max_pooling:  bool = True          # 是否通过池化将图像高、宽减半。

    @nn.compact
    def __call__(self, x):
        # 初始化卷积权重；首层卷积调整通道数，保持高、宽不变。
        initializer = nn.initializers.xavier_uniform()
        x = nn.Conv(
            self.num_features, (3, 3), padding="SAME", kernel_init=initializer,
        )(x)
        # 3×3 最大池化，步长为 2；例如 [B,128,128,16] → [B,64,64,16]。
        if self.max_pooling:
            x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")

        # 每个残差块由两次 ReLU → 卷积组成，块内形状保持不变。
        for _ in range(self.num_blocks):
            # 保存块的输入，供后面与卷积分支的结果相加。
            residual = x
            x = nn.relu(x)
            x = nn.Conv(
                self.num_features, (3, 3), padding="SAME", kernel_init=initializer,
            )(x)
            x = nn.relu(x)
            x = nn.Conv(
                self.num_features, (3, 3), padding="SAME", kernel_init=initializer,
            )(x)
            # 残差连接：输出 = 输入 + 学到的修正量，也为梯度提供直接路径。
            x = x + residual
        return x


#################################
#          单相机编码器           #
#################################

class ImpalaEncoder(nn.Module):
    """单相机编码器；默认结构与官方 ImpalaEncoder 一致。

    [B,128,128,3] → [B,64,64,16] → [B,32,32,32]
                  → [B,16,16,32] → [B,8192] → [B,512]
    输入为原始 0~255 像素；单样本可省略 B 维。
    """

    width:            int = 1                             # 通道数倍率，实际通道数 = size × width。
    stack_sizes:      tuple[int, ...] = (16, 32, 32)      # 三个卷积阶段的基础通道数。
    num_blocks:       int = 2                             # 每阶段残差块数量；small 配置传入 1。
    dropout_rate:     float | None = None                 # 可选 Dropout，默认关闭。
    mlp_hidden_dims:  Sequence[int] = (512,)              # 展平后的 MLP 层宽，默认输出 512 维。
    layer_norm:       bool = False                        # 控制末端特征及 MLP 的 LayerNorm。

    def setup(self):
        # 按配置创建三个阶段，每经过一个阶段，高、宽减半。
        self.stack_blocks = [
            ResnetStack(num_features=size * self.width, num_blocks=self.num_blocks)
            for size in self.stack_sizes
        ]
        # Dropout 训练时随机置零部分特征，推理时关闭。
        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def __call__(self, x, train=False):
        """归一化像素、提取空间特征，展平后映射为特征向量。"""

        # 原始 uint8 像素 [0,255] → float32 [0,1]，上游不要重复除以 255。
        x = x.astype(jnp.float32) / 255.0
        features = x

        # 依次提取图像特征：128×128 → 64×64 → 32×32 → 16×16。
        for stack in self.stack_blocks:
            features = stack(features)
            if self.dropout_rate is not None:
                features = self.dropout(features, deterministic=not train)

        # 卷积阶段结束后统一激活，并按配置对最后的通道维做 LayerNorm。
        features = nn.relu(features)
        if self.layer_norm:
            features = nn.LayerNorm()(features)

        # 保留批量维，将空间和通道展平：[B,16,16,32] → [B,8192]。
        features = features.reshape((*x.shape[:-3], -1))

        # 映射为 512 维图像特征；这里输出是特征，因此最后一层也激活。
        return MLP(
            self.mlp_hidden_dims, activate_final=True, layer_norm=self.layer_norm,
        )(features)


#################################
#          双相机编码器           #
#################################

class ObservationEncoder(nn.Module):
    """适配我们的观测字典，两路图像编码器参数独立。

    front [B,128,128,3] → IMPALA → [B,512] ──┐
    wrist [B,128,128,3] → IMPALA → [B,512] ──┼→ 拼接 [B,1047]
    state [B,23]   ──────────────────────────┘

    默认每个 stack 使用一个残差块，对应官方 impala_small。
    状态仅转为 float32;状态标准化留给统一数据预处理。
    """

    camera_keys:     tuple[str, ...] = ("front", "wrist")  # 相机字段名，同时确定拼接顺序。
    state_dim:       int = 23                              # 每个观测中的机器人状态维度。

    # 以下配置传给每一路图像编码器，默认对应 impala_small。
    width:           int = 1
    stack_sizes:     tuple[int, ...] = (16, 32, 32)
    num_blocks:      int = 1
    mlp_hidden_dims: Sequence[int] = (512,)
    layer_norm:      bool = False

    def setup(self):
        # 为 front、wrist 分别创建编码器：结构相同，但权重独立学习。
        self.image_encoders = {
            key: ImpalaEncoder(
                width           =self.width,
                stack_sizes     =self.stack_sizes,
                num_blocks      =self.num_blocks,
                mlp_hidden_dims =self.mlp_hidden_dims,
                layer_norm      =self.layer_norm,
                name=f"{key}_encoder",
            )
            for key in self.camera_keys
        }

    def __call__(self, observations: Mapping):
        """分别编码图像，再按 front、wrist、state 顺序拼接。"""
        
        # 取出状态并检查维度；这里只转换类型，不改变状态的数值尺度。
        state = jnp.asarray(observations["state"], dtype=jnp.float32)

        if state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state dimension {self.state_dim}, got {state.shape[-1]}.")
        
        # 检查图像与状态的批量维一致，并逐路得到 [B,512] 图像特征。
        # 单样本没有 B 维，同样适用。
        features = []
        for key in self.camera_keys:
            image = observations[key]
            if image.shape[:-3] != state.shape[:-1] or image.shape[-1] != 3:
                raise ValueError(f"Camera {key!r} must contain RGB images with matching batch dimensions.")
            features.append(self.image_encoders[key](image))

        # 直接拼接机器人状态：[B,512] + [B,512] + [B,23] → [B,1047]。
        features.append(state)

        return jnp.concatenate(features, axis=-1)
