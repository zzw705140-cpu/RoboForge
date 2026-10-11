#####################################
#           FQL网络组件部分           #
#####################################
from typing import Any, Sequence

import flax.linen as nn
import jax.numpy as jnp


# 将普通网络类转换为集成网络类，让多个参数独立的网络处理同一份输入
def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):

    # 通过 Flax 的 nn.vmap 转换网络类，返回一个可以批量执行多个网络的类
    return nn.vmap(
        cls,
        variable_axes={"params": 0, "intermediates": 0},
        split_rngs   ={"params": True},      #为各网络的参数初始化分配不同随机密钥，使两个网络的初始权重不同
        in_axes      =in_axes,               #两个网络都接收完整的同一份输入
        out_axes     =out_axes,              #表示将各网络的输出沿第 0 维堆叠
        axis_size    =num_qs,                # 指定的网络数量
        **kwargs,
    )


#######################
#         MLP         #
#######################

class MLP(nn.Module):
    """
    x [B, D+7]
        ▼
    [Dense(512) → GELU → 可选 LayerNorm] * 4
        ├──► 记录 feature [B, 512]
        ▼
    Dense(7)
        ▼
    output [B, 7]

inputs:
    单个样本：
        x: [input_dim]
    批量样本：
        x: [B, input_dim]

outputs:
    单个样本：
        output: [output_dim]
    批量样本：
        output: [B, output_dim]
    """

    hidden_dims:    Sequence[int]
    activations:    Any = nn.gelu          # gelu激活函数
    activate_final: bool = False           # 默认最后一层只做线性映射，不接激活和 LayerNorm。
    kernel_init:    Any = nn.initializers.variance_scaling(1.0, "fan_avg", "uniform")
    layer_norm:     bool = False           # 默认关闭 LayerNorm，可以通过配置开启

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            # 默认仅隐藏层使用激活函数和可选的 LayerNorm。
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
            # 记录倒数第二层特征，不改变前向计算结果。
            if i == len(self.hidden_dims) - 2:
                self.sow("intermediates", "feature", x)
        return x


######################
#       Value        #
######################

class Value(nn.Module):
    """
    双相机图像各 [B, 128, 128, 3]＋状态 [B, 23]
            │
        encoder
            │
        特征 [B, D] ── 拼接动作 [B, 7]
                              │
                          [B, D+7]
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
                   MLP₁                MLP₂
                    │                   │
                Q₁ [B, 1]           Q₂ [B, 1]
                    └─────────┬─────────┘
                              ▼
                          合并 [2, B, 1]
                              │ squeeze(-1)
                              ▼
                            [2, B]
    """

    hidden_dims:   Sequence[int]
    layer_norm:    bool = True
    num_ensembles: int = 2                      #集成中包含的网络数量。
    encoder:       nn.Module | None = None      #可选的编码器模块，用于编码输入。

    # 创建网络
    def setup(self):
        # 仅集成后面的 MLP；观测编码器由两个 Q 网络共享。
        mlp_class = MLP

        # 两个MLP接收相同的输入，但各自拥有独立的参数
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)

        self.value_net = mlp_class(
            (*self.hidden_dims, 1),         #在隐藏层后面追加输出层维度1
            activate_final=False,           #最后一层不使用激活函数和LayerNorm
            layer_norm=self.layer_norm,
        )

    def __call__(self, observations, actions=None):
        """编码观测并拼接动作，计算价值；默认双 Q 输出形状为 [2, B]"""

        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]

        # 将动作加入列表
        if actions is not None:
            inputs.append(actions)

        # 观测特征 [B, D] 与动作 [B, 7] 拼接成 [B, D+7]。
        inputs = jnp.concatenate(inputs, axis=-1)

        # 返回两个Q值
        return self.value_net(inputs).squeeze(-1)


######################
#      策略网络       #
######################

class ActorVectorField(nn.Module):
    """
    双相机图像＋状态[B,23]  → encoder → 观测特征 [B,1047]

    BC flow:
        特征[B,1047] ＋插值动作[B,7] ＋时间[B,1] → [B,1055] → MLP → 速度 [B,7]

    单步策略：
        特征[B,1047] ＋噪声[B,7]               → [B,1054] → MLP → 动作 [B,7]

    两种策略分别创建实例，参数独立。
    
    上述维度对应默认 ObservationEncoder;单样本可省略 B 维
    """

    hidden_dims: Sequence[int]              # 隐藏层宽度，由 agent 配置。
    action_dim: int = 7                     # 输出速度或动作的维度。
    layer_norm: bool = False                # 官方策略 MLP 默认不使用 LayerNorm。
    encoder: nn.Module | None = None        # 接收我们观测字典的编码器。

    def setup(self):
        # 在隐藏层后追加 7 维输出层；输出层不加激活或 LayerNorm。
        self.mlp = MLP(
            (*self.hidden_dims, self.action_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )

    def __call__(self, observations, actions, times=None, is_encoded=False):
        """编码并拼接输入,输出速度或动作;is_encoded=True 时复用观测特征。"""
        
        # Euler 积分时可预先编码一次，之后各步传入特征，避免重复处理图像。
        # 无 encoder 或 is_encoded=True 时，observations 必须是特征数组。
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)

        # actions 沿用官方命名：BC flow 传插值动作，单步策略传高斯噪声。
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            # 时间形状为 [B,1]（单样本为 [1]），与输入的批量维一致。
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        # 保留原始网络输出；积分、动作裁剪和损失计算由 agent 负责。
        return self.mlp(inputs)
