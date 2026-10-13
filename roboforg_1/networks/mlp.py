from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp

from roboforg_1.networks.common import default_init   # 神经网络权重初始化方法


##############################################################################
#                              Basic MLP                                     #
##############################################################################
# 最基础的MLP网络


class MLP(nn.Module):
    """由多层全连接层组成的基础神经网络."""

    """
    注!: 下面的维度都是举例, 在创建网络是要传入具体数值
    [B,128]
       ↓ Dense → LayerNorm → GELU
    [B,256]
       ↓ Dense → LayerNorm → GELU
    [B,256]
       ↓ Dense
     [B,1]
    """

    hidden_dims: Sequence[int]                                          # 隐藏层维度列表, 例如 [256, 256]
    activations: Callable[[jnp.ndarray], jnp.ndarray] | str = nn.swish  # 激活函数, 可以是 flax.linen 中的函数或字符串名称
    activate_final: bool = False                                        # 是否在最后一层后应用激活函数
    use_layer_norm: bool = False                                        # 控制是否进行归一化（对特征）
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        activations = self.activations
        if isinstance(activations, str):
            activations = getattr(nn, activations)

        # 依次构建全连接层，并按配置处理每层输出。
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init)(x)

            # 如果不是最后一层，或者 activate_final 为 True，则应用激活函数、dropout 和 layer norm
            if i < len(self.hidden_dims) - 1 or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)

                # 添加激活函数  
                x = activations(x)
 
        return x


##############################################################################
#                            MLP with condition                              #
##############################################################################
# 让 MLP 根据条件信息处理主输入。 它先拼接两份输入，再调用前面讲的普通 MLP

class MLPWithCondition(nn.Module):
    """
    在输入端拼接主输入与条件特征的 MLP, 与 FiLM 的逐残差块调制不同, 本模块只在进入 MLP 前执行一次: 
    ``inputs = concatenate([x, condition], axis=-1)``

    对于 Flow Matching, ``x`` 通常是当前带噪动作 ``x_t``, ``condition`` 通常是编码后的 observation 特征与时间特征

    
    x:[B,7]       condition:[B,128]
      └──────────┬──────────┘
                拼接
                 ├→ 沿特征维拼接 → 普通 MLP → 输出
    
    输入支持单条特征 [D] 或一批特征 [B, D], 两份输入的 batch 大小必须一致.
    hidden_dims 包含最后的输出层宽度, 例如 (256, 256, 7) 表示输出 7 维.
    本模块接收已编码的条件, 不负责观测编码, 时间编码或动作 chunk 的展平与恢复.
    """

    # --------------------------- 网络结构配置 --------------------------- #

    hidden_dims: Sequence[int]                                         # 各层宽度，最后一项为输出维度
    activations: Callable[[jnp.ndarray], jnp.ndarray] | str = nn.swish
    activate_final: bool = False                                       # 是否对输出层使用激活等处理
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None

    # --------------------------- 条件拼接与前向计算 --------------------------- #

    @nn.compact
    def __call__(self, x: jnp.ndarray, condition: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """检查输入形状, 沿特征维拼接一次, 再调用普通 MLP."""
        # 主输入和条件均为 [D]，或均为 [B, D]；特征宽度可以不同。
        if x.ndim not in (1, 2) or condition.ndim != x.ndim:
            raise ValueError(f"MLPWithCondition expects x and condition to both be [D] or both be [B, D], got {x.shape} and {condition.shape}.")
        if x.shape[:-1] != condition.shape[:-1]:
            raise ValueError(f"x and condition must have matching leading dimensions, got {x.shape} and {condition.shape}.")

        # 例如 [B, 7] 与 [B, 128] 拼接后为 [B, 135]，batch 维保持不变。
        inputs = jnp.concatenate([x, condition], axis=-1)

        # 层宽、激活、归一化和 Dropout 均转交给已有 MLP，避免重复实现。
        return MLP(
            hidden_dims=self.hidden_dims,
            activations=self.activations,
            activate_final=self.activate_final,
            use_layer_norm=self.use_layer_norm,
            dropout_rate=self.dropout_rate,
        )(inputs, train=train)
