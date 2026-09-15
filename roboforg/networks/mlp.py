from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp

from roboforg.networks.common import default_init   # 神经网络权重初始化方法


##############################################################################
#                              Basic MLP                                     #
##############################################################################
#最基础的MLP网络


class MLP(nn.Module):
    """由多层全连接层组成的基础神经网络。"""

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
