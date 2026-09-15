from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import flax.linen as nn
import jax
from flax.core import FrozenDict

# 神经网络参数的默认 Xavier 均匀初始化方法。
default_init = nn.initializers.xavier_uniform()

# Flax 模型参数树的类型别名。
Params = FrozenDict[str, Any]

# JAX 随机数密钥的类型别名。
PRNGKey = jax.Array
