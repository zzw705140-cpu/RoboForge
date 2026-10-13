from functools import wraps
from collections.abc import Callable

import jax
import jax.numpy as jnp
import flax.linen as nn

##############################################################################
#                        State Value Network                                 #
##############################################################################

class StateValueCritic(nn.Module):
    """
    通用状态价值 head: features → V(s)

    ``output_activation=None`` 时返回无界标量; 需要把价值解释为概率或归一化进度时, 可由 Agent 显式传入 ``nn.sigmoid``
    """

    network: nn.Module
    output_activation: Callable[[jax.Array], jax.Array] | None = None

    @nn.compact
    def __call__(self, features: jax.Array, train: bool = False) -> jax.Array:
        """
        计算状态价值, 支持以下输入类型: 

        1. 单个状态
            features: ``[F]``
            output:   ``[]``

        2. 一批状态
            features: ``[B, F]``
            output:   ``[B]``
        """

        value = self.network(features, train=train).squeeze(-1)

        if self.output_activation is not None:
            value = self.output_activation(value)

        return value

##############################################################################
#                         Action State Network                               #
##############################################################################

def multiple_action_q_function(forward):
    """
    装饰器, 让 Q(s, a) 同时支持单个动作和 N 个候选动作评估

    case 0:   features [F],   actions [A],     输出 [] (标量)
    case 1:   features [F],   actions [N,A],   输出 [N]
    case 2:   features [B,F], actions [B,A],   输出 [B]
    case 3:   features [B,F], actions [B,N,A], 输出 [B,N]
    """

    @wraps(forward)
    def wrapped(self, features, actions, train=False):
        """根据动作形状组织单动作或多候选动作的 Q 值计算."""

        # case 1: 固定 features, 沿 actions 的候选维并行计算
        if features.ndim == 1 and actions.ndim == 2:
            """
            输入: 沿 in_axes=0 候选动作维 N 并行计算

            中间过程:

            第1组动作 actions[0, :]   → [A] → Q值 []
            第2组动作 actions[1, :]   → [A] → Q值 []
            ...
            第N组动作 actions[N-1, :] → [A] → Q值 []

            输出: 在 N 个 forward 计算结果 [] 在最后一维拼接, 形成 [N]
            """
            return jax.vmap(lambda action: forward(self, features, action, train), in_axes=0, out_axes=0)(actions)

        # case 3: actions [B,N,A], 输出 q_values [B,N]
        if actions.ndim == 3:
            """
            输入: 沿 in_axes=1 候选动作维 N 并行计算

            中间过程:

            第1组动作 actions[:, 0, :]   → [B,A] → Q值 [B]
            第2组动作 actions[:, 1, :]   → [B,A] → Q值 [B]
            ...
            第N组动作 actions[:, N-1, :] → [B,A] → Q值 [B]

            输出: 在 N 个 forward 计算结果 [B] 在最后一维拼接, 形成 [B, N]
            """
            return jax.vmap(lambda a: forward(self, features, a, train), in_axes=1, out_axes=-1)(actions)

        # case 0 或 case 2
        return forward(self, features, actions, train)

    return wrapped

class ActionValueCritic(nn.Module):
    """通用动作价值 head: (features, action chunk) → Q(s,a)

    输出变换由具体 Agent 决定: 普通回报 Critic 通常使用线性输出; 把 Q 值解释为成功概率或归一化进度时可以传入 ``nn.sigmoid``
    """

    network: nn.Module
    output_activation: Callable[[jax.Array], jax.Array] | None = None

    @nn.compact
    @multiple_action_q_function
    def __call__(self, features: jax.Array, actions: jax.Array, train: bool = False) -> jax.Array:
        """
        计算状态-动作对的价值, 支持以下输入类型: 

        1. 单个状态, 且每个状态对应 1 个候选动作: 
            features: ``[F]``
            actions:  ``[A]``
            output:   ``[]``

        2. 一批状态, 且每个状态对应 1 个候选动作: 
            features: ``[B, F]``
            actions:  ``[B, A]``
            output:   ``[B]``

        3. 单个状态, 且每个状态对应 N 个候选动作: 
            features: ``[F]``
            actions:  ``[N, A]``
            output:   ``[N]``

        4. 一批状态, 且每个状态对应 N 个候选动作: 
            features: ``[B, F]``
            actions:  ``[B, N, A]``
            output:   ``[B, N]``

        若使用 ensemblize(..., num_qs=M) 后, 以上输出会在最前面增加独立 Q-head 维, 变为 ``[M]``, ``[M,B]``, ``[M,N]`` 或 ``[M,B,N]``
        """

        q = self.network(jnp.concatenate([features, actions], axis=-1), train=train).squeeze(-1)

        if self.output_activation is not None:
            q = self.output_activation(q)

        return q

##############################################################################
#                              Q Ensemble                                    #
##############################################################################

def ensemblize(cls, num_qs, out_axes=0):
    """
    将 Q 网络复制为多个拥有独立参数的 Q head, 接收同一份 features 和 actions

    若单个 Q head 分别输出:  ``[]``, ``[B]``, ``[N]`` 或 ``[B,N]``
    则 Ensemble 对应输出:   ``[num_qs]``, ``[num_qs,B]``, ``[num_qs,N]`` 或 ``[num_qs,B,N]``
    """

    if num_qs <= 0:
        raise ValueError("num_qs must be positive.")

    class EnsembleModule(nn.Module):
        """参数独立的 Q 网络集合"""

        @nn.compact
        def __call__(self, *args, train=False, **kwargs):
            """使用独立参数的多个 Q 网络评估相同输入, 并堆叠结果."""

            # 将 Flax 网络类批量复制成 num_qs 个结构相同, 参数独立的 Q critic, 并用 jax "并行" 计算, 构成 Critic ensemble
            ensemble = nn.vmap(
                cls,
                variable_axes={"params": 0},                      # 参数沿第 0 维独立存储.
                split_rngs={"params": True, "dropout": True},      # 独立初始化和 Dropout 随机数.
                in_axes=None,                                    # 所有 Q 网络接收同一份输入.
                out_axes=out_axes,                               # 默认在第 0 维堆叠输出.
                axis_size=num_qs,
            )

            # 实例化 ensemble 类 (只声明网络结构, 参数保存在外部, 开销不大)
            ensemble_model = ensemble()

            # 调用实例 ensemble() 的前向计算
            # 将 train 作为位置参数传入, 确保每个 Q 网络都收到训练标志.
            return ensemble_model(*args, train, **kwargs)

    return EnsembleModule
