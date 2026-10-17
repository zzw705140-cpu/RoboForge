from __future__ import annotations

import jax
import jax.numpy as jnp

#######################################################
#             Action and Gripper Clipping             #
#######################################################

"""策略动作的单位球投影与夹爪范围限制.
   
   动作限制：
   对策略生成的动作进行判断,将动作长度超过规定1时将动作裁剪到球面上
   注！：这里会做一次检测和裁剪，底层环境中只做判断

   夹爪限制：
   将策略输出的夹爪动作数值限制在【-1,1】中
"""


def project_action_chunks_to_unit_balls(flat_actions: jax.Array, *, action_horizon: int, action_dim: int) -> jax.Array:
    """逐步约束展平动作块, 返回与输入形状相同的动作数组.

    输入: 
    [..., action_horizon * action_dim], action_dim 为 6 或 7.

    每步的三维平移和旋转分别限制在单位 L2 球内, 球内向量保持原样.
    第七维夹爪指令裁剪至 [-1, 1]
    """
    # 检查动作布局, 前导维度可以是单样本, 批量或多候选动作维度.
    flat_actions = jnp.asarray(flat_actions)
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive.")
    if action_dim not in (6, 7):
        raise ValueError(f"action_dim must be 6 or 7, got {action_dim}.")
    if flat_actions.ndim == 0 or flat_actions.shape[-1] != action_horizon * action_dim:
        raise ValueError(f"flat_actions must end with action_horizon * action_dim = {action_horizon * action_dim}, got shape {flat_actions.shape}.")

    # 恢复动作块布局, 例如 [B,H*A] -> [B,H,A].
    chunk_shape = (*flat_actions.shape[:-1], action_horizon, action_dim)
    action_chunks = jnp.reshape(flat_actions, chunk_shape)

    # 分别投影平移和旋转, 分母等价于 max(L2 范数, 1), 保留方向.
    # 在开平方前设置下限, 避免零向量处的范数梯度出现 NaN.
    translation = action_chunks[..., :3]
    translation_scale = jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(translation), axis=-1, keepdims=True), 1.0))
    translation = translation / translation_scale

    rotation = action_chunks[..., 3:6]
    rotation_scale = jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(rotation), axis=-1, keepdims=True), 1.0))
    rotation = rotation / rotation_scale

    # 七维动作额外限制夹爪, 六维动作只包含平移和旋转.
    projected_parts = [translation, rotation]
    if action_dim == 7:
        projected_parts.append(jnp.clip(action_chunks[..., 6:7], -1.0, 1.0))

    # 恢复原输入的展平布局, 供上层策略或算法调用.
    projected_chunks = jnp.concatenate(projected_parts, axis=-1)
    return jnp.reshape(projected_chunks, flat_actions.shape)
