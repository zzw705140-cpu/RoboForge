import flax.linen as nn
import jax
import jax.numpy as jnp


##############################################################################
#                          One Step Actor Net                                #
##############################################################################

class OneStepActorNet(nn.Module):
    """
    将编码后的观测特征和初始噪声, 一步映射为最终动作, 不包含迭代去噪过程

    #                     ┌─────────────────────┐
    #    features ───────►│                     │
    #                     │       Network       │──► actions
    #    noise ──────────►│                     │
    #                     └─────────────────────┘

    Args:
        network: 条件动作主干, 调用接口必须为 network(noise, features, train=train)

    输入: 
        单个样本: 
            features: [feature_dim]
            noise:    [flow_dim]

        批量样本: 
            features: [B, feature_dim]
            noise:    [B, flow_dim]

    输出: 
        展平的动作 chunk, 形状为 [flow_dim] 或 [B, flow_dim]

    注意, 本网络不使用时间编码, 也不进行 ODE/SDE 积分或动作投影
    """

    network: nn.Module

    @nn.compact
    def __call__(self, features: jax.Array, noise: jax.Array, train: bool = False) -> jax.Array:
        if features.ndim not in (1, 2):
            raise ValueError(f"features must have shape [feature_dim] or [batch, feature_dim], got {features.shape}")
        if noise.ndim != features.ndim:
            raise ValueError(f"features and noise must either both be unbatched or both be batched, got {features.shape} and {noise.shape}")
        if features.shape[:-1] != noise.shape[:-1]:
            raise ValueError(f"features and noise must have matching leading dimensions, got {features.shape} and {noise.shape}")

        # 直接以编码后的观测特征作为条件，将初始噪声一次映射为最终动作 chunk
        actions = self.network(noise, features, train=train)

        # 按当前动作空间噪声的约定, 输出动作必须与噪声形状一致.
        if actions.shape != noise.shape:
            raise ValueError(f"network must output actions with the same shape as noise, got {actions.shape} and {noise.shape}")

        return actions
