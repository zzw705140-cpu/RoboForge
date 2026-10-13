import flax.linen as nn
import jax
import jax.numpy as jnp


##############################################################################
#                         Entropy Temperature                                #
##############################################################################

class EntropyTemperature(nn.Module):
    """保存并返回 entropy temperature: ``alpha=exp(log_alpha)``"""

    initial_temperature: float

    @nn.compact
    def __call__(self) -> jax.Array:

        # 创建可训练参数 log_alpha, 并不存储在 EntropyTemperature 类实例里，而是由 Flax 放在外部参数树中
        log_alpha = self.param(
            "log_alpha",
            lambda _: jnp.log(jnp.asarray(self.initial_temperature, dtype=jnp.float32)),
        )
        return jnp.exp(log_alpha)
