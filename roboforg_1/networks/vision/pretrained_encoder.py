from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from roboforg_1.networks.vision.spatial_pooling import _apply_spatial_pooling_and_bottleneck


# ------------------------------------------------------------------------- #
#                       预训练 ResNet 编码包装器                               #
# ------------------------------------------------------------------------- #

class PreTrainedResNetEncoder(nn.Module):
    pooling_method: str = "avg"
    train_backbone: bool = False
    use_spatial_softmax: bool = False
    softmax_temperature: float = 1.0
    num_spatial_blocks: int = 8
    bottleneck_dim: Optional[int] = None
    pretrained_encoder: nn.module = None

    def encode_backbone(self, observations: jnp.ndarray, *, train: bool = False, task_features: Optional[jax.Array] = None) -> jax.Array:
        """返回空间池化之前的特征图; 仅在启用微调的训练调用中回传梯度"""

        update_backbone = train and self.train_backbone
        return self.pretrained_encoder(
            observations,
            train=update_backbone,
            stop_gradient=not update_backbone,
            task_features=task_features,
        )

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        encode: bool = True,
        train: bool = True,
        task_features: Optional[jax.Array] = None,
    ):
        """按需执行 Backbone, 随后进行空间池化和瓶颈投影."""

        x = observations
        if encode:
            x = self.encode_backbone(x, train=train, task_features=task_features)

        return _apply_spatial_pooling_and_bottleneck(
            x,
            pooling_method=self.pooling_method,
            num_spatial_blocks=self.num_spatial_blocks,
            softmax_temperature=self.softmax_temperature,
            bottleneck_dim=self.bottleneck_dim,
            train=train,
            task_features=task_features,
        )
