"""视觉编码器共用的图像尺寸调整函数."""

import jax


def resize(image, image_dim):
    """按双线性插值调整图像高宽, 保留前导维度和通道数."""
    # 与 GCP 的 resize 保持一致, 仅替换最后三个维度中的高宽.
    assert len(image_dim) == 2
    new_shape = list(image.shape)
    new_shape[-3:-1] = image_dim
    return jax.image.resize(image, new_shape, method="bilinear")
