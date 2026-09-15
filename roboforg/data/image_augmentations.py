from functools import partial

import jax
import jax.numpy as jnp

################################################################
#                  图像增强工具，训练视觉策略时做出的调整            #
################################################################


# 对单张图像随机裁剪，保持输出尺寸不变。
def random_crop(img, rng, *, padding):
    crop_from = jax.random.randint(rng, (2,), 0, 2 * padding + 1)
    crop_from = jnp.concatenate([crop_from, jnp.zeros((1,), dtype=jnp.int32)])
    padded_img = jnp.pad(
        img,
        (
            (padding, padding),
            (padding, padding),
            (0, 0),
        ),
        mode="edge",
    )
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


# 对一批图像分别进行随机裁剪。
@partial(jax.jit, static_argnames=("padding", "num_batch_dims"))
def batched_random_crop(img, rng, *, padding, num_batch_dims: int = 1):
    # Flatten batch dims
    original_shape = img.shape
    img = jnp.reshape(img, (-1, *img.shape[num_batch_dims:]))

    rngs = jax.random.split(rng, img.shape[0])
 
    img = jax.vmap(
        lambda i, r: random_crop(i, r, padding=padding), in_axes=(0, 0), out_axes=0
    )(img, rngs)

    # Restore batch dims
    img = jnp.reshape(img, original_shape)
    return img

# 使用双线性插值调整图像尺寸。
def resize(image, image_dim):
    assert len(image_dim) == 2
    new_shape = list(image.shape)
    new_shape[-3:-1] = image_dim
    return jax.image.resize(image, new_shape, method="bilinear")


# 按指定概率决定是否执行增强操作。
def _maybe_apply(apply_fn, inputs, rng, apply_prob):
    should_apply = jax.random.uniform(rng, shape=()) <= apply_prob
    return jax.lax.cond(should_apply, inputs, apply_fn, inputs, lambda x: x)


# 对各图像通道分别做卷积，为高斯模糊提供底层计算。
def _depthwise_conv2d(inputs, kernel, strides, padding):
    """Computes a depthwise conv2d in Jax.
    Args:
        inputs: an NHWC tensor with N=1.
        kernel: a [H", W", 1, C] tensor.
        strides: a 2d tensor.
        padding: "SAME" or "VALID".
    Returns:
        The depthwise convolution of inputs with kernel, as [H, W, C].
    """
    return jax.lax.conv_general_dilated(
        inputs,
        kernel,
        strides,
        padding,
        feature_group_count=inputs.shape[-1],
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )


# 对单张图像进行指定强度的高斯模糊。
# sigma 小：主要参考附近的像素，模糊较轻。
# sigma 大：更远的像素也有较大影响，模糊更明显。
def _gaussian_blur_single_image(image, kernel_size, padding, sigma):
    """Applies gaussian blur to a single image, given as NHWC with N=1."""
    radius = int(kernel_size / 2)
    kernel_size_ = 2 * radius + 1
    x = jnp.arange(-radius, radius + 1).astype(jnp.float32)
    blur_filter = jnp.exp(-(x**2) / (2.0 * sigma**2))
    blur_filter = blur_filter / jnp.sum(blur_filter)
    blur_v = jnp.reshape(blur_filter, [kernel_size_, 1, 1, 1])
    blur_h = jnp.reshape(blur_filter, [1, kernel_size_, 1, 1])
    num_channels = image.shape[-1]
    blur_h = jnp.tile(blur_h, [1, 1, 1, num_channels])
    blur_v = jnp.tile(blur_v, [1, 1, 1, num_channels])
    expand_batch_dim = len(image.shape) == 3
    if expand_batch_dim:
        image = image[jnp.newaxis, ...]
    blurred = _depthwise_conv2d(image, blur_h, strides=[1, 1], padding=padding)
    blurred = _depthwise_conv2d(blurred, blur_v, strides=[1, 1], padding=padding)
    blurred = jnp.squeeze(blurred, axis=0)
    return blurred


# 随机决定是否进行高斯模糊，并随机选择模糊程度。
def _random_gaussian_blur(
    image, rng, *, kernel_size, padding, sigma_min, sigma_max, apply_prob
):
    """Applies a random gaussian blur."""
    apply_rng, transform_rng = jax.random.split(rng)

    # 随机选择模糊强度，并对图像执行高斯模糊。
    def _apply(image):
        (sigma_rng,) = jax.random.split(transform_rng, 1)
        sigma = jax.random.uniform(
            sigma_rng, shape=(), minval=sigma_min, maxval=sigma_max, dtype=jnp.float32
        )
        return _gaussian_blur_single_image(image, kernel_size, padding, sigma)

    return _maybe_apply(_apply, image, apply_rng, apply_prob)


# 将 RGB 转为 HSV，便于调整色相和饱和度。
def rgb_to_hsv(r, g, b):
    """Converts R, G, B  values to H, S, V values.
    Reference TF implementation:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/adjust_saturation_op.cc
    Only input values between 0 and 1 are guaranteed to work properly, but this
    function complies with the TF implementation outside of this range.
    Args:
        r: A tensor representing the red color component as floats.
        g: A tensor representing the green color component as floats.
        b: A tensor representing the blue color component as floats.
    Returns:
        H, S, V values, each as tensors of shape [...] (same as the input without
        the last dimension).
    """
    vv = jnp.maximum(jnp.maximum(r, g), b)
    range_ = vv - jnp.minimum(jnp.minimum(r, g), b)
    sat = jnp.where(vv > 0, range_ / vv, 0.0)
    norm = jnp.where(range_ != 0, 1.0 / (6.0 * range_), 1e9)

    hr = norm * (g - b)
    hg = norm * (b - r) + 2.0 / 6.0
    hb = norm * (r - g) + 4.0 / 6.0

    hue = jnp.where(r == vv, hr, jnp.where(g == vv, hg, hb))
    hue = hue * (range_ > 0)
    hue = hue + (hue < 0)

    return hue, sat, vv


# 将 HSV 颜色表示转换回 RGB。
def hsv_to_rgb(h, s, v):
    """Converts H, S, V values to an R, G, B tuple.
    Reference TF implementation:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/adjust_saturation_op.cc
    Only input values between 0 and 1 are guaranteed to work properly, but this
    function complies with the TF implementation outside of this range.
    Args:
        h: A float tensor of arbitrary shape for the hue (0-1 values).
        s: A float tensor of the same shape for the saturation (0-1 values).
        v: A float tensor of the same shape for the value channel (0-1 values).
    Returns:
        An (r, g, b) tuple, each with the same dimension as the inputs.
    """
    c = s * v
    m = v - c
    dh = (h % 1.0) * 6.0
    fmodu = dh % 2.0
    x = c * (1 - jnp.abs(fmodu - 1))
    hcat = jnp.floor(dh).astype(jnp.int32)
    rr = (
        jnp.where(
            (hcat == 0) | (hcat == 5), c, jnp.where((hcat == 1) | (hcat == 4), x, 0)
        )
        + m
    )
    gg = (
        jnp.where(
            (hcat == 1) | (hcat == 2), c, jnp.where((hcat == 0) | (hcat == 3), x, 0)
        )
        + m
    )
    bb = (
        jnp.where(
            (hcat == 3) | (hcat == 4), c, jnp.where((hcat == 2) | (hcat == 5), x, 0)
        )
        + m
    )
    return rr, gg, bb


# 给各通道像素值加上指定偏移量，调整亮度。
def adjust_brightness(rgb_tuple, delta):
    return jax.tree_map(lambda x: x + delta, rgb_tuple)


# 按指定系数调整图像对比度。
def adjust_contrast(image, factor):
    # 调整单个通道中像素偏离通道均值的程度。
    def _adjust_contrast_channel(channel):
        mean = jnp.mean(channel, axis=(-2, -1), keepdims=True)
        return factor * (channel - mean) + mean

    return jax.tree_map(_adjust_contrast_channel, image)


# 调整 HSV 的饱和度，让颜色更鲜艳或更接近灰色。
def adjust_saturation(h, s, v, factor):
    return h, jnp.clip(s * factor, 0.0, 1.0), v


# 调整 HSV 的色相，使颜色在色相环上偏移。
def adjust_hue(h, s, v, delta):
    # Note: this method exactly matches TF"s adjust_hue (combined with the hsv/rgb
    # conversions) when running on GPU. When running on CPU, the results will be
    # different if all RGB values for a pixel are outside of the [0, 1] range.
    return (h + delta) % 1.0, s, v


# 随机选择偏移量，调整图像亮度。
def _random_brightness(rgb_tuple, rng, max_delta):
    delta = jax.random.uniform(rng, shape=(), minval=-max_delta, maxval=max_delta)
    return adjust_brightness(rgb_tuple, delta)


# 随机选择系数，调整图像对比度。
def _random_contrast(rgb_tuple, rng, max_delta):
    factor = jax.random.uniform(
        rng, shape=(), minval=1 - max_delta, maxval=1 + max_delta
    )
    return adjust_contrast(rgb_tuple, factor)


# 转换到 HSV 随机调整饱和度，再转回 RGB。
def _random_saturation(rgb_tuple, rng, max_delta):
    h, s, v = rgb_to_hsv(*rgb_tuple)
    factor = jax.random.uniform(
        rng, shape=(), minval=1 - max_delta, maxval=1 + max_delta
    )
    return hsv_to_rgb(*adjust_saturation(h, s, v, factor))


# 转换到 HSV 随机调整色相，再转回 RGB。
def _random_hue(rgb_tuple, rng, max_delta):
    h, s, v = rgb_to_hsv(*rgb_tuple)
    delta = jax.random.uniform(rng, shape=(), minval=-max_delta, maxval=max_delta)
    return hsv_to_rgb(*adjust_hue(h, s, v, delta))


# 将彩色图像转换为灰度图，同时保留三个通道。
def _to_grayscale(image):
    rgb_weights = jnp.array([0.2989, 0.5870, 0.1140])
    grayscale = jnp.tensordot(image, rgb_weights, axes=(-1, -1))[..., jnp.newaxis]
    return jnp.tile(grayscale, (1, 1, 3))  # Back to 3 channels.


# 按概率组合颜色扰动和灰度转换，可打乱颜色调整顺序。
def color_transform(
    image,
    rng,
    *,
    brightness,
    contrast,
    saturation,
    hue,
    to_grayscale_prob,
    color_jitter_prob,
    apply_prob,
    shuffle
):
    """Applies color jittering to a single image."""
    apply_rng, transform_rng = jax.random.split(rng)
    perm_rng, b_rng, c_rng, s_rng, h_rng, cj_rng, gs_rng = jax.random.split(
        transform_rng, 7
    )

    # Whether the transform should be applied at all.
    should_apply = jax.random.uniform(apply_rng, shape=()) <= apply_prob
    # Whether to apply grayscale transform.
    should_apply_gs = jax.random.uniform(gs_rng, shape=()) <= to_grayscale_prob
    # Whether to apply color jittering.
    should_apply_color = jax.random.uniform(cj_rng, shape=()) <= color_jitter_prob

    # 为颜色调整函数包装执行条件。
    def _make_cond(fn, idx):
        # 跳过增强，直接返回输入。
        def identity_fn(x, unused_rng, unused_param):
            return x

        # 根据概率条件和操作编号决定是否执行颜色调整。
        def cond_fn(args, i):
            # 将各通道处理结果限制在 [0, 1] 范围内。
            def clip(args):
                return jax.tree_map(lambda arg: jnp.clip(arg, 0.0, 1.0), args)

            out = jax.lax.cond(
                should_apply & should_apply_color & (i == idx),
                args,
                lambda a: clip(fn(*a)),
                args,
                lambda a: identity_fn(*a),
            )
            return jax.lax.stop_gradient(out)

        return cond_fn

    random_brightness_cond = _make_cond(_random_brightness, idx=0)
    random_contrast_cond = _make_cond(_random_contrast, idx=1)
    random_saturation_cond = _make_cond(_random_saturation, idx=2)
    random_hue_cond = _make_cond(_random_hue, idx=3)

    # 按指定或随机顺序串联亮度、对比度、饱和度和色相调整。
    def _color_jitter(x):
        rgb_tuple = tuple(jax.tree_map(jnp.squeeze, jnp.split(x, 3, axis=-1)))
        if shuffle:
            order = jax.random.permutation(perm_rng, jnp.arange(4, dtype=jnp.int32))
        else:
            order = range(4)
        for idx in order:
            if brightness > 0:
                rgb_tuple = random_brightness_cond((rgb_tuple, b_rng, brightness), idx)
            if contrast > 0:
                rgb_tuple = random_contrast_cond((rgb_tuple, c_rng, contrast), idx)
            if saturation > 0:
                rgb_tuple = random_saturation_cond((rgb_tuple, s_rng, saturation), idx)
            if hue > 0:
                rgb_tuple = random_hue_cond((rgb_tuple, h_rng, hue), idx)
        return jnp.stack(rgb_tuple, axis=-1)

    out_apply = _color_jitter(image)
    out_apply = jax.lax.cond(
        should_apply & should_apply_gs, out_apply, _to_grayscale, out_apply, lambda x: x
    )
    return jnp.clip(out_apply, 0.0, 1.0)


# 以约 50% 的概率左右翻转图像。
def random_flip(image, rng):
    _, flip_rng = jax.random.split(rng)
    should_flip_lr = jax.random.uniform(flip_rng, shape=()) <= 0.5
    image = jax.lax.cond(should_flip_lr, image, jnp.fliplr, image, lambda x: x)
    return image


# 配置模糊核大小、强度范围和概率，调用随机高斯模糊。
def gaussian_blur(
    image, rng, *, blur_divider=10.0, sigma_min=0.1, sigma_max=2.0, apply_prob=1.0
):
    """Applies gaussian blur to a batch of images.
    Args:
        images: an NHWC tensor, with C=3.
        rng: a single PRNGKey.
        blur_divider: the blurring kernel will have size H / blur_divider.
        sigma_min: the minimum value for sigma in the blurring kernel.
        sigma_max: the maximum value for sigma in the blurring kernel.
        apply_prob: the probability of applying the transform to a batch element.
    Returns:
        A NHWC tensor of the blurred images.
    """
    kernel_size = image.shape[0] / blur_divider
    blur_fn = partial(
        _random_gaussian_blur,
        kernel_size=kernel_size,
        padding="SAME",
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        apply_prob=apply_prob,
    )
    return blur_fn(image, rng)


# 按指定概率执行基于阈值的局部反色。
def solarize(image, rng, *, threshold, apply_prob):
    # 保留低于阈值的像素，其余像素替换为 1 减去原值。
    def _apply(image):
        return jnp.where(image < threshold, image, 1.0 - image)

    return _maybe_apply(_apply, image, rng, apply_prob)
