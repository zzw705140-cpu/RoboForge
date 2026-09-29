"""Diffusion Policy 观测编码：共享 ResNet-10 提取相机特征，并拼接状态。

输入图像须先由 DataNormalizer.normalize_image 处理；本模块不再改变像素值。
每路图像与状态保留相同的 [B, To] 前缀，输出每帧观测特征 [B, To, D]。
"""
from __future__ import annotations

import pickle
from collections.abc import Mapping
from functools import partial
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict

from roboforg.networks.vision.resnet_v1 import MyGroupNorm, ResNetBlock


# ---------------------------------------------------------------------------
# 视觉骨干：沿用现有预训练 ResNet-10 的层名与参数形状，输入已经归一化。
# ---------------------------------------------------------------------------
class ResNet10ImageEncoder(nn.Module):
    """输入 [N,H,W,3]，经四个残差阶段和全局平均池化输出 [N,512]。"""

    num_filters: int = 64        #卷积通道数

    # 提取一批图像的特征；不在网络内部重复做 /255 或通道标准化。
    @nn.compact
    def __call__(self, images: jax.Array) -> jax.Array:

        # 检查图像形状和类型，输入已归一化
        if images.ndim != 4 or images.shape[-1] != 3 or any(size <= 0 for size in images.shape):
            raise ValueError("images must have shape [N,H,W,3] with nonempty dimensions.")
        if not jnp.issubdtype(images.dtype, jnp.floating):
            raise TypeError("images must be normalized floating-point RGB values.")

        # 与已存在的 ResNet-10 相同的卷积、GroupNorm 和残差层参数结构。
        conv = partial(nn.Conv, use_bias=False, dtype=jnp.float32, kernel_init=nn.initializers.kaiming_normal())
        norm = partial(MyGroupNorm, num_groups=4, epsilon=1e-5, dtype=jnp.float32)

        # 网络部分：提取初始图像特征
        x = conv(self.num_filters, (7, 7), (2, 2), padding=[(3, 3), (3, 3)], name="conv_init")(images)
        x = norm(name="norm_init")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")

        # 每阶段一个残差块；后面三个阶段各在首块将空间尺寸减半。
        for stage in range(4):
            x = ResNetBlock(self.num_filters * 2**stage, strides=(2, 2) if stage > 0 else (1, 1), conv=conv, norm=norm, act=nn.relu)(x)

        # 全局平均池化只压缩空间维；最后一阶段输出 512 通道。
        return jnp.mean(x, axis=(1, 2))


# ---------------------------------------------------------------------------
# 多相机观测：相机和时间步共用一个骨干，每帧拼接图像特征与状态。
# ---------------------------------------------------------------------------
class ObservationEncoder(nn.Module):
    """默认 front、wrist 两路相机；输出 [B, To, n_cameras*512+state_dim]。"""

    camera_keys: tuple[str, ...] = ("front", "wrist")
    state_dim:   int = 23     # 每帧23维
    n_obs_steps: int = 2      # 连续2帧观测

    # 编码已归一化的图像和状态；相机顺序由 camera_keys 固定。
    @nn.compact
    def __call__(self, observations: Mapping[str, jax.Array]) -> jax.Array:
        if not self.camera_keys or len(set(self.camera_keys)) != len(self.camera_keys):
            raise ValueError("camera_keys must be nonempty and unique.")
        if self.state_dim <= 0 or self.n_obs_steps <= 0:
            raise ValueError("state_dim and n_obs_steps must be positive.")
        if "state" not in observations:
            raise KeyError("observations is missing 'state'.")
        state = jnp.asarray(observations["state"])
        if state.ndim != 3 or state.shape[1:] != (self.n_obs_steps, self.state_dim) or state.shape[0] <= 0:
            raise ValueError("state must have shape [B, n_obs_steps, state_dim].")
        if not jnp.issubdtype(state.dtype, jnp.floating):
            raise TypeError("state must be normalized floating-point values.")

        # 同一个 Flax 模块对象被所有相机与时间帧调用，保证只有一套 ResNet 参数。
        backbone = ResNet10ImageEncoder(name="backbone")
        features = []
        batch_size = state.shape[0]
        for key in self.camera_keys:
            if key not in observations:
                raise KeyError(f"observations is missing camera {key!r}.")
            image = jnp.asarray(observations[key])
            if image.ndim != 5 or image.shape[:2] != (batch_size, self.n_obs_steps) or image.shape[-1] != 3 or min(image.shape[2:4]) <= 0:
                raise ValueError(f"Camera {key!r} must have shape [B, n_obs_steps, H, W, 3].")
            # 转换数据格式
            image_batch = image.reshape(batch_size * self.n_obs_steps, *image.shape[2:])

            #共享ResNet维每张图生成512维特征，然后恢复时间维度，加入features
            encoded     = backbone(image_batch).reshape(batch_size, self.n_obs_steps, -1)
            features.append(encoded)

        # 每帧顺序为 camera_keys 中的各相机特征，最后是状态；不混合时间步。
        return jnp.concatenate((*features, state), axis=-1)


# ---------------------------------------------------------------------------
# 预训练权重：仅替换 backbone 子树，其余网络参数保留传入值。
# ---------------------------------------------------------------------------

def load_pretrained_resnet10(params, checkpoint_path: str | Path) -> Mapping:
    """将本地可信 ResNet-10 checkpoint 加载到 ObservationEncoder 参数树。"""
    # 读取本地预训练权重。
    path = Path(checkpoint_path).expanduser()
    with path.open("rb") as file:
        pretrained = pickle.load(file)

    # 若权重外层包含 params，则取出内部参数树。
    if "params" in pretrained:
        pretrained = pretrained["params"]

    # 检查当前模型是否包含视觉骨干。
    if "backbone" not in params:
        raise KeyError("ObservationEncoder parameters are missing 'backbone'.")

    # 展开当前骨干和预训练参数树，方便按参数路径匹配。
    target = flatten_dict(unfreeze(params["backbone"]))
    source = flatten_dict(unfreeze(pretrained))

    # 检查预训练权重是否缺少当前骨干需要的参数。
    missing = set(target) - set(source)
    if missing:
        raise ValueError(f"Pretrained ResNet-10 is missing parameters: {sorted(missing)}")

    # 逐项检查参数形状，并转换为与当前参数类型一致的 JAX 数组。
    loaded = {}
    for key, value in target.items():
        if np.shape(source[key]) != value.shape:
            raise ValueError(f"Pretrained parameter {'/'.join(key)} has incompatible shape.")
        loaded[key] = jnp.asarray(source[key], dtype=value.dtype)

    # 恢复嵌套结构，只替换 backbone，其余参数保留原值。
    updated = unfreeze(params)
    updated["backbone"] = unflatten_dict(loaded)

    # 返回不可直接修改的参数字典，不影响后续训练。
    return freeze(updated)
