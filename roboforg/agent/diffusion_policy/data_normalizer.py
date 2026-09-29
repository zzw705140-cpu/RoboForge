"""DP 动作、状态与图像的数值处理。

图像输入保持 [B, To, H, W, 3]，要求原始 uint8 RGB。
训练和推理必须使用相同的统计文件，视觉编码器不要再次归一化。
"""
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np


# ============================================================================
# 共享数值变换：动作采用逐维仿射变换，图像使用训练集逐相机 RGB 统计量
# ============================================================================
class DataNormalizer:
    """默认 7 维动作恒等变换；图像统计默认来自随模块保存的 train_1 文件。

    更换任务可传入新的动作 scale/offset、图像和状态统计文件。
    保存模型时应一并保存这些参数和统计文件，不能在评估集上重新拟合。
    """

    # 加载并验证图像统计量，配置动作缩放参数；不读取原始演示数据。
    def __init__(self, *, action_scale=None, action_offset=None, image_stats_path: str | Path | None = None, state_stats_path: str | Path | None = None) -> None:
       
        # 当前环境已使用单位范围动作，包括夹爪，默认不改变任何动作数值。
        # 配置动作的缩放与偏移
        scale  = np.ones(7, dtype=np.float32) if action_scale is None else np.asarray(action_scale, dtype=np.float32)
        offset = np.zeros_like(scale) if action_offset is None else np.asarray(action_offset, dtype=np.float32)

        # 检查动作是否有合法
        if scale.ndim != 1 or scale.size == 0 or offset.shape != scale.shape:
            raise ValueError("Action scale/offset must be matching nonempty vectors.")
        if not np.isfinite(scale).all() or not np.isfinite(offset).all() or np.any(scale == 0):
            raise ValueError("Action parameters must be finite and scale must be nonzero.")

        self.action_scale  = jnp.asarray(scale)
        self.action_offset = jnp.asarray(offset)

        # 统计量对应已经映射到 [-1, 1] 的像素，各路相机分别保存。
        # 定位并读取图像统计文件
        path     = Path(image_stats_path) if image_stats_path is not None else Path(__file__).with_name('image_stats.json')
        metadata = json.loads(path.read_text())

        # 检查统计文件是否符合规
        if metadata.get('pixel_space') != '2 * (uint8 / 255) - 1':
            raise ValueError("Image statistics must describe pixels after the [-1, 1] transform.")
        if not metadata.get('images'):
            raise ValueError("Image statistics must contain at least one camera.")

        self.image_stats = {}

        # 逐路读取非空相机统计信息
        for key, entry in metadata['images'].items():
            mean = np.asarray(entry['mean'], dtype=np.float32)
            std  = np.asarray(entry['std'], dtype=np.float32)

            if mean.shape != (3,) or std.shape != (3,):
                raise ValueError(f"Camera {key!r} requires three RGB means and standard deviations.")
            if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 0):
                raise ValueError(f"Camera {key!r} contains invalid image statistics.")

            # 恒定或近乎恒定通道使用标准差下限，避免除零。
            self.image_stats[key] = (jnp.asarray(mean), jnp.asarray(np.maximum(std, 1e-6)))

        # 加载训练集状态变换参数；布局说明保存在 JSON 中，不新增环境或数据布局校验机制。
        state_path = Path(state_stats_path) if state_stats_path is not None else Path(__file__).with_name('state_stats.json')
        state_metadata = json.loads(state_path.read_text())
        state_scale = np.asarray(state_metadata['state_scale'], dtype=np.float32)
        state_offset = np.asarray(state_metadata['state_offset'], dtype=np.float32)

        # 当前输入固定为 23 维；只检查参数形状和数值，避免错误广播或无效计算。
        if state_scale.shape != (23,) or state_offset.shape != (23,):
            raise ValueError("State scale/offset must have shape (23,).")
        if not np.isfinite(state_scale).all() or not np.isfinite(state_offset).all() or np.any(state_scale <= 0):
            raise ValueError("State parameters must be finite and scale must be positive.")
        self.state_scale = jnp.asarray(state_scale)
        self.state_offset = jnp.asarray(state_offset)

    # ------------------------------------------------------------------------
    # 状态变换：位置与速度按训练范围缩放，夹爪开度映射，其余维度保持原值
    # ------------------------------------------------------------------------

    # 对 [B, To, 23] 状态的最后一维进行仿射变换，保持 batch 和历史帧结构。
    def normalize_state(self, state) -> jnp.ndarray:
        state = jnp.asarray(state, dtype=jnp.float32)
        if state.ndim != 3 or state.shape[-1] != 23 or any(size <= 0 for size in state.shape):
            raise ValueError("State must have shape [B, To, 23] with nonempty dimensions.")

        # [0:1] 开度用 2x-1；[5:8] 位置、[17:23] 速度按训练集范围缩放。
        # 旋转、夹爪目标及占位力/力矩使用 scale=1、offset=0。
        # 近恒定范围维度使用 scale=1、offset=-min；推理越界值不裁剪。
        return state * self.state_scale + self.state_offset

    # ------------------------------------------------------------------------
    # 动作变换：共享同一组参数，支持单步动作以及 [B, horizon, action_dim]
    # ------------------------------------------------------------------------

    # 对动作最后一维逐维缩放和平移；当前配置下为恒等变换。
    def normalize_action(self, action) -> jnp.ndarray:
        action = jnp.asarray(action, dtype=jnp.float32)

        # 前面的 batch、时间维不受影响，不进行裁剪或夹爪阈值判断。
        if action.ndim < 1 or action.shape[-1] != self.action_scale.shape[0]:
            raise ValueError("Action last dimension must match action_scale.")
        return action * self.action_scale + self.action_offset

    # 将网络生成的归一化动作还原为环境控制指令。
    def denormalize_action(self, action) -> jnp.ndarray:
        action = jnp.asarray(action, dtype=jnp.float32)
        if action.ndim < 1 or action.shape[-1] != self.action_scale.shape[0]:
            raise ValueError("Action last dimension must match action_scale.")

        # 逆变换后由环境处理夹爪的 ±0.5 阈值及持续开合目标。
        return (action - self.action_offset) / self.action_scale

    # ------------------------------------------------------------------------
    # 图像处理：先转换像素范围，再使用该相机训练集统计量进行标准化
    # ------------------------------------------------------------------------

    # 处理单路相机的原始 RGB batch，保留历史时间维、空间维及 RGB 顺序。
    def normalize_image(self, image, *, image_key: str) -> jnp.ndarray:
        image = jnp.asarray(image)

        # 只接受原始像素，避免已经处理过的浮点图像被再次归一化。
        if image.dtype != jnp.uint8:
            raise TypeError("Image must be raw uint8 RGB pixels.")
        if image.ndim != 5 or image.shape[-1] != 3 or any(size <= 0 for size in image.shape):
            raise ValueError("Image must have shape [B, To, H, W, 3] with nonempty dimensions.")
        if image_key not in self.image_stats:
            raise KeyError(f"No training image statistics for camera {image_key!r}.")

        # 保留约定的两阶段变换；统计值来自用户训练集，而非 ImageNet。
        image = image.astype(jnp.float32) / 255.0
        image = image * 2.0 - 1.0
        mean, std = self.image_stats[image_key]

        # [3] 的统计向量自动广播到每个 batch、时间步与像素位置。
        return (image - mean) / std
