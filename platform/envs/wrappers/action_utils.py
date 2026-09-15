from __future__ import annotations

import numpy as np

#######################################################
#               动作合法性检查与归一化工具                #
#######################################################


def project_action_to_unit_balls(action: np.ndarray) -> np.ndarray:
    """
    针对动作空间 [dx, dy, dz, dRx, dRy, dRz] (可选夹爪指令)

    * 分别将 “平移 [dx, dy, dz]” 与 “旋转 [dRx, dRy, dRz]” 独立投影至单位球，但方向保持不变
    * 夹爪指令独立剪裁至 [-1, 1] 之间
    """

    action = np.asarray(action)

    # 检查动作维度及类型，由于 tolerance=np.inf, 不检查速度
    validate_action_in_unit_balls(action, tolerance=np.inf)

    # 平移 & 旋转动作投影
    projected = action.copy()
    for start in (0, 3):
        vector = projected[..., start : start + 3]
        norm = np.linalg.norm(vector, axis=-1, keepdims=True)
        projected[..., start : start + 3] = vector / np.maximum(norm, 1.0)

    # 夹爪动作投影
    if projected.shape[-1] == 7:
        projected[..., 6] = np.clip(projected[..., 6], -1.0, 1.0)

    return projected


def validate_action_in_unit_balls(
    action: np.ndarray,
    *,
    tolerance: float = 1e-5,
) -> None:
    """检查动作维度及类型，同时检查是否满足单位球约束"""

    action = np.asarray(action)
    if action.ndim == 0 or action.shape[-1] not in (6, 7):
        raise ValueError(f"action must have a final dimension of 6 or 7, got shape {action.shape}.")
    if not np.issubdtype(action.dtype, np.floating):
        raise TypeError(f"action must use a floating dtype, got {action.dtype}.")
    if not np.all(np.isfinite(action)):
        raise ValueError("action contains NaN or Inf.")
    if tolerance < 0.0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}.")

    translation_norm = np.linalg.norm(action[..., :3], axis=-1)
    if np.any(translation_norm > 1.0 + tolerance):
        raise ValueError(f"translation action must lie in the unit L2 ball, but the maximum norm is {float(np.max(translation_norm)):.6f}.")

    rotation_norm = np.linalg.norm(action[..., 3:6], axis=-1)
    if np.any(rotation_norm > 1.0 + tolerance):
        raise ValueError(f"rotation action must lie in the unit L2 ball, but the maximum norm is {float(np.max(rotation_norm)):.6f}.")

    if action.shape[-1] == 7:
        gripper = action[..., 6]
        if np.any((gripper < -1.0 - tolerance) | (gripper > 1.0 + tolerance)):
            raise ValueError(f"gripper action must lie in [-1, 1], but the observed range is [{float(np.min(gripper)):.6f}, {float(np.max(gripper)):.6f}].")
