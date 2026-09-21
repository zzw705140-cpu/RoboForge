"""ACT 训练所需的状态、动作标准化工具。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np


# 防止某个几乎不变化的维度出现除零或过度放大的标准化结果。
_MIN_STD = 1e-2


# 保存训练集统计量，并负责 ACT 的状态、动作标准化。
@dataclass(frozen=True)
class DataNormalizer:
    """使用训练集逐维均值和标准差标准化 state/action。"""

    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    # 创建 normalizer 时统一检查统计量的维度、数值与最小标准差。
    def __post_init__(self) -> None:
        state_mean = _as_stat_vector(self.state_mean, name="state_mean")
        state_std = _as_stat_vector(self.state_std, name="state_std")
        action_mean = _as_stat_vector(self.action_mean, name="action_mean")
        action_std = _as_stat_vector(self.action_std, name="action_std")

        if state_mean.shape != state_std.shape:
            raise ValueError("state_mean and state_std must have the same shape.")
        if action_mean.shape != action_std.shape:
            raise ValueError("action_mean and action_std must have the same shape.")

        # 标准差不可能为负；为 0 的恒定维度交给下方的最小值保护处理。
        if np.any(state_std < 0.0) or np.any(action_std < 0.0):
            raise ValueError("All standard deviations must be non-negative.")

        object.__setattr__(self, "state_mean", state_mean)
        object.__setattr__(self, "state_std", np.maximum(state_std, _MIN_STD))
        object.__setattr__(self, "action_mean", action_mean)
        object.__setattr__(self, "action_std", np.maximum(action_std, _MIN_STD))

    # 从完整训练轨迹统计 state/action 的逐维均值与标准差。
    @classmethod
    def fit(cls, trajectories: Iterable[Mapping[str, Any]]) -> "DataNormalizer":
        """Fit statistics from train trajectories in compact DataBuffer format."""
        state_sequences: list[np.ndarray] = []
        action_sequences: list[np.ndarray] = []
        state_dim: int | None = None
        action_dim: int | None = None

        for trajectory_index, trajectory in enumerate(trajectories):
            try:
                states = np.asarray(trajectory["observations"]["state"], dtype=np.float32)
                actions = np.asarray(trajectory["actions"], dtype=np.float32)
            except (KeyError, TypeError) as error:
                raise KeyError(
                    "Each trajectory must contain observations['state'] and actions."
                ) from error

            if states.ndim != 2 or actions.ndim != 2:
                raise ValueError(
                    f"Trajectory {trajectory_index} must contain [T+1,S] states and [T,A] actions."
                )
            if states.shape[0] != actions.shape[0] + 1:
                raise ValueError(
                    f"Trajectory {trajectory_index} has {states.shape[0]} states but "
                    f"{actions.shape[0]} actions; expected one extra terminal state."
                )
            if actions.shape[0] == 0:
                raise ValueError(f"Trajectory {trajectory_index} must contain at least one action.")
            if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
                raise ValueError(f"Trajectory {trajectory_index} contains NaN or Inf.")
            if state_dim is None:
                state_dim, action_dim = states.shape[1], actions.shape[1]
            elif states.shape[1] != state_dim or actions.shape[1] != action_dim:
                raise ValueError(
                    f"Trajectory {trajectory_index} has state/action dimensions "
                    f"{states.shape[1:]}/{actions.shape[1:]}, expected {state_dim}/{action_dim}."
                )

            # 训练样本只会用 obs_t 配对 action_t；末尾 obs_T 没有后续动作，不计入统计。
            state_sequences.append(states[:-1])
            action_sequences.append(actions)

        if not state_sequences:
            raise ValueError("Cannot fit DataNormalizer from zero trajectories.")

        all_states = np.concatenate(state_sequences, axis=0)
        all_actions = np.concatenate(action_sequences, axis=0)
        return cls(
            state_mean=all_states.mean(axis=0),
            state_std=all_states.std(axis=0),
            action_mean=all_actions.mean(axis=0),
            action_std=all_actions.std(axis=0),
        )

    # 将原始状态转为以 0 为中心、标准差约为 1 的网络输入。
    def normalize_state(self, state: Any) -> jnp.ndarray:
        """Normalize an array whose final dimension is the state dimension."""
        return _normalize(state, mean=self.state_mean, std=self.state_std, name="state")

    # 将原始动作或动作 chunk 转为网络训练使用的标准化标签。
    def normalize_action(self, action: Any) -> jnp.ndarray:
        """Normalize an array whose final dimension is the action dimension."""
        return _normalize(action, mean=self.action_mean, std=self.action_std, name="action")

    # 将网络预测的标准化动作恢复为环境可执行的原始动作尺度。
    def denormalize_action(self, action: Any) -> jnp.ndarray:
        """Map normalized action predictions back to the environment action scale."""
        action_array = _as_last_dim_array(action, dimension=self.action_mean.size, name="action")
        return action_array * jnp.asarray(self.action_std) + jnp.asarray(self.action_mean)

    # 转为普通 NumPy 字典，供训练 checkpoint 与评估脚本保存和加载。
    def to_dict(self) -> dict[str, np.ndarray]:
        """Return a serializable copy of the four training-set statistics."""
        return {
            "state_mean": self.state_mean.copy(),
            "state_std": self.state_std.copy(),
            "action_mean": self.action_mean.copy(),
            "action_std": self.action_std.copy(),
        }

    # 从 checkpoint 中保存的统计量重建同一个 normalizer。
    @classmethod
    def from_dict(cls, statistics: Mapping[str, Any]) -> "DataNormalizer":
        """Restore a normalizer previously produced by :meth:`to_dict`."""
        required_keys = {"state_mean", "state_std", "action_mean", "action_std"}
        missing_keys = required_keys - set(statistics)
        if missing_keys:
            raise KeyError(f"Normalizer statistics are missing keys: {sorted(missing_keys)}.")
        return cls(**{key: statistics[key] for key in required_keys})


# 将保存的统计量转换为一维、有限的 float32 数组。
def _as_stat_vector(value: Any, *, name: str) -> np.ndarray:
    """Validate one saved mean/std vector."""
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array, got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf.")
    return array.copy()


# 检查输入最后一维，并转换为 JAX 数组以便纳入后续 jitted 前向传播。
def _as_last_dim_array(value: Any, *, dimension: int, name: str) -> jnp.ndarray:
    """Convert input to float32 JAX array and validate its final dimension."""
    array = jnp.asarray(value, dtype=jnp.float32)
    if array.ndim == 0 or array.shape[-1] != dimension:
        raise ValueError(
            f"{name} must have final dimension {dimension}, got shape {array.shape}."
        )
    return array


# 应用逐维 z-score 标准化；广播机制自动支持单样本、batch 和动作 chunk。
def _normalize(value: Any, *, mean: np.ndarray, std: np.ndarray, name: str) -> jnp.ndarray:
    """Apply (value - mean) / std along the final dimension."""
    array = _as_last_dim_array(value, dimension=mean.size, name=name)
    return (array - jnp.asarray(mean)) / jnp.asarray(std)
