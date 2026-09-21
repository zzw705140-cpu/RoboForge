"""ACT 执行期的 action chunk 时间对齐与 Temporal Ensemble。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# 缓存一次策略查询的完整动作块，并记录它对应的真实环境起始时间步。
@dataclass(frozen=True)
class _CachedActionChunk:
    start_step: int
    actions: np.ndarray

    # chunk 中最后一个动作对应的真实时间步，闭区间 [start_step, end_step] 都有效。
    @property
    def end_step(self) -> int:
        return self.start_step + self.actions.shape[0] - 1


# 将多个历史 chunk 中针对同一真实时间步的动作建议进行时间对齐和加权融合。
class ActionEnsemble:
    """Stateful ACT Temporal Ensemble for one environment rollout."""

    def __init__(
        self,
        action_horizon: int,
        *,
        action_dim: int = 7,
        decay: float = 0.01,
        prefer_recent: bool = False,
    ) -> None:
        """Create an empty chunk cache.

        ``prefer_recent=False`` reproduces the ordering in the original ACT
        evaluation code: earlier-starting chunks get larger exponential weight.
        """
        if action_horizon <= 0 or action_dim <= 0:
            raise ValueError("action_horizon and action_dim must be positive.")
        if decay < 0.0:
            raise ValueError("decay must be non-negative.")

        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.decay = float(decay)
        self.prefer_recent = bool(prefer_recent)
        self._chunks: list[_CachedActionChunk] = []
        self._last_added_step: int | None = None

    # 新回合开始时清空历史预测，防止不同 episode 的动作被错误融合。
    def reset(self) -> None:
        """Discard every cached prediction and begin a fresh rollout."""
        self._chunks.clear()
        self._last_added_step = None

    # 缓存一个在 start_step 时刻预测的完整 [k,A] 动作块。
    def add_prediction(self, start_step: int, action_chunk: np.ndarray) -> None:
        """Store one raw-scale action chunk predicted at ``start_step``."""
        if not isinstance(start_step, (int, np.integer)) or start_step < 0:
            raise ValueError("start_step must be a non-negative integer.")
        actions = np.asarray(action_chunk, dtype=np.float32)
        expected_shape = (self.action_horizon, self.action_dim)
        if actions.shape != expected_shape:
            raise ValueError(f"action_chunk must have shape {expected_shape}, got {actions.shape}.")
        if not np.all(np.isfinite(actions)):
            raise ValueError("action_chunk contains NaN or Inf.")
        if self._last_added_step is not None and start_step < self._last_added_step:
            raise ValueError("start_step must be non-decreasing within one rollout; call reset first.")

        # 同一环境时刻重复查询时覆盖旧预测，而不是把同一个时刻重复计入平均。
        if self._chunks and start_step == self._chunks[-1].start_step:
            self._chunks[-1] = _CachedActionChunk(int(start_step), actions.copy())
        else:
            self._chunks.append(_CachedActionChunk(int(start_step), actions.copy()))
        self._last_added_step = int(start_step)

        # 早于当前查询时刻便已失效的 chunk 不可能再参与后续融合，立即释放。
        self._discard_expired(current_step=int(start_step))

    # 取出所有覆盖 step 的历史建议，按时间顺序加权平均为一个环境动作。
    def get_action(self, step: int) -> np.ndarray:
        """Return the fused raw-scale action [A] for one environment time step."""
        if not isinstance(step, (int, np.integer)) or step < 0:
            raise ValueError("step must be a non-negative integer.")
        self._discard_expired(current_step=int(step))

        # chunks 保持起始时间升序；每个 chunk 的索引 step-start_step 正好是它对当前时刻的建议。
        applicable = [
            chunk.actions[step - chunk.start_step]
            for chunk in self._chunks
            if chunk.start_step <= step <= chunk.end_step
        ]
        if not applicable:
            raise RuntimeError(
                f"No cached action chunk covers step {step}; add a prediction before requesting an action."
            )

        actions = np.stack(applicable, axis=0)
        weights = self._weights(len(applicable))
        return np.sum(actions * weights[:, None], axis=0, dtype=np.float32)

    # 便捷接口：在当前步记录新 chunk，再直接取得该步应执行的融合动作。
    def add_and_get_action(self, step: int, action_chunk: np.ndarray) -> np.ndarray:
        """Store the prediction from ``step`` and return the fused action for that step."""
        self.add_prediction(step, action_chunk)
        return self.get_action(step)

    # 供 workflow 日志和测试查看当前尚未过期的历史 chunk 数量。
    @property
    def num_cached_chunks(self) -> int:
        """Return the number of currently retained, not-yet-expired chunks."""
        return len(self._chunks)

    # 删除在 current_step 之前已经完全结束的 chunk；全零动作也照常保留和融合。
    def _discard_expired(self, *, current_step: int) -> None:
        self._chunks = [chunk for chunk in self._chunks if chunk.end_step >= current_step]

    # 根据原版 ACT 或“偏好最新预测”的可选语义，生成并归一化指数权重。
    def _weights(self, count: int) -> np.ndarray:
        if count <= 0:
            raise ValueError("count must be positive.")

        # 缓存按起始时间从早到晚排列。原版 ACT 对 [oldest,...,newest] 使用 exp(-decay*[0,...])。
        exponents = np.arange(count, dtype=np.float32)
        if self.prefer_recent:
            exponents = exponents[::-1]
        weights = np.exp(-self.decay * exponents)
        return weights / weights.sum()
