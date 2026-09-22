from collections.abc import Mapping, Sequence
from typing import Any

import jax
import numpy as np

Array = np.ndarray | jax.Array


class ACTBatchConverter:
    """将 ChunkBCBuffer 的单帧 batch 整理为 ACT 使用的数据格式。"""

    # 保存动作 chunk 长度和需要处理的相机字段。
    def __init__(
        self,
        action_horizon: int,
        image_keys: Sequence[str] = ("front", "wrist"),
    ) -> None:
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive.")
        self.action_horizon = int(action_horizon)
        self.image_keys = tuple(image_keys)

    # 统一转换一个 ChunkBCBuffer 训练 batch。
    def __call__(self, chunk_batch: Mapping[str, Any]) -> tuple[dict[str, Array], Array]:
        """检查并转换一个训练 batch。"""
        # 检查 batch 顶层字段。
        missing = {"observations", "actions"} - chunk_batch.keys()
        if missing:
            raise KeyError(f"Chunk batch is missing fields: {sorted(missing)}.")

        # 读取观测字典。
        observations = chunk_batch["observations"]
        if not isinstance(observations, Mapping):
            raise TypeError("observations must be a mapping.")

        # 动作保持 [B, k, 7]，并以它确定统一的 batch 大小。
        actions, batch_size = self._convert_actions(chunk_batch["actions"])
        converted_observations = {
            "state": self._convert_state(observations, batch_size),
            **self._convert_images(observations, batch_size),
        }

        # 返回 ACT 使用的观测字典和目标动作 chunk。
        return converted_observations, actions

    # 检查目标动作 chunk 的形状，并获取 batch 大小。
    def _convert_actions(self, actions: Array) -> tuple[Array, int]:
        """检查动作 chunk 的长度和七维动作结构。"""
        
        expected_tail = (self.action_horizon, 7)
        if actions.ndim != 3 or actions.shape[1:] != expected_tail or actions.shape[0] <= 0:
            raise ValueError(f"actions must have shape [B, {self.action_horizon}, 7] with B > 0.")
        return actions, actions.shape[0]

    @staticmethod
    # 去掉状态观测中长度为 1 的时间维。
    def _convert_state(observations: Mapping[str, Array], batch_size: int) -> Array:
        """将单帧状态从 [B,1,S] 整理为 [B,S]。"""

        # 检查状态字段及其 batch、时间维。
        if "state" not in observations:
            raise KeyError("Observations are missing field: 'state'.")
        state = observations["state"]
        if state.ndim != 3 or state.shape[0] != batch_size or state.shape[1] != 1:
            raise ValueError("state must have shape [B, 1, S] and match the action batch size.")
        return state[:, 0]

    # 去掉每路图像中长度为 1 的时间维。
    def _convert_images(
        self,
        observations: Mapping[str, Array],
        batch_size: int,
    ) -> dict[str, Array]:
        """将各相机图像从 [B,1,H,W,3] 整理为 [B,H,W,3]。"""

        converted = {}

        # 逐路检查相机字段、batch、时间维和 RGB 通道。
        for key in self.image_keys:
            if key not in observations:
                raise KeyError(f"Observations are missing image field: {key!r}.")
            image = observations[key]
            if image.ndim != 5 or image.shape[0] != batch_size or image.shape[1] != 1 or image.shape[-1] != 3:
                raise ValueError(
                    f"Image {key!r} must have shape [B, 1, H, W, 3] "
                    "and match the action batch size."
                )
            converted[key] = image[:, 0]

        return converted
