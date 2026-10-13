"""基础内存轨迹存储：接收完整轨迹，校验后供采样器按编号读取。

数据流：load_dataset → InMemoryDataStore.from_trajectories → chunk 采样器。
容量按 transition 步数计算，容量不足直接报错，已导入轨迹不会被淘汰。
本模块不负责文件读写、随机采样、chunk 切分或设备传输。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from numbers import Integral
from typing import Any, Protocol

import numpy as np

from roboforg_1.data.utils import tree_asarray, tree_leading_length


Trajectory = dict[str, Any]

"""
读取一条轨迹的数据格式为：
{
    "observations": {
        "state": ...,         # [T+1, 23]
        "images": {
            "front": ...,     # [T+1, 128, 128, 3]
            "wrist": ...,     # [T+1, 128, 128, 3]
        },
    },
    "actions": ...,           # [T, 7]
    "rewards": ...,           # [T]
    "masks": ...,             # [T]
    "dones": ...,             # [T]，仅最后一步为 True
}
"""


# ======================================================================================= #
#                                    DataStore 统一读取协议                                  #
# ======================================================================================= #

class DataStore(Protocol):
    """规定采样器需要的轨迹访问接口，具体存储类只需提供这些属性和方法。"""

    @property
    def num_trajectories(self) -> int:
        """返回当前保存的完整轨迹数量。"""
        ...

    @property
    def num_transitions(self) -> int:
        """返回所有轨迹包含的 transition 总步数。"""
        ...

    @property
    def trajectory_lengths(self) -> np.ndarray:
        """按轨迹编号顺序返回各条轨迹的步数。"""
        ...

    def get_trajectory(self, index: int) -> Trajectory:
        """根据编号读取一条完整轨迹。"""
        ...


# ======================================================================================= #
#                                InMemoryDataStore 内存轨迹存储                              #
# ======================================================================================= #

class InMemoryDataStore:
    """将完整轨迹常驻 RAM，适用于当前离线 BC 数据集的单线程读取。

    每条轨迹包含 observations、actions、rewards、masks、dones 五个训练字段。
    观测可以是任意嵌套数组树，不限制状态维度、相机名称或图像尺寸。
    同一 Store 内各轨迹的字段结构、非时间维形状和 dtype 必须一致，轨迹长度可不同。
    数组不做额外复制；导入后调用方和采样器都应避免原地修改轨迹数据。
    """

    REQUIRED_TRAJECTORY_KEYS = ("observations", "actions", "rewards", "masks", "dones")

    # ----------------------------------- 配置与初始化 ----------------------------------- #

    def __init__(self, capacity: int) -> None:
        """创建空存储；capacity 是可保存的 transition 总步数上限。"""
        if isinstance(capacity, bool) or not isinstance(capacity, Integral) or capacity <= 0:
            raise ValueError("capacity must be a positive integer.")
        self._capacity = int(capacity)
        self._trajectories: list[Trajectory] = []
        self._num_transitions = 0

    # ----------------------------------- 从已有完整轨迹创建 Store ----------------------------------- #

    @classmethod
    def from_trajectories(cls, trajectories: Iterable[Mapping[str, Any]], *, capacity: int | None = None) -> InMemoryDataStore:
        """一次导入完整轨迹集合，默认容量等于全部轨迹的总步数。"""
        prepared = [cls._prepare_trajectory(trajectory) for trajectory in trajectories]
        if not prepared:
            raise ValueError("trajectories must not be empty.")

        # 导入前先校验全部轨迹，显式容量不足时拒绝整个导入过程。
        total_transitions = sum(length for _, length in prepared)
        store = cls(total_transitions if capacity is None else capacity)
        if total_transitions > store.capacity:
            raise ValueError(f"Dataset contains {total_transitions} transitions, exceeding capacity {store.capacity}.")

        for trajectory, length in prepared:
            store._add_prepared_trajectory(trajectory, length)
        return store

    # ----------------------------------- 存储状态与读取接口 ----------------------------------- #

    @property
    def capacity(self) -> int:
        """返回 transition 容量上限，不代表 RAM 字节数。"""
        return self._capacity

    @property
    def num_trajectories(self) -> int:
        """返回已导入的完整轨迹数量。"""
        return len(self._trajectories)

    @property
    def num_transitions(self) -> int:
        """返回已导入轨迹包含的 transition 总步数。"""
        return self._num_transitions

    @property
    def trajectory_lengths(self) -> np.ndarray:
        """按编号顺序返回各条轨迹的步数，供采样器建立索引。"""
        return np.asarray([len(trajectory["actions"]) for trajectory in self._trajectories], dtype=np.int64)

    def __len__(self) -> int:
        """返回 transition 总步数，与 num_transitions 一致。"""
        return self._num_transitions

    def get_trajectory(self, index: int) -> Trajectory:
        """返回存储中的轨迹引用，支持负索引；调用方应只读使用。"""
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("trajectory index must be an integer.")
        if not -len(self._trajectories) <= index < len(self._trajectories):
            raise IndexError("trajectory index out of range.")
        return self._trajectories[int(index)]

    # ----------------------------------- 完整轨迹写入与容量检查 ----------------------------------- #

    def add_trajectory(self, trajectory: Mapping[str, Any]) -> None:
        """整理并校验一条完整轨迹，容量不足时拒绝写入。"""
        compact, length = self._prepare_trajectory(trajectory)
        self._add_prepared_trajectory(compact, length)

    def _add_prepared_trajectory(self, trajectory: Trajectory, length: int) -> None:
        """存入已校验轨迹，失败时不改变已有轨迹和步数计数。"""
        if self._num_transitions + length > self.capacity:
            raise ValueError(f"Adding {length} transitions to {self._num_transitions} stored transitions exceeds capacity {self.capacity}.")
        # 以第一条轨迹为结构基准，提前拒绝无法共同组装 batch 的数据。
        if self._trajectories:
            self._validate_compatible_tree(trajectory, self._trajectories[0])
        self._trajectories.append(trajectory)
        self._num_transitions += length

    # ----------------------------------- 轨迹数组整理与结构校验 ----------------------------------- #

    @staticmethod
    def _validate_compatible_tree(value: Any, reference: Any, path: str = "trajectory") -> None:
        """递归比较结构、非时间维形状和 dtype，不扫描或复制数组内容。"""
        if isinstance(reference, Mapping):
            if not isinstance(value, Mapping) or value.keys() != reference.keys():
                raise ValueError(f"Trajectory schema mismatch at {path}: dictionary keys differ.")
            for key in reference:
                InMemoryDataStore._validate_compatible_tree(value[key], reference[key], f"{path}/{key}")
            return

        # 时间维 T 可以变化，状态维度、图像尺寸和动作维度等必须保持一致。
        if not isinstance(value, np.ndarray) or value.shape[1:] != reference.shape[1:] or value.dtype != reference.dtype:
            raise ValueError(f"Trajectory schema mismatch at {path}: expected shape [T, {reference.shape[1:]}] and dtype {reference.dtype}.")

    @classmethod
    def _prepare_trajectory(cls, trajectory: Mapping[str, Any]) -> tuple[Trajectory, int]:
        """提取五个训练字段并转为数组，保留图像等数组的原有 dtype。"""
        if not isinstance(trajectory, Mapping):
            raise TypeError("trajectory must be a mapping.")
        missing = set(cls.REQUIRED_TRAJECTORY_KEYS) - trajectory.keys()
        if missing:
            raise KeyError(f"Trajectory is missing required keys: {sorted(missing)}.")

        # 不在此处做图像归一化，也不预先构造 chunk。
        compact = {key: tree_asarray(trajectory[key]) for key in cls.REQUIRED_TRAJECTORY_KEYS}
        length = cls._validate_trajectory(compact)
        return compact, length

    @staticmethod
    def _validate_trajectory(trajectory: Mapping[str, Any]) -> int:
        """检查动作、观测和单步标记的时间维及结束位置，返回轨迹步数 T。"""
        actions = trajectory["actions"]
        if not isinstance(actions, np.ndarray) or actions.ndim != 2 or min(actions.shape) <= 0:
            raise ValueError("Trajectory actions must have nonempty shape [T, action_dim].")
        length = len(actions)

        # observations 保存 o_0 到 o_T，比动作 a_0 到 a_(T-1) 多一帧。
        observation_count = tree_leading_length(trajectory["observations"])
        if observation_count != length + 1:
            raise ValueError(f"Expected {length + 1} observations for {length} actions, got {observation_count}.")

        # 标量字段保持 [T]，拒绝 [T,1]，避免后续累计奖励计算发生错误广播。
        for key in ("rewards", "masks", "dones"):
            value = trajectory[key]
            if not isinstance(value, np.ndarray) or value.shape != (length,):
                raise ValueError(f"Trajectory field {key!r} must have shape ({length},).")

        # mask 表示能否 bootstrap，done 表示 episode 边界，两者不要求互为取反。
        for key in ("masks", "dones"):
            if not np.all(np.isin(trajectory[key], (0, 1))):
                raise ValueError(f"Trajectory field {key!r} must contain only 0 or 1.")
        dones = trajectory["dones"]
        if not bool(dones[-1]) or np.any(dones[:-1]):
            raise ValueError("Only the final transition must have dones=True.")
        return length


__all__ = ["Trajectory", "DataStore", "InMemoryDataStore"]
