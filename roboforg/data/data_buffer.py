from __future__ import annotations

import collections
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

import jax
import numpy as np
from flax.core import frozen_dict

ArrayTree = np.ndarray | dict[str, "ArrayTree"]
Trajectory = dict[str, ArrayTree]

##############################################################################
#                                   tool                                     #
##############################################################################
#负责经验池管理的“基础框架”


################################################################
# 整理数据结构 


# 将多份结构相同的数据按字段堆叠成数组。
def _stack_tree(values: Sequence[Any]) -> ArrayTree:
    """将结构相同的数据沿第 0 维递归堆叠。"""
    if not values:
        raise ValueError("Cannot stack an empty sequence.")

    first = values[0]
    if isinstance(first, Mapping):
        expected_keys = tuple(first.keys())
        expected_key_set = set(expected_keys)

        # 所有数据必须具有相同的字典结构。
        for value in values[1:]:
            if not isinstance(value, Mapping):
                raise TypeError("All elements must have the same mapping structure.")
            if set(value.keys()) != expected_key_set:
                raise ValueError("All nested mappings must contain identical keys.")

        # 按第一份数据的字段顺序递归堆叠。
        return {key: _stack_tree([value[key] for value in values]) for key in expected_keys}

    return np.stack([np.asarray(value) for value in values], axis=0)


# 对嵌套字典中的所有数组，按相同索引取数据。
def _tree_take(tree: ArrayTree, indices: Any) -> ArrayTree:
    """使用相同索引提取数据，并保留原有嵌套结构。"""
    if isinstance(tree, np.ndarray):
        return tree[indices]

    if isinstance(tree, Mapping):
        # 对每个字段递归应用相同索引。
        return {key: _tree_take(value, indices) for key, value in tree.items()}

    raise TypeError(f"Unsupported tree value type: {type(tree).__name__}.")


# 将嵌套数据的非字典叶子转换为 NumPy 数组。
def _tree_asarray(tree: Any) -> ArrayTree:
    """递归保留嵌套字典结构，并将其他叶子整体转换成 NumPy 数组"""

    if isinstance(tree, Mapping):
        return {key: _tree_asarray(value) for key, value in tree.items()}

    # 字典继续递归处理，Python 列表则整体转换为 NumPy 数组。
    return np.asarray(tree)


# 检查所有数组第 0 维长度一致，并返回该长度。
def _tree_leading_length(tree: ArrayTree) -> int:
    lengths = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(lambda value: len(value), tree)
    )

    # 检查第 0 维长度是否一致
    if not lengths:
        raise ValueError("An array tree must contain at least one array.")
    if any(length != lengths[0] for length in lengths[1:]):
        raise ValueError(f"Array-tree leading dimensions disagree: {lengths}.")

    return int(lengths[0])


##############################################################################
#                            Basic DataBuffer                                #
##############################################################################

class DataBuffer:
    """数据缓冲区"""

    # 创建一个元组供后续插入数据时检查字段是否齐全
    REQUIRED_TRANSITION_KEYS = (
        "observations",
        "actions",
        "rewards",
        "dones",
        "next_observations",
    )

    def __init__(self, capacity: int, *, seed: int | None = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")

        self._capacity = int(capacity)                                               # transition 容量上限
        self._trajectories: collections.deque[Trajectory] = collections.deque()      # 完整轨迹队列
        self._num_transitions = 0                                                    # 已入池 transition 数
        self._current_episode: dict[str, list[Any]] | None = None                    # 未完成轨迹暂存区
        self._rng = np.random.default_rng(seed)                                      # 采样随机数生成器

    # 设置随机种子，让采样结果可复现。
    def seed(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    @classmethod
    # 从已有的单步经验序列创建并填充经验池。
    def from_transitions(
        cls,
        transitions: Sequence[Mapping[str, Any]],
        *,
        capacity: int | None = None,
        seed: int | None = None,
    ) -> DataBuffer:
        """从按时间排列的单步 transition 创建经验池。"""
        if not transitions:
            raise ValueError("transitions must not be empty.")

        # 默认容量刚好容纳全部输入数据。
        if capacity is None:
            capacity = len(transitions)
        if capacity < len(transitions):
            raise ValueError("capacity cannot be smaller than the number of transitions.")

        buffer = cls(capacity, seed=seed)
        # 按时间顺序插入，由 dones 划分完整轨迹。
        for transition in transitions:
            buffer.insert(transition)

        if buffer.has_incomplete_episode:
            raise ValueError("The final trajectory must end with dones=True.")

        return buffer

    @classmethod
    # 从已有的完整轨迹创建并填充经验池。
    def from_trajectories(
        cls,
        trajectories: Sequence[Mapping[str, Any]],
        *,
        capacity: int | None = None,
        seed: int | None = None,
    ) -> DataBuffer:
        """从多条完整轨迹创建经验池。"""
        if not trajectories:
            raise ValueError("trajectories must not be empty.")

        # 轨迹长度由其中包含的动作数量决定。
        total_transitions = sum(len(trajectory["actions"]) for trajectory in trajectories)
        if capacity is None:
            capacity = total_transitions
        if capacity < total_transitions:
            raise ValueError("capacity cannot be smaller than the number of transitions.")

        buffer = cls(capacity, seed=seed)
        # 将整理好的完整轨迹逐条加入经验池。
        for trajectory in trajectories:
            buffer.add_trajectory(trajectory)

        return buffer

#######################################################
# 经验池状态查询

    @property
    # 返回经验池的容量上限。
    def capacity(self) -> int:
        return self._capacity

    @property
    # 返回已保存的完整轨迹数量。
    def num_trajectories(self) -> int:
        return len(self._trajectories)

    @property
    # 判断是否有尚未结束的轨迹正在暂存。
    def has_incomplete_episode(self) -> bool:
        return self._current_episode is not None

    @property
    # 返回每条已保存轨迹的步数。
    def trajectory_lengths(self) -> np.ndarray:
        return np.asarray([len(trajectory["actions"]) for trajectory in self._trajectories], dtype=np.int64)

    # 返回已入池的 transition 总数。
    def __len__(self) -> int:
        """所有轨迹 transition 数目"""
        return self._num_transitions


########################################################
# 写入和管理轨迹，包括容量淘汰 判断轨迹类型

    # 插入一条 transition，轨迹结束时将整条轨迹入池。
    def insert(self, transition: Mapping[str, Any]) -> None:
        # 检查 transition 的必要字段是否齐全。
        missing = set(self.REQUIRED_TRANSITION_KEYS) - transition.keys()
        if missing:
            raise KeyError(f"Transition is missing required keys: {sorted(missing)}.")

        # 读取当前轨迹是否结束。
        done = bool(transition["dones"])

        # mask 用于表示下一状态能否参与后续的价值估计。
        if "masks" in transition:
            mask = np.float32(transition["masks"])
        elif "terminated" in transition:
            mask = np.float32(1.0 - float(transition["terminated"]))
        else:
            raise KeyError("Transition requires either 'masks' or 'terminated'.")

        # 第一条 transition 到来时，创建当前轨迹的暂存结构。
        if self._current_episode is None:
            self._current_episode = {
                "observations": [transition["observations"]],
                "actions": [],
                "rewards": [],
                "masks": [],
                "dones": [],
            }

        # 将当前 transition 追加到暂存轨迹。
        self._current_episode["actions"].append(transition["actions"])
        self._current_episode["rewards"].append(np.float32(transition["rewards"]))
        self._current_episode["masks"].append(mask)
        self._current_episode["dones"].append(np.bool_(done))
        self._current_episode["observations"].append(transition["next_observations"])

        # 轨迹结束后整理数据并加入经验池。
        if done:
            trajectory: Trajectory = {key: _stack_tree(values) for key, values in self._current_episode.items()}

            self.add_trajectory(trajectory)
            self._current_episode = None

    # 按顺序批量插入多条 transition。
    def extend(self, transitions: Iterable[Mapping[str, Any]]) -> None:
        #按时间顺序插入经验池

        for transition in transitions:
            self.insert(transition)

    # 丢弃当前暂存的未完成轨迹。
    def discard_incomplete_episode(self) -> None:
        #主动丢弃未完成的

        self._current_episode = None

    # 整理、检查并存入完整轨迹，容量不足时淘汰旧轨迹。
    def add_trajectory(self, trajectory: Mapping[str, Any]) -> None:
        # 提取训练所需字段，并将所有叶子统一转换为 NumPy 数组。
        compact_trajectory: Trajectory = {
            key: _tree_asarray(trajectory[key])
            for key in ("observations", "actions", "rewards", "masks", "dones")
        }

        # 合法性检查将在后续单独实现，并返回该轨迹的 transition 数量。
        trajectory_length = self._validate_trajectory(compact_trajectory)
        if trajectory_length > self._capacity:
            raise ValueError(
                f"Trajectory length {trajectory_length} exceeds buffer capacity "
                f"{self._capacity}; complete trajectories are never truncated."
            )

        # 容量不足时，从队首逐条删除最早加入的完整轨迹。
        while self._num_transitions + trajectory_length > self._capacity:
            removed = self._trajectories.popleft()
            self._num_transitions -= len(removed["actions"])

        # 加入新轨迹并更新已存 transition 总数。
        self._trajectories.append(compact_trajectory)
        self._num_transitions += trajectory_length


    # 检查轨迹字段、数组形状和结束标志，并返回步数。
    def _validate_trajectory(self, trajectory: Mapping[str, Any]) -> int:
        required_keys = {"observations", "actions", "rewards", "masks", "dones"}
        missing = required_keys - trajectory.keys()
        if missing:
            raise KeyError(f"Trajectory is missing required keys: {sorted(missing)}.")

        # 动作数量定义轨迹包含的 transition 数量。
        actions = trajectory["actions"]
        if not isinstance(actions, np.ndarray):
            raise TypeError("Trajectory actions must be a NumPy array.")
        if actions.ndim < 1:
            raise ValueError("Trajectory actions must have a leading time dimension.")

        transition_count = len(actions)
        if transition_count == 0:
            raise ValueError("Trajectory must contain at least one transition.")

        # 单步标量字段必须是一维数组，并与动作数量一致。
        for key in ("rewards", "masks", "dones"):
            value = trajectory[key]
            if not isinstance(value, np.ndarray):
                raise TypeError(f"Trajectory field {key!r} must be a NumPy array.")
            if value.shape != (transition_count,):
                raise ValueError(
                    f"Trajectory field {key!r} must have shape ({transition_count},), "
                    f"got {value.shape}."
                )

        # 所有观测叶子都必须是数组，并包含初始观测和每一步的下一观测。
        observation_leaves = jax.tree_util.tree_leaves(trajectory["observations"])
        if not observation_leaves:
            raise ValueError("Trajectory observations must contain at least one array.")
        if any(not isinstance(value, np.ndarray) for value in observation_leaves):
            raise TypeError("All trajectory observation leaves must be NumPy arrays.")
        if any(value.ndim < 1 for value in observation_leaves):
            raise ValueError("All trajectory observation leaves must have a leading time dimension.")

        observation_count = _tree_leading_length(trajectory["observations"])
        if observation_count != transition_count + 1:
            raise ValueError(
                f"Trajectory observations must have length {transition_count + 1}, "
                f"got {observation_count}."
            )

        # 一条完整轨迹只能在最后一步结束。
        dones = trajectory["dones"]
        if np.any(dones[:-1]):
            raise ValueError("Trajectory dones cannot be True before the final transition.")
        if not bool(dones[-1]):
            raise ValueError("The final trajectory transition must have dones=True.")

        return transition_count

##########################################################
# 数据读取与训练供给

    # 按索引获取一条完整轨迹。
    def get_trajectory(self, index: int) -> frozen_dict.FrozenDict:
        if not -len(self._trajectories) <= index < len(self._trajectories):
            raise IndexError("trajectory index out of range.")

        # frozen_dict.freeze(...) 将轨迹字典包装为不可修改的 FrozenDict, 防止意外修改
        return frozen_dict.freeze(self._trajectories[index])

    # 随机抽取指定数量的完整轨迹。
    def sample_trajectories(self, batch_size: int) -> tuple[frozen_dict.FrozenDict, ...]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not self._trajectories:
            raise ValueError("Cannot sample from an empty buffer.")

        # 有放回地随机选择完整轨迹。
        indices = self._rng.integers(0, len(self._trajectories), size=batch_size)
        return tuple(self.get_trajectory(int(index)) for index in indices)

    # 从已保存轨迹中均匀随机抽取单步经验，组成训练 batch。
    def sample(self, batch_size: int) -> frozen_dict.FrozenDict:
        """从所有轨迹中均匀随机采样 batch_size 条 transition"""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if len(self) == 0:
            raise ValueError("Cannot sample from an empty buffer.")

        # 随机生成全局 transition 下标
        global_indices = self._rng.integers(0, self._num_transitions, size=batch_size)

        """
        计算每条轨迹结束时对应的全局 transition 下标边界, 例如
        
        1. self.trajectory_lengths = [3, 5, 2]
        2. cumulative_lengths = [3, 8, 10]
        """
        cumulative_lengths = np.cumsum(self.trajectory_lengths)

        samples: dict[str, list[Any]] = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "masks": [],
            "dones": [],
            "next_observations": [],
        }

        # 依次处理每个随机采样的全局 transition 下标
        for global_index in global_indices:
            # 根据轨迹累计长度，确定该 transition 属于第 i 条轨迹
            trajectory_index = int(np.searchsorted(cumulative_lengths, global_index, side="right"))

            # 获取当前轨迹在全局 transition 序列中起始位置
            previous_end = 0 if trajectory_index == 0 else cumulative_lengths[trajectory_index - 1]

            # 将全局下标转为轨迹内部下标
            step_index = int(global_index - previous_end)

            # 取出完整的第 i 条轨迹
            trajectory = self._trajectories[trajectory_index]

            # 存入 “observation，next_observation, ....”
            samples["observations"].append(_tree_take(trajectory["observations"], step_index))
            samples["next_observations"].append(_tree_take(trajectory["observations"], step_index + 1))
            for key in ("actions", "rewards", "masks", "dones"):
                samples[key].append(_tree_take(trajectory[key], step_index))

        """
        通过 _stack_tree(...) 将 samples 中每个字段的样本列表堆叠成 Batch 数组, 即 

        {
            "observations": {
                                "state": [B, state_dim],
                                "front": [B, H, W, C],
                            }
            "actions": ...,  # [B, A]
            "rewards": ...,  # [B]
            ...
        }
        """
        return frozen_dict.freeze({key: _stack_tree(values) for key, values in samples.items()})

    # 持续提供训练 batch，并预先传到 JAX 设备。
    def get_iterator(
        self,
        *,
        sample_args: dict[str, Any],        # 例如 sample_args={"batch_size": 256}
        queue_size: int = 2,                # 预先准备多少 batch
        device=None,
    ) -> Iterator:                          #表示返回一个迭代器，通过next（）逐次获得batch
        """创建一个持续产生训练 Batch 的 "迭代器"，并提前把数据传到 JAX 设备"""

        # 创建双端队列，保存已经准备好的 Batch
        queue = collections.deque()

        # 准备指定数量的 batch，传到 JAX 设备并加入预取队列。
        def enqueue(count: int) -> None:
            for _ in range(count):
                queue.append(jax.device_put(self.sample(**sample_args), device=device))

        # 生成器
        enqueue(queue_size)                 # 预先取 queue_size 个 Batch
        while queue:
            yield queue.popleft()           # yield 暂停函数，下次调用从此处继续
            enqueue(1)                      # 每取一个就补充一个，生成器会一直运行，直到外部停止迭代