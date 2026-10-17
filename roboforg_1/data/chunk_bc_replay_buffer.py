"""BC chunk 采样：从 DataStore 读取轨迹，组织历史观测、动作块及有效标记。

observation_horizon 控制输入观测帧数；action_horizon 控制目标动作数，模型输出应使用同一个配置。
执行长度由运行脚本控制，不参与采样；这里返回 NumPy batch，设备传输由训练流程负责。
"""

from __future__ import annotations

from collections.abc import Iterator
from numbers import Integral
from typing import Any

import numpy as np

from roboforg_1.data.data_store import DataStore, Trajectory
from roboforg_1.data.utils import stack_tree, tree_leading_length, tree_take


##############################################################################
#                           Chunk BC Replay Buffer                           #
##############################################################################

class ChunkBCReplayBuffer:
    """按当前时刻 t 对齐历史观测和未来动作，所有切片均限制在同一条轨迹内。

    返回的 batch 结构：
        {
            "observations": ...,   # 每个观测叶子为 [B, observation_horizon, ...]
            "actions": ...,        # [B, action_horizon,  action_dim]，保留时间维
            "action_valid": ...,   # [B, action_horizon]， bool，True 表示真实动作
        }

    action_valid 是动作填充标记，与轨迹中用于 RL bootstrap 的 masks 不同。

    算法层必须用它排除填充位置的损失，并检查网络内部的时间维交互。
    例如误差：  [0.2, 0.4, 1.8, 2.1]
              action_valid：[ 1,   1,   0,   0 ]
              最终损失：   (0.2 + 0.4) / 2 = 0.3

    本类只读取 Store；使用期间不要原地修改已存轨迹，或并发写入 Store。
    """

    # --------------------------- 初始化与状态查询 --------------------------- #

    def __init__(self, store: DataStore, *, seed: int | None = None) -> None:
        """接收轨迹存储和采样种子，不复制整份数据，也不提前展开全部 chunk。"""
        self._store = store
        self._rng = np.random.default_rng(seed)

    @property
    def store(self) -> DataStore:
        """返回轨迹存储后端。"""
        return self._store

    @property
    def num_trajectories(self) -> int:
        """返回完整轨迹数量。"""
        return self._store.num_trajectories

    @property
    def trajectory_lengths(self) -> np.ndarray:
        """返回各条轨迹的真实动作数，也就是各自可采样的起点数。"""
        return self._store.trajectory_lengths

    def __len__(self) -> int:
        """返回全部可采样起点数；启用尾部填充后，它等于 transition 总数。"""
        return self._store.num_transitions

    def seed(self, seed: int | None = None) -> None:
        """重置随机数生成器，同时影响随机采样和之后创建的 epoch 顺序。"""
        self._rng = np.random.default_rng(seed)

    # --------------------------- 随机有放回采样 --------------------------- #

    def sample_chunk(self, batch_size: int, *, observation_horizon: int, action_horizon: int) -> dict[str, Any]:
        """从所有真实动作起点中等概率抽取一个 batch，同一起点可能重复出现。"""
        self._validate_sample_args(batch_size, observation_horizon, action_horizon)
        cumulative_counts = self._cumulative_start_counts()

        # 按全局起点均匀采样，长轨迹因起点更多而提供更多样本。
        global_starts = self._rng.integers(0, int(cumulative_counts[-1]), size=batch_size)
        return self._build_batch(global_starts, cumulative_counts, observation_horizon, action_horizon)

    def get_chunk_iterator(self, *, batch_size: int, observation_horizon: int, action_horizon: int) -> Iterator[dict[str, Any]]:
        """持续提供随机 batch；每次 next() 同步采样，调用方控制总训练步数。"""
        while True:
            yield self.sample_chunk(batch_size, observation_horizon=observation_horizon, action_horizon=action_horizon)

    # --------------------------- 一轮无放回遍历 --------------------------- #

    def get_epoch_iterator(
        self,
        *,
        batch_size: int,
        observation_horizon: int,
        action_horizon: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """遍历一轮起点，默认打乱且保留最后不足一批的样本，适用于离线 BC。

        drop_last=False 时每个起点恰好出现一次；True 时丢弃不足一批的剩余起点。
        第一次 next() 时确定本轮起点集合，此后追加的轨迹从下一轮开始使用。
        """
        self._validate_sample_args(batch_size, observation_horizon, action_horizon)
        if not isinstance(shuffle, bool) or not isinstance(drop_last, bool):
            raise TypeError("shuffle and drop_last must be booleans.")

        cumulative_counts = self._cumulative_start_counts()
        total_starts = int(cumulative_counts[-1])
        global_starts = np.arange(total_starts, dtype=np.int64)
        if shuffle:
            self._rng.shuffle(global_starts)

        # 只丢弃不完整的 batch；与单条样本内部的动作尾部填充无关。
        epoch_size = total_starts - total_starts % batch_size if drop_last else total_starts
        for batch_start in range(0, epoch_size, batch_size):
            selected = global_starts[batch_start : min(batch_start + batch_size, epoch_size)]
            yield self._build_batch(selected, cumulative_counts, observation_horizon, action_horizon)

    # --------------------------- 起点索引与 batch 组装 --------------------------- #

    def _cumulative_start_counts(self) -> np.ndarray:
        """建立全局起点索引；长度为 T 的轨迹贡献 t=0,...,T-1 共 T 个起点。"""
        cumulative_counts = np.cumsum(self.trajectory_lengths, dtype=np.int64)
        if cumulative_counts.size == 0 or cumulative_counts[-1] == 0:
            raise ValueError("Cannot sample chunks from an empty DataStore.")
        return cumulative_counts

    def _build_batch(
        self,
        global_starts: np.ndarray,
        cumulative_counts: np.ndarray,
        observation_horizon: int,
        action_horizon: int,
    ) -> dict[str, Any]:
        """将全局编号映射为轨迹及内部时间步，两种采样方式共用此方法。"""
        samples: dict[str, list[Any]] = {"observations": [], "actions": [], "action_valid": []}
        
        for global_start in global_starts:
            # 例如轨迹长度为 [3, 2]，累计数为 [3, 5]，全局编号 3 对应第 1 条轨迹的 t=0。
            trajectory_index = int(np.searchsorted(cumulative_counts, global_start, side="right"))
            previous_count = 0 if trajectory_index == 0 else int(cumulative_counts[trajectory_index - 1])
            start_step = int(global_start - previous_count)
            trajectory = self._store.get_trajectory(trajectory_index)
            sample = self._build_chunk_sample(trajectory, start_step, observation_horizon, action_horizon)
            for key, value in sample.items():
                samples[key].append(value)

        # 为每个叶子新增 batch 维 B，保留 observations 内部的任意嵌套结构和 dtype。
        return {key: stack_tree(values) for key, values in samples.items()}

    # --------------------------- 单条样本：历史、动作与填充 --------------------------- #

    @staticmethod
    def _build_chunk_sample(trajectory: Trajectory, start_step: int, observation_horizon: int, action_horizon: int) -> dict[str, Any]:
        """以真实动作时刻 t 为起点；末帧 o_T 没有对应动作，不能单独作为起点。"""
        trajectory_length = len(trajectory["actions"])
        if not 0 <= start_step < trajectory_length:
            raise IndexError("Chunk start must correspond to a real action in this trajectory.")

        # 1. 观测按从旧到新排列并以 o_t 结尾；历史不足时重复 o_0。
        #    例如 t=1、observation_horizon=4，下标为 [0, 0, 0, 1]。
        observation_indices = np.arange(start_step - observation_horizon + 1, start_step + 1, dtype=np.int64)
        observation_indices = np.maximum(observation_indices, 0)

        # 2. 动作从 a_t 开始；先标记真实位置，再将越界下标截到最后一个动作。
        #    例如 T=5、t=3、action_horizon=4：下标 [3, 4, 4, 4]，有效标记 [1, 1, 0, 0]。
        action_indices = np.arange(start_step, start_step + action_horizon, dtype=np.int64)
        action_valid = action_indices < trajectory_length
        action_indices = np.minimum(action_indices, trajectory_length - 1)

        # 3. 填充仅用于对齐形状，不是新示范；每条样本至少有一个真实动作。
        #    损失可先对动作维求均值，再按 action_valid 加权并除以该样本的有效动作数。
        return {
            "observations": tree_take(trajectory["observations"], observation_indices),
            "actions": tree_take(trajectory["actions"], action_indices),
            "action_valid": action_valid,
        }

    #####################################
    #        critic数据处理和调用         #
    #####################################

    @staticmethod
    def build_chunk_transition(
        trajectory: Trajectory,
        start_step: int,
        observation_horizon: int,
        action_horizon: int,
        discount: float,
    ) -> dict[str, Any]:
        """从一条轨迹构造完整 H 步 transition，不采样、不拼 batch、不填充动作。

        输入：观测叶子为 [L+1,...]，动作为 [L,A]，奖励与 masks 为 [L]。
        输出：动作展平为 [H*A]    ，奖励与 mask 为 float32 标量；

        当前和下一历史窗口分别以 o_t 和 o_(t+H) 结尾。
        masks 表示能否 bootstrap，与 BC 的 action_valid 不同。
        """
        ChunkBCReplayBuffer._validate_sample_args(1, observation_horizon, action_horizon)
        if isinstance(start_step, bool) or not isinstance(start_step, Integral):
            raise TypeError("start_step must be an integer.")
        if not 0.0 <= discount <= 1.0:
            raise ValueError("discount must lie in [0, 1].")
        for key in ("observations", "actions", "rewards", "masks"):
            if key not in trajectory:
                raise KeyError(f"Trajectory requires {key!r}.")

        actions = np.asarray(trajectory["actions"])
        if actions.ndim != 2 or min(actions.shape) <= 0:
            raise ValueError("Trajectory actions must have non-empty shape [L, action_dim].")
        if not np.issubdtype(actions.dtype, np.floating):
            raise TypeError("Trajectory actions must have a floating dtype.")
        length = len(actions)
        if tree_leading_length(trajectory["observations"]) != length + 1:
            raise ValueError("Trajectory must contain L+1 observations for L actions.")
        next_step = start_step + action_horizon
        if start_step < 0 or next_step > length:
            raise IndexError("The requested complete action chunk exceeds the trajectory.")

        rewards = np.asarray(trajectory["rewards"])
        masks = np.asarray(trajectory["masks"])
        if rewards.shape != (length,) or masks.shape != (length,):
            raise ValueError("Trajectory rewards and masks must have shape [L].")
        if not np.all(np.isfinite(rewards)):
            raise ValueError("Trajectory rewards must be finite.")
        if not np.all(np.isin(masks, (0, 1))):
            raise ValueError("Trajectory masks must contain only 0 or 1.")
        if np.any(masks[start_step : next_step - 1] == 0):
            raise ValueError("An action chunk must not cross a terminal transition.")
        if "dones" in trajectory:
            dones = np.asarray(trajectory["dones"])
            if dones.shape != (length,) or not np.all(np.isin(dones, (0, 1))):
                raise ValueError("Trajectory dones must have shape [L] and contain only 0 or 1.")
            if np.any(dones[start_step : next_step - 1]):
                raise ValueError("An action chunk must not cross an episode boundary.")

        # 历史不足时重复 o_0；动作始终取真实的完整 H 步。
        observation_indices = np.maximum(
            np.arange(start_step - observation_horizon + 1, start_step + 1, dtype=np.int64), 0
        )
        next_observation_indices = np.maximum(
            np.arange(next_step - observation_horizon + 1, next_step + 1, dtype=np.int64), 0
        )
        discount_powers = np.power(float(discount), np.arange(action_horizon))
        return {
            "observations": tree_take(trajectory["observations"], observation_indices),
            "actions": actions[start_step:next_step].reshape(-1),
            "rewards": np.asarray(np.sum(discount_powers * rewards[start_step:next_step]), dtype=np.float32),
            "next_observations": tree_take(trajectory["observations"], next_observation_indices),
            "masks": np.asarray(np.prod(masks[start_step:next_step]), dtype=np.float32),
        }

    def sample_fql_chunk(
        self,
        batch_size: int,
        *,
        observation_horizon: int,
        action_horizon: int,
        discount: float,
    ) -> dict[str, Any]:
        """均匀有放回采样完整动作块，构造 FQL 所需的 NumPy batch。

        输出 observations、next_observations 的叶子为 [B,T,...]，
        actions 为 [B,H*A]，rewards 和 masks 为 [B]。
        不足 H 步的尾部起点不参与采样；discount 必须与算法一致。
        """
        self._validate_sample_args(batch_size, observation_horizon, action_horizon)
        if not 0.0 <= discount <= 1.0:
            raise ValueError("discount must lie in [0, 1].")

        # 长度 L 的轨迹可提供 max(L-H+1, 0) 个完整动作块。
        valid_counts = np.maximum(self.trajectory_lengths - action_horizon + 1, 0)
        cumulative_counts = np.cumsum(valid_counts, dtype=np.int64)
        if cumulative_counts.size == 0 or cumulative_counts[-1] == 0:
            raise ValueError("No trajectory contains a complete action chunk for this action_horizon.")
        global_starts = self._rng.integers(0, int(cumulative_counts[-1]), size=batch_size)
        samples: dict[str, list[Any]] = {
            "observations": [], "actions": [], "rewards": [],
            "next_observations": [], "masks": [],
        }
        for global_start in global_starts:
            trajectory_index = int(np.searchsorted(cumulative_counts, global_start, side="right"))
            previous_count = 0 if trajectory_index == 0 else int(cumulative_counts[trajectory_index - 1])
            start_step = int(global_start - previous_count)
            sample = self.build_chunk_transition(
                self._store.get_trajectory(trajectory_index),
                start_step,
                observation_horizon,
                action_horizon,
                discount,
            )
            for key, value in sample.items():
                samples[key].append(value)
        return {key: stack_tree(values) for key, values in samples.items()}

    # --------------------------- 公共采样参数检查 --------------------------- #

    @staticmethod
    def _validate_sample_args(batch_size: int, observation_horizon: int, action_horizon: int) -> None:
        """三个长度都要求正整数；允许历史长度或动作长度超过轨迹长度。"""
        for name, value in (("batch_size", batch_size), ("observation_horizon", observation_horizon), ("action_horizon", action_horizon)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")


__all__ = ["ChunkBCReplayBuffer"]
