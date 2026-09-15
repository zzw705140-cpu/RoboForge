from __future__ import annotations

import collections
from collections.abc import Iterator
from typing import Any

import jax
import numpy as np
from flax.core import frozen_dict

from roboforg.data.data_buffer import DataBuffer, Trajectory, _stack_tree, _tree_take

##############################################################################
#                              Chunk BC Buffer                               #
##############################################################################
#模仿学习用的数据缓冲区

class ChunkBCBuffer(DataBuffer):
    """为行为克隆算法构造历史观测和连续 action chunk。"""

    # 只计算每条轨迹能取出完整动作块的有效起点数量，轨迹太短时返回 0。
    def _num_valid_chunks_per_trajectory(self, action_horizon: int) -> np.ndarray:
        """计算每条轨迹能够产生多少合法 action chunk 起点"""
        return np.maximum(self.trajectory_lengths - action_horizon + 1, 0)

    # 从所有有效起点中随机抽样，组成包含观测和连续动作块的训练 batch。
    # 
    def sample_chunk(
        self,
        batch_size: int,
        *,
        observation_horizon: int,   #每个样本取几帧观测，以选中的时刻为结尾
        action_horizon: int,        #每个样本取几个连续的动作，从选中的时刻开始
    ) -> frozen_dict.FrozenDict:
        """从所有合法 chunk 起点中均匀随机采样一个训练 batch, 返回结构：
        ```
            {
                "observations": ...,  # [B, observation_horizon, ...]
                "actions": ...,       # [B, action_horizon, action_dim]
            }
        ```
        当轨迹开头没有足够的历史观测时，使用该轨迹的第 0 帧进行左侧填充。动作 chunk 不进行尾部填充，只采样能够容纳完整 chunk 的起点
        """

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if observation_horizon <= 0:
            raise ValueError("observation_horizon must be positive.")
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive.")

        # 每个合法起点被等概率采样；长轨迹因为包含更多合法起点，会自然提供更多样本
        valid_counts = self._num_valid_chunks_per_trajectory(action_horizon)
        total_valid_starts = int(valid_counts.sum())
        if total_valid_starts == 0:
            raise ValueError(f"No stored trajectory is long enough for the requested action_horizon={action_horizon}.")

        # 随机生成全局 transition 下标
        global_starts = self._rng.integers(0, total_valid_starts, size=batch_size)

        """
        计算每条轨迹合法 chunk 数量的累计和, 例如

        1. valid_counts = [3, 0, 6]
        2. cumulative_counts = [3, 3, 9]
        """
        cumulative_counts = np.cumsum(valid_counts)

        return self._build_batch_from_global_starts(
            global_starts,                                 #随机生成的起点编号
            cumulative_counts=cumulative_counts,           #每条轨迹有效起点数的累计结果
            observation_horizon=observation_horizon,       #每个样本取多少帧的历史观测
            action_horizon=action_horizon,                 #每个样本取多少个连续的动作
        )

    # 将全局起点编号换算为轨迹和时间步，构造各样本并堆叠成 batch。
    def _build_batch_from_global_starts(
        self,
        global_starts: np.ndarray,
        *,
        cumulative_counts: np.ndarray,
        observation_horizon: int,
        action_horizon: int,
    ) -> frozen_dict.FrozenDict:
        """根据全局合法起点下标构造一个 batch"""

        if len(global_starts) == 0:
            raise ValueError("Cannot build a batch from zero chunk starts.")

        samples: dict[str, list[Any]] = {
            "observations": [],
            "actions": [],
        }

        for global_start in global_starts:
            # 将所有轨迹的合法起点视为一个连续的一维区间，再映射回具体轨迹及轨迹内部的起点
            trajectory_index = int(np.searchsorted(cumulative_counts, global_start, side="right"))

            # 获取当前轨迹在全局 transition 序列中起始位置
            previous_count = (0 if trajectory_index == 0 else int(cumulative_counts[trajectory_index - 1]))

            # 将全局下标转为轨迹内部下标
            start_step = int(global_start - previous_count)

            # 取出完整的第 i 条轨迹
            trajectory = self._trajectories[trajectory_index]

            sample = self._build_chunk_sample(
                trajectory,                                #选中的这一条完整的轨迹
                start_step=start_step,                     #动作chunk的起点
                observation_horizon=observation_horizon,   #向前取多少帧历史观测
                action_horizon=action_horizon,             #从起点开始，向后取多少连续动作
            )
            #将单个样本的数据分别放入总列表
            for key, value in sample.items():
                samples[key].append(value)

        return frozen_dict.freeze({key: _stack_tree(values) for key, values in samples.items()})

    @staticmethod
    # 从指定起点取出历史观测和连续动作块，历史不足时重复首帧。
    #######################
    #    和流匹配相关的     #
    #######################
    def _build_chunk_sample(
        trajectory: Trajectory,
        *,
        start_step: int,
        observation_horizon: int,
        action_horizon: int,
    ) -> dict[str, Any]:
        """从一条轨迹的指定起点构造一条行为克隆训练样本。"""

        trajectory_length = len(trajectory["actions"])
        if start_step < 0 or start_step + action_horizon > trajectory_length:
            raise IndexError("The requested action chunk exceeds the trajectory.")

        # 例如 start_step=1、observation_horizon=4 时，下标为 [0, 0, 0, 1]。第 0 帧左侧填充保证所有样本具有相同的时间长度
        observation_indices = np.arange(
            start_step - observation_horizon + 1,
            start_step + 1,
            dtype=np.int64,
        )
        # 和零比较，取较大的那个例如（-1，0，1）变为（0，0，1）
        observation_indices = np.maximum(observation_indices, 0)

        """
        生成 start_step 开始、长度为 action_horizon 的连续动作下标，例如

        start_step = 2
        action_horizon = 4
        
        则得到 action_indices = [2, 3, 4, 5]
        """
        action_indices = np.arange(
            start_step,
            start_step + action_horizon,
            dtype=np.int64,
        )
        #_tree_take从整条轨迹中取出对应下标的动作，然后转换成np数组
        actions = np.asarray(_tree_take(trajectory["actions"], action_indices))

        return {
            "observations": _tree_take(trajectory["observations"], observation_indices),
            "actions": actions,       # 保留形状 [action_horizon, action_dim]
        }

    # ------------------------------------------------------------------- #
    #                 从整个数据集中 “无放回” 抽取 1 batch                    #
    # ------------------------------------------------------------------- #

    # 按批次遍历一轮有效动作块样本，支持打乱顺序和丢弃不足一批的尾部样本。
    def get_epoch_iterator(
        self,
        *,
        batch_size: int,
        observation_horizon: int,
        action_horizon: int,
        shuffle: bool = True,           # 每个 epoch 开始前，是否随机打乱所有训练样本的顺序
        drop_last: bool = False,        # 当最后剩余的样本不足一个完整 batch 时，是否丢弃这些样本
        device=None,
    ) -> Iterator:
        """遍历一次全部合法 action chunk 每次调用该方法都会创建一个有限的 epoch

        所有合法的 ``(trajectory_index, start_step)`` 在该 epoch 中最多出现一次；当 ``drop_last=False`` 时会恰好出现一次
        """

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if observation_horizon <= 0:
            raise ValueError("observation_horizon must be positive.")
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive.")

        valid_counts = self._num_valid_chunks_per_trajectory(action_horizon)
        total_valid_starts = int(valid_counts.sum())
        if total_valid_starts == 0:
            raise ValueError(f"No stored trajectory is long enough for the requested action_horizon={action_horizon}.")

        # 全局下标与 sample_chunk 中的定义完全一致，但这里采用无放回遍历
        global_starts = np.arange(total_valid_starts, dtype=np.int64)
        if shuffle:
            self._rng.shuffle(global_starts)

        # 丢弃不足以构成完整 batch 的样本
        if drop_last:
            epoch_size = total_valid_starts - total_valid_starts % batch_size
        else:
            epoch_size = total_valid_starts

        cumulative_counts = np.cumsum(valid_counts)
        for batch_start in range(0, epoch_size, batch_size):
            # 确定一个批次 chunk 起点
            batch_global_starts = global_starts[batch_start : min(batch_start + batch_size, epoch_size)]

            batch = self._build_batch_from_global_starts(
                batch_global_starts,
                cumulative_counts=cumulative_counts,
                observation_horizon=observation_horizon,
                action_horizon=action_horizon,
            )
            yield jax.device_put(batch, device=device)

    # ------------------------------------------------------------------- #
    #               从整个数据集中 “有放回的” 随机抽取 1 batch                 #
    # ------------------------------------------------------------------- #

    # 持续随机提供动作块训练 batch，并预取到指定的 JAX 设备。
    def get_chunk_iterator(
        self,
        *,
        sample_args: dict[str, Any],        # 例如 sample_args={"batch_size": 256}
        queue_size: int = 2,                # 预先准备多少 batch
        device=None,
    ) -> Iterator:
        """持续预取行为克隆 batch，并将其放到指定 JAX 设备。"""

        if queue_size <= 0:
            raise ValueError("queue_size must be positive.")

        # 创建双端队列，保存已经准备好的 Batch
        queue = collections.deque()

        # 准备指定数量的 batch，传到 JAX 设备并加入预取队列。
        def enqueue(count: int) -> None:
            for _ in range(count):
                queue.append(jax.device_put(self.sample_chunk(**sample_args), device=device))

        # 生成器
        enqueue(queue_size)                 # 预先取 queue_size 个 Batch
        while queue:
            yield queue.popleft()           # yield 暂停函数，下次调用从此处继续
            enqueue(1)                      # 每取一个就补充一个，生成器会一直运行，直到外部停止迭代
