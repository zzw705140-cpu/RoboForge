"""Diffusion Policy 的时间窗口采样与 batch 格式检查。

默认观测 2 帧、预测 16 步、执行 8 步。观测与预测动作从同一时刻开始。
图像保留 [B, To, H, W, 3]，这里不进行像素归一化或增强。
"""
from collections.abc import Callable, Mapping, Sequence
from numbers import Integral
from typing import Any

import numpy as np
from flax.core import frozen_dict

from roboforg.data.data_buffer import DataBuffer, _stack_tree


# ============================================================================
# 参数检查与可替换的边界填充策略
# ============================================================================

# 检查窗口长度、batch 大小等参数，拒绝布尔值、非整数和非正数。
def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


# 在序列开头重复第一项，补齐缺失的历史时间步。
def pad_before(values: np.ndarray, count: int) -> np.ndarray:
    """沿时间维重复首项；用于补齐不存在的历史观测和历史动作。"""

    # 填充数量不能为负，且必须有真实数据作为重复来源。
    if count < 0 or len(values) == 0:
        raise ValueError("Padding requires nonempty values and count >= 0.")
    # 保留首项的时间维，重复 count 次后拼接到原序列前方。
    return np.concatenate((np.repeat(values[:1], count, axis=0), values), axis=0)


# 在序列末尾重复最后一项，补齐超出轨迹末端的时间步。
def pad_after(values: np.ndarray, count: int) -> np.ndarray:
    """沿时间维重复末项；用于补齐轨迹结束后的动作。"""
    
    # 填充数量不能为负，且必须有真实数据作为重复来源。
    if count < 0 or len(values) == 0:
        raise ValueError("Padding requires nonempty values and count >= 0.")
    # 保留末项的时间维，重复 count 次后拼接到原序列后方。
    return np.concatenate((values, np.repeat(values[-1:], count, axis=0)), axis=0)


# ============================================================================
# 时间窗口采样：完整轨迹 → 对齐的观测窗口与动作窗口 → batch
# ============================================================================

class DiffusionPolicySampler:
    """复用 DataBuffer 存储，单独实现 DP 采样，不修改 ChunkBCBuffer。

    窗口起点 s 的范围为 [-To+1, T-horizon+n_action_steps-1]，
    同时要求当前时刻 s+To-1 不超过 T-1。太短而没有合法窗口的轨迹不采样。
    padding 函数接收 (有效序列, 填充数量)，返回补齐后的序列。
    """

    # 保存采样配置、轨迹缓冲区及首尾填充函数，建立独立随机数生成器。
    def __init__(
        self, buffer: DataBuffer, *, n_obs_steps: int = 2,
        horizon: int = 16, n_action_steps: int = 8,
        before_padding: Callable = pad_before,
        after_padding: Callable = pad_after,
        seed: int | None = None,
    ) -> None:
        # 检查三个时间长度，并保证预测序列能容纳历史部分与执行部分。
        for name, value in (("n_obs_steps", n_obs_steps), ("horizon", horizon),
                            ("n_action_steps", n_action_steps)):
            _positive_integer(value, name)
        if n_obs_steps - 1 + n_action_steps > horizon:
            raise ValueError("n_obs_steps - 1 + n_action_steps must be <= horizon.")

        # 保存配置；填充策略通过函数参数传入，后续可独立替换。
        self.buffer = buffer
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.before_padding = before_padding
        self.after_padding = after_padding
        self._rng = np.random.default_rng(seed)

    # 计算每条轨迹的合法窗口数量；没有合法起点的轨迹返回 0。
    def _counts(self) -> np.ndarray:
        # T 是动作数量；最早起点为 1 - n_obs_steps，使当前时刻恰好为 0。
        lengths = self.buffer.trajectory_lengths

        # 最晚起点同时满足：尾部最多填充 n_action_steps - 1 步，
        # 且最后一帧输入观测仍对应一个真实动作，不能落在终止观测 o_T。
        last_start = np.minimum(lengths - self.horizon + self.n_action_steps - 1,
                                lengths - self.n_obs_steps)

        # 闭区间起点数量 = 最晚起点 - 最早起点 + 1；负数截为 0。
        return np.maximum(last_start + self.n_obs_steps, 0)

    # 截取一个时间窗口，并分别调用首部、尾部策略补齐越界部分。
    def _window(self, values: Any, start: int, length: int) -> Any:
        # 同一组时间边界递归应用到状态与各路相机。
        if isinstance(values, Mapping):
            return {key: self._window(value, start, length) for key, value in values.items()}

        # 只读取轨迹中真实存在的交集，end 是不包含在切片内的右边界。
        values = np.asarray(values)
        end = start + length
        window = values[max(start, 0):min(end, len(values))]

        # 先补缺失的历史，再补超出轨迹末端的部分；不会读取其他轨迹。
        if start < 0:
            window = self.before_padding(window, -start)
        if end > len(values):
            window = self.after_padding(window, end - len(values))

        # 替换策略也必须保持窗口长度以及各个非时间维度不变。
        if window.shape != (length, *values.shape[1:]):
            raise ValueError("Padding function returned an invalid window shape.")
        return window

    # 根据轨迹编号和共同起点，构造一份观测、动作对齐的训练样本。
    def build_sample(self, trajectory_index: int, *, start_step: int) -> dict:
        """start_step 是最早观测及动作的共同起点，允许为负。"""
        # 检查轨迹编号及起点是否落在该轨迹的合法采样区间。
        counts = self._counts()
        if not 0 <= trajectory_index < len(counts):
            raise IndexError("trajectory_index out of range.")
        first = 1 - self.n_obs_steps
        if not first <= start_step < first + counts[trajectory_index]:
            raise IndexError("start_step is outside the valid sampling range.")

        # 两个窗口起点相同、长度不同；当前时刻对应动作下标 n_obs_steps - 1。
        trajectory = self.buffer.get_trajectory(trajectory_index)
        return {
            "observations": self._window(trajectory["observations"], start_step, self.n_obs_steps),
            "actions": self._window(trajectory["actions"], start_step, self.horizon),
        }

    # 将全局窗口编号映射回轨迹及局部起点，再堆叠成训练 batch。
    def _batch(self, indices: np.ndarray, cumulative: np.ndarray):
        # cumulative 是各轨迹窗口数量的累计和，用于定位全局编号所属轨迹。
        samples = []
        for index in indices:
            trajectory_index = int(np.searchsorted(cumulative, index, side="right"))
            # 减去前面轨迹的窗口总数，再加最早起点，得到实际时间下标。
            previous = cumulative[trajectory_index - 1] if trajectory_index else 0
            start = int(index - previous) + 1 - self.n_obs_steps
            samples.append(self.build_sample(trajectory_index, start_step=start))

        # 沿新建的 batch 维堆叠各字段，FrozenDict 固定字典结构。
        return frozen_dict.freeze(_stack_tree(samples))

    # 建立合法窗口的累计数量索引，并在没有可用样本时明确报错。
    def _cumulative(self) -> np.ndarray:
        cumulative = np.cumsum(self._counts())
        if len(cumulative) == 0 or cumulative[-1] == 0:
            raise ValueError("No valid diffusion windows; trajectories are empty or too short.")
        return cumulative

    # 在全部合法窗口中均匀随机抽取一个 batch，允许重复抽到同一窗口。
    def sample(self, batch_size: int):
        """在所有合法窗口中均匀有放回采样，返回 NumPy batch。"""
        _positive_integer(batch_size, "batch_size")
        cumulative = self._cumulative()
        # 最后一个累计值是窗口总数；随机整数就是全局窗口编号。
        return self._batch(self._rng.integers(cumulative[-1], size=batch_size), cumulative)

    # 按 batch 遍历一轮合法窗口，可打乱顺序或丢弃不足一批的尾部。
    def get_epoch_iterator(self, *, batch_size: int, shuffle: bool = True,
                           drop_last: bool = False):
        """无放回遍历合法窗口；迭代期间不要修改 buffer。"""
        _positive_integer(batch_size, "batch_size")
        cumulative = self._cumulative()

        # 为每个合法窗口生成唯一编号，打乱只改变遍历顺序。
        indices = np.arange(cumulative[-1])
        if shuffle:
            self._rng.shuffle(indices)

        # 确定本轮样本数量；drop_last=True 时截去不足一个 batch 的余数。
        size = len(indices) - len(indices) % batch_size if drop_last else len(indices)

        # 每次只构造并返回一个 batch，避免一次性生成全部图像窗口。
        for offset in range(0, size, batch_size):
            yield self._batch(indices[offset:min(offset + batch_size, size)], cumulative)


# ============================================================================
# 模型输入转换：检查字段与形状，保留时间维，不负责采样和图像增强
# ============================================================================

class DiffusionPolicyBatchConverter:
    """检查 batch 并选择模型输入字段，保留观测的时间维和原始像素值。"""

    # 配置观测长度、预测长度及需要送入模型的相机字段。
    def __init__(self, *, n_obs_steps: int = 2, horizon: int = 16,
                 image_keys: Sequence[str] = ("front", "wrist")) -> None:
        # 观测窗口不能长于预测窗口。
        _positive_integer(n_obs_steps, "n_obs_steps")
        _positive_integer(horizon, "horizon")
        if n_obs_steps > horizon:
            raise ValueError("n_obs_steps must be <= horizon.")
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        # 固定相机字段顺序，并防止重复字段或与机器人状态字段冲突。
        self.image_keys = tuple(image_keys)
        if "state" in self.image_keys or len(set(self.image_keys)) != len(self.image_keys):
            raise ValueError("image_keys must be unique and must not include 'state'.")

    # 将采样 batch 整理为 (观测字典, 动作目标)，供后续训练调用。
    def __call__(self, batch: Mapping[str, Any]):
        # 读取必要字段，检查观测容器；缺少字段时直接抛出 KeyError。
        observations, actions = batch["observations"], batch["actions"]
        if not isinstance(observations, Mapping):
            raise TypeError("observations must be a mapping.")

        # 动作保留 [B, horizon, action_dim]，不固定机器人动作维数。
        if actions.ndim != 3 or actions.shape[0] <= 0 or actions.shape[1] != self.horizon or actions.shape[2] <= 0:
            raise ValueError("actions must have shape [B, horizon, action_dim].")

        # 以动作 batch 大小为基准检查状态，保留历史与当前帧的时间维。
        prefix = (actions.shape[0], self.n_obs_steps)
        state = observations["state"]
        if state.ndim != 3 or state.shape[:2] != prefix or state.shape[-1] <= 0:
            raise ValueError("state must have shape [B, n_obs_steps, state_dim].")
        converted = {"state": state}

        # 逐路检查图像的 batch、时间维、空间大小和 RGB 通道。
        # 直接保留原数组，因此数据类型、像素值和时间顺序均不改变。
        for key in self.image_keys:
            image = observations[key]
            if image.ndim != 5 or image.shape[:2] != prefix or image.shape[-1] != 3 or min(image.shape[2:4]) <= 0:
                raise ValueError(f"Image {key!r} must have shape [B, n_obs_steps, H, W, 3].")
            converted[key] = image

        # 仅返回状态、选定相机和动作；归一化或增强由后续模块负责。
        return converted, actions
