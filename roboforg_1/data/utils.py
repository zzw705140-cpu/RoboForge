"""数据模块共用的嵌套数组树操作。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


ArrayTree = np.ndarray | dict[str, "ArrayTree"]

#####################################################################################
#    注！！  Roboforge里面没有task_condition这个观测数据
#           此文件只是对数据格式进行转换


def stack_tree(values: Sequence[Any]) -> ArrayTree:
    """
    将多个结构相同的数据沿第 0 维堆叠，同时支持嵌套字典。

    例如，输入多个 observation:

        values = [
            {
                "state": state_0,                       # [D]
                "images": {
                    "front": front_0,                   # [H, W, C]
                    "wrist": wrist_0,                   # [H, W, C]
                },
                "task_condition": {
                    "task_one_hot": task_one_hot_0,     # [task_dim]
                },
            },
            {
                "state": state_1,   
                "images": {
                    "front": front_1,
                    "wrist": wrist_1,
                },
                "task_condition": {
                    "task_one_hot": task_one_hot_1,
                },
            },
        ]

    堆叠后得到：

        {
            "state": ...,                               # [B, D]
            "images": {
                "front": ...,                           # [B, H, W, C]
                "wrist": ...,                           # [B, H, W, C]
            },
            "task_condition": {
                "task_one_hot": ...,                    # [B, task_dim]
            },
        }

    如果输入的是一条轨迹中的连续观测，则新增的第 0 维表示时间 T; 如果输入的是多个采样结果，则新增的第 0 维表示 batch B
    """
    if not values:
        raise ValueError("Cannot stack an empty sequence.")

    # 以第一个元素的结构作为基准，递归检查其余元素的字典字段
    first = values[0]
    if isinstance(first, Mapping):
        expected_keys = tuple(first.keys())
        expected_key_set = set(expected_keys)
        for value in values[1:]:
            if not isinstance(value, Mapping):
                raise ValueError("All nested values must have identical keys.")
            if set(value.keys()) != expected_key_set:
                raise ValueError("All nested values must have identical keys.")
        return {
            key: stack_tree([value[key] for value in values])
            for key in expected_keys
        }

    # 到达普通叶子后，统一转成数组并创建新的第 0 维
    return np.stack([np.asarray(value) for value in values], axis=0)


def tree_take(tree: ArrayTree, indices: Any) -> ArrayTree:
    """
    对嵌套数据树的所有数组叶子应用相同索引

    例如输入：

        tree = {
            "state": states,                            # [T, state_dim]
            "images": {
                "front": front_images,                  # [T, H, W, C]
                "wrist": wrist_images,                  # [T, H, W, C]
            },
            "task_condition": {
                "task_one_hot": task_one_hots,          # [T, task_dim]
            },
        }

    调用 tree_take(tree, 3) 后得到：

        {
            "state": states[3],
            "images": {
                "front": front_images[3],
                "wrist": wrist_images[3],
            },
            "task_condition": {
                "task_one_hot": task_one_hots[3],
            },
        }

    indices 也可以是数组，例如 [1, 2, 3]，此时所有叶子都会选取相同位置，并保留对应的批量维度
    """
    if isinstance(tree, np.ndarray):
        return tree[indices]
    if isinstance(tree, Mapping):
        return {key: tree_take(value, indices) for key, value in tree.items()}
    raise TypeError(f"Unsupported array-tree leaf: {type(tree).__name__}.")


def tree_asarray(tree: Any) -> ArrayTree:
    """
    保留嵌套字典结构，将所有非字典叶子整体转换为 NumPy 数组

    例如：

        tree = {
            "state": [[1, 2], [3, 4]],
            "task_condition": {
                "task_one_hot": [[1, 0], [1, 0]],
            },
        }

    转换后字典层级保持不变，而两个 Python list 叶子分别成为形状 [2, 2] 的 NumPy 数组。完整 list 会一次性转换，不会被拆成多个树节点
    """
    if isinstance(tree, Mapping):
        return {key: tree_asarray(value) for key, value in tree.items()}
    return np.asarray(tree)


def tree_leading_length(tree: ArrayTree) -> int:
    """
    检查数据树所有数组叶子的第 0 维长度一致，并返回该长度。

    例如输入：

        tree = {
            "state": states,                            # [101, state_dim]
            "images": {
                "front": front_images,                  # [101, H, W, C]
                "wrist": wrist_images,                  # [101, H, W, C]
            },
            "task_condition": {
                "task_one_hot": task_one_hots,          # [101, task_dim]
            },
        }

    递归取得的首维长度为 [101, 101, 101, 101]，因此返回 101。 如果任意叶子的首维不同、叶子是标量，或者树中不存在数组叶子，则抛出异常，避免后续采样时发生隐式错位
    """
    lengths: list[int] = []

    def collect_lengths(value: Any) -> None:
        if isinstance(value, Mapping):
            for child in value.values():
                collect_lengths(child)
            return

        array = np.asarray(value)
        if array.ndim == 0:
            raise ValueError("Every array-tree leaf must have a leading dimension.")
        lengths.append(len(array))

    collect_lengths(tree)
    if not lengths:
        raise ValueError("An array tree must contain at least one leaf.")
    if any(length != lengths[0] for length in lengths[1:]):
        raise ValueError(f"Array-tree leading dimensions disagree: {lengths}.")
    return lengths[0]


__all__ = [
    "ArrayTree",
    "stack_tree",
    "tree_asarray",
    "tree_leading_length",
    "tree_take",
]
