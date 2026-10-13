"""RoboForge 观测格式的规范化与旧数据兼容。"""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any


DEFAULT_CAMERA_KEYS = ("front", "wrist")

# 注！！！！！！
#
# 新增的 [observation.py](/home/xinyu/project/RoboForge/roboforg_1/data/observation.py) 用来集中处理观测格式转换和旧数据兼容：
# - 新采集数据：把 front、wrist 放进 images。
# - 旧数据：读取时做相同转换，无需重新采集。
# 单独放一页，是为了让采集和读取复用同一套转换逻辑。utils.py 继续只负责通用数组操作。

def canonicalize_observation(observation: Mapping[str, Any], *, camera_keys: tuple[str, ...] = DEFAULT_CAMERA_KEYS) -> dict[str, Any]:
    """将顶层相机字段整理到 images 子字典，兼容已经规范化的观测。"""
    if not isinstance(observation, Mapping):
        raise TypeError("observation must be a mapping.")
    if "state" not in observation:
        raise KeyError("observation is missing 'state'.")

    images_value = observation.get("images", {})
    if not isinstance(images_value, Mapping):
        raise TypeError("observation['images'] must be a mapping.")
    result = {key: value for key, value in observation.items() if key not in camera_keys and key != "images"}
    images = dict(images_value)
    for key in camera_keys:
        if key in observation:
            if key in images:
                raise ValueError(f"Camera {key!r} appears both at the top level and inside 'images'.")
            images[key] = observation[key]
    if not images:
        raise ValueError("observation must contain at least one camera image.")
    result["images"] = images
    return result


def canonicalize_trajectory(trajectory: Mapping[str, Any], *, camera_keys: tuple[str, ...] = DEFAULT_CAMERA_KEYS) -> dict[str, Any]:
    """规范化整条轨迹的观测树，保留其他 transition 字段。"""
    if not isinstance(trajectory, Mapping):
        raise TypeError("trajectory must be a mapping.")
    if "observations" not in trajectory:
        raise KeyError("trajectory is missing 'observations'.")

    result = dict(trajectory)
    result["observations"] = canonicalize_observation(trajectory["observations"], camera_keys=camera_keys)
    return result


def load_dataset(path: str | Path, *, camera_keys: tuple[str, ...] = DEFAULT_CAMERA_KEYS) -> dict[str, Any]:
    """读取演示数据，并将新旧轨迹统一为规范观测结构。"""
    with Path(path).expanduser().open("rb") as file:
        payload = pickle.load(file)
    if not isinstance(payload, Mapping) or "trajectories" not in payload:
        raise ValueError("Dataset must be a mapping containing 'trajectories'.")

    result = dict(payload)
    result["trajectories"] = [canonicalize_trajectory(trajectory, camera_keys=camera_keys) for trajectory in payload["trajectories"]]
    return result


__all__ = ["DEFAULT_CAMERA_KEYS", "canonicalize_observation", "canonicalize_trajectory", "load_dataset"]
