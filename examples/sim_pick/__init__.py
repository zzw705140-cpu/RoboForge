"""Cube-picking task."""

# 导入环境配置，任务环境，训练配置
from examples.sim_pick.config import EnvConfig, PickEnv, TrainConfig

# 该任务对外提供的公共接口
__all__ = ["EnvConfig", "PickEnv", "TrainConfig"]
