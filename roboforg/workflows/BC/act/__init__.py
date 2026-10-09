"""ACT 的训练、离线评估与仿真执行工作流。"""

# workflow 对外只公开共享配置；train/eval/run 是独立命令入口，不在导入包时自动加载。
from .common import ACTWorkflowConfig

__all__ = ["ACTWorkflowConfig"]
