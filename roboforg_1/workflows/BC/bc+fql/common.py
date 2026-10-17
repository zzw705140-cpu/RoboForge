"""BC + FQL 训练与评估共享的配置, 数据转换和 checkpoint 工具.

不包含训练循环, 环境 rollout 或资源监控.算法更新由 FQLAGENT 负责.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any
import os
import pickle
import tempfile

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from roboforg_1.agent.algorithm.fql import FQLAGENT
from roboforg_1.data.chunk_bc_replay_buffer import ChunkBCReplayBuffer
from roboforg_1.data.data_store import InMemoryDataStore
from roboforg_1.data.observation import load_dataset
from roboforg_1.utils.checkpoint_untils import find_latest_checkpoint, find_latest_run_dir

PROJECT_ROOT = Path(__file__).resolve().parents[4]
ALGORITHM = "fql"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints" / ALGORITHM
MODEL_NAMES = ("encoder", "behavior_flow", "one_step_actor", "critic", "target_critic")


"""------------------------- 共享配置 -------------------------"""

@dataclass(frozen=True)
class FQLWorkflowConfig:
    """训练和评估共用的数据与网络配置, 运行节奏由各入口管理."""

    # 数据与任务;环境根据 task_name 由入口查询 examples.mappings.
    task_name: str = "pick"
    dataset_root: Path = PROJECT_ROOT / "datasets"
    train_dir_name: str = "train"
    eval_dir_name: str = "eval"
    demo_pattern: str = "*.pkl"
    camera_keys: tuple[str, ...] = ("front", "wrist")
    batch_size: int = 16
    observation_horizon: int = 1
    action_horizon: int = 4
    use_temporal_history: bool = True

    # Encoder 的视觉池化和冻结设置目前固定在算法中.
    encoder_lr: float = 3e-4
    # Behavior Flow / Actor 共用的主干规格.
    network_type: str = "condition"
    hidden_dim: int = 512
    num_blocks: int = 4
    time_embed_dim: int = 32
    dropout_rate: float = 0.0
    num_ode_steps: int = 10
    behavior_flow_lr: float = 3e-4
    one_step_actor_lr: float = 3e-4
    # Critic, 目标更新与 FQL 蒸馏约束.
    critic_hidden_dims: tuple[int, ...] = (512, 512, 512, 512)
    num_qs: int = 2
    critic_lr: float = 3e-4
    discount: float = 0.99
    tau: float = 0.005
    alpha: float = 100.0
    q_aggregation: str = "mean"

    def __post_init__(self) -> None:
        """检查共享流程配置, 网络超参数由算法创建函数检查."""
        for name in ("task_name", "train_dir_name", "eval_dir_name"):
            value = getattr(self, name)
            if not value or value in {".", ".."} or Path(value).name != value:
                raise ValueError(f"{name} must be a single directory name.")
        if self.train_dir_name == self.eval_dir_name:
            raise ValueError("Train and eval directories must differ.")
        if not self.camera_keys or len(set(self.camera_keys)) != len(self.camera_keys):
            raise ValueError("camera_keys must be non-empty and unique.")
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if not self.use_temporal_history and self.observation_horizon != 1:
            raise ValueError("Non-temporal observations require observation_horizon=1.")
        # 其余网络和算法超参数由 FQLAGENT.create 统一检查.

    def split_directory(self, split: str) -> Path:
        """根据 train 或 eval 返回对应的数据集目录."""
        if split not in {"train", "eval"}:
            raise ValueError("split must be 'train' or 'eval'.")
        name = self.train_dir_name if split == "train" else self.eval_dir_name
        return Path(self.dataset_root).expanduser() / self.task_name / name

    def agent_kwargs(self) -> dict[str, Any]:
        """只将算法接受的参数传给 create,不混入训练流程配置."""
        excluded = {"task_name", "dataset_root", "train_dir_name", "eval_dir_name", "demo_pattern", "camera_keys", "batch_size"}
        return {field.name: getattr(self, field.name) for field in fields(self) if field.name not in excluded}


"""------------------------- 数据读取与 batch 转换 -------------------------"""

def load_split_trajectories(config: FQLWorkflowConfig, split: str) -> list[dict[str, Any]]:
    """读取指定数据划分, 统一观测格式并选取配置中的相机."""
    directory = config.split_directory(split)
    paths = sorted(directory.glob(config.demo_pattern))
    if not paths:
        raise FileNotFoundError(f"No demonstration files in {directory}.")
    trajectories = []
    for path in paths:
        payload = load_dataset(path, camera_keys=config.camera_keys)
        if payload.get("task") not in (None, config.task_name):
            raise ValueError(f"Dataset task mismatch: {path}")
        if payload.get("split") not in (None, split):
            raise ValueError(f"Dataset split mismatch: {path}")
        for trajectory in payload["trajectories"]:
            images = trajectory["observations"]["images"]
            if any(key not in images for key in config.camera_keys):
                raise ValueError(f"Dataset is missing a configured camera: {path}")
            trajectory["observations"]["images"] = {key: images[key] for key in config.camera_keys}
            trajectories.append(trajectory)
    if not trajectories:
        raise ValueError("Dataset contains no trajectories.")
    return trajectories


def make_store(trajectories: Sequence[Mapping[str, Any]]) -> InMemoryDataStore:
    """将完整轨迹装入共享的内存 Store."""
    return InMemoryDataStore.from_trajectories(trajectories)


def make_bc_sampler(store: InMemoryDataStore, *, seed: int) -> ChunkBCReplayBuffer:
    """创建 BC 动作块采样器, 不复制 Store 中的轨迹."""
    return ChunkBCReplayBuffer(store, seed=seed)


def convert_batch(batch: Mapping[str, Any], config: FQLWorkflowConfig) -> dict[str, Any]:
    """NumPy → JAX;数据采样器统一保留历史维,无历史模式显式去掉该维."""
    result = dict(batch)
    if not config.use_temporal_history:
        for name in ("observations", "next_observations"):
            if name in result:
                def remove_history(value):
                    """移除长度为 1 的历史维, 保留 batch 维."""
                    value = np.asarray(value)
                    if value.ndim < 2 or value.shape[1] != 1:
                        raise ValueError("Non-temporal conversion requires history length 1.")
                    return value[:, 0]
                result[name] = jax.tree_util.tree_map(remove_history, result[name])
    return jax.tree_util.tree_map(jnp.asarray, result)


"""------------------------- Agent 创建 -------------------------"""

def create_agent(config: FQLWorkflowConfig, sample_observations: Mapping[str, Any],
                 sample_actions: Any, *, seed: int, load_pretrained_backbone: bool = True) -> FQLAGENT:
    """输入单个观测和单个动作块,不能带 batch 维."""
    return FQLAGENT.create(
        seed=seed, sample_observations=sample_observations,
        sample_actions=jnp.asarray(sample_actions).reshape(-1),
        load_pretrained_backbone=load_pretrained_backbone, **config.agent_kwargs(),
    )


"""------------------------- Checkpoint 定位与保存恢复 -------------------------"""

def resolve_checkpoint(*, checkpoint: str | Path | None = None, run_dir: str | Path | None = None,
                       checkpoint_root: str | Path = CHECKPOINT_ROOT, task_name: str = "pick") -> Path:
    """选择指定 checkpoint, 或查找指定/最新实验中的最新 checkpoint."""
    if checkpoint is not None and run_dir is not None:
        raise ValueError("Specify checkpoint or run_dir, not both.")
    if checkpoint is not None:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    directory = run_dir if run_dir is not None else find_latest_run_dir(checkpoint_root, task_name, ALGORITHM)
    return find_latest_checkpoint(directory)


def save_checkpoint(path: str | Path, *, agent: FQLAGENT, epoch: int,
                    config: FQLWorkflowConfig, sample_observations: Mapping[str, Any],
                    metrics: Mapping[str, Any], runtime: Mapping[str, Any] | None = None) -> Path:
    """保存全部网络, 优化器与 RNG;样例只保存形状/dtype,不复制图像内容."""
    template = jax.tree_util.tree_map(
        lambda value: {"shape": tuple(value.shape), "dtype": str(value.dtype)}, sample_observations,
    )
    payload = {
        "format_version": 1, "algorithm": ALGORITHM, "epoch": int(epoch),
        "config": asdict(config), "agent_config": dict(agent.config),
        "observation_spec": template,
        "state": flax.serialization.to_state_dict(jax.device_get(agent)),
        "metrics": jax.device_get(dict(metrics)), "runtime": dict(runtime or {}),
    }
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def read_checkpoint(path: str | Path) -> dict[str, Any]:
    """读取 checkpoint 并检查算法标识和格式版本."""
    with Path(path).expanduser().open("rb") as file:
        payload = pickle.load(file)
    required = {"format_version", "algorithm", "epoch", "config", "agent_config", "observation_spec", "state"}
    if not isinstance(payload, dict) or required - payload.keys():
        raise ValueError("Not a complete FQL checkpoint.")
    if payload["format_version"] != 1 or payload["algorithm"] != ALGORITHM:
        raise ValueError("Unsupported FQL checkpoint format.")
    return payload


def config_from_dict(values: Mapping[str, Any]) -> FQLWorkflowConfig:
    """将保存的字典还原为配置对象, 恢复路径和元组类型."""
    values = dict(values)
    values["dataset_root"] = Path(values["dataset_root"])
    for key in ("camera_keys", "critic_hidden_dims"):
        values[key] = tuple(values[key])
    return FQLWorkflowConfig(**values)


def load_checkpoint(path: str | Path) -> tuple[FQLAGENT, FQLWorkflowConfig, dict[str, Any]]:
    """仅凭 checkpoint 重建网络并恢复完整状态,不需要数据集或预训练文件."""
    payload = read_checkpoint(path)
    config = config_from_dict(payload["config"])
    observations = jax.tree_util.tree_map(
        lambda spec: jnp.zeros(tuple(spec["shape"]), dtype=spec["dtype"]),
        payload["observation_spec"], is_leaf=lambda value: isinstance(value, dict) and set(value) == {"shape", "dtype"},
    )
    agent = create_agent(config, observations, jnp.zeros(payload["agent_config"]["flow_dim"]),
                         seed=0, load_pretrained_backbone=False)
    if dict(agent.config) != payload["agent_config"]:
        raise ValueError("Checkpoint model configuration does not match the reconstructed model.")
    # Flax 恢复后逐叶检查形状,避免不兼容权重静默替换初始化参数.
    restored = flax.serialization.from_state_dict(agent, payload["state"])
    for expected, actual in zip(jax.tree_util.tree_leaves(agent), jax.tree_util.tree_leaves(restored)):
        if np.shape(expected) != np.shape(actual):
            raise ValueError("Checkpoint state shape mismatch.")
    restored = jax.tree_util.tree_map(jnp.asarray, restored)
    return restored, config, {"epoch": payload["epoch"], "metrics": payload.get("metrics", {}), "runtime": payload.get("runtime", {})}


"""------------------------- 指标汇总 -------------------------"""

def mean_metrics(history: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """计算各批次标量指标的算术平均值."""
    if not history:
        raise ValueError("Cannot average zero batches.")
    return {key: float(np.mean([float(np.asarray(item[key])) for item in history])) for key in history[0]}
