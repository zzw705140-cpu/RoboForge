"""ACT 训练、评估、执行工作流共用的数据、模型与 checkpoint 函数。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import os
import pickle

import flax.serialization
import jax
import numpy as np

from roboforg.agent.act import ACTAgent, ACTPolicy, DataNormalizer, LatentEncoder
from roboforg.data.act_data.data_conversion import ACTBatchConverter
from roboforg.data.chunk_bc_buffer import ChunkBCBuffer
from roboforg.model.resnet_loader import load_resnet10_into_policy_state
from roboforg.networks.vision.resnet_v1 import resnetv1_configs


# 工作流默认参数；训练与手动仿真共用。
DEFAULT_NUM_EPOCHS = 1000
CHECKPOINT_EVERY_EPOCHS = 200
CHECKPOINT_KEEP_LAST = 5
ROLLOUT_EVERY_EPOCHS = 500  # 仿真成功率评估间隔，不是离线 eval loss 的间隔
HEADLESS_NUM_ROLLOUTS = 40
VIEWER_NUM_ROLLOUTS = 10


# 集中保存 ACT workflow 的网络、数据和优化器超参数；训练、评估、执行使用同一份配置。
@dataclass(frozen=True)
class ACTWorkflowConfig:
    """Configuration shared by ACT training, offline evaluation, and rollout."""

    dataset_root: Path = Path("datasets")              # 数据集根目录
    task_name: str = "pick"                            # 当前任务名称
    camera_keys: tuple[str, ...] = ("front", "wrist")  # 使用的相机顺序
    image_height: int = 128                            # 网络输入图像高度
    image_width: int = 128                             # 网络输入图像宽度
    action_dim: int = 7                                # 单步动作维度
    observation_horizon: int = 1                       # 每个样本使用几帧历史观测
    action_horizon: int = 4                            # 每次预测的连续动作数
    batch_size: int = 16                               # 每次更新同时使用的样本数
    token_dim: int = 256                               # Transformer token 特征维度
    latent_dim: int = 32                               # CVAE 潜变量 z 的维度
    num_heads: int = 8                                 # 注意力头数量
    feedforward_dim: int = 2048                        # Transformer 前馈层隐藏维度
    latent_encoder_layers: int = 4                     # CVAE 编码器层数
    policy_encoder_layers: int = 4                     # 观测编码器层数
    policy_decoder_layers: int = 7                     # 动作解码器层数
    dropout_rate: float = 0.1                          # 训练期随机失活比例
    learning_rate: float = 1e-4                        # 非视觉网络学习率
    backbone_learning_rate: float = 1e-5               # ResNet 微调学习率
    weight_decay: float = 0.0                          # AdamW 权重衰减
    kl_weight: float = 10.0                            # CVAE KL loss 权重
    resnet_name: str = "resnetv1-10"                   # 视觉主干配置名称

    # 配置创建时先拦截会导致张量形状不一致或优化器无效的明显参数。
    def __post_init__(self) -> None:
        if self.task_name != "pick":
            raise ValueError("This first ACT workflow currently supports only task_name='pick'.")
        if not self.camera_keys or len(set(self.camera_keys)) != len(self.camera_keys):
            raise ValueError("camera_keys must be non-empty and contain no duplicates.")
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("image_height and image_width must be positive.")
        if self.action_dim <= 0 or self.observation_horizon <= 0 or self.action_horizon <= 0:
            raise ValueError("action_dim, observation_horizon, and action_horizon must be positive.")
        if self.batch_size <= 0 or self.token_dim <= 0 or self.latent_dim <= 0:
            raise ValueError("batch_size, token_dim, and latent_dim must be positive.")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if self.token_dim % 4 or self.token_dim % self.num_heads:
            raise ValueError("token_dim must be divisible by 4 and by num_heads.")
        if min(self.feedforward_dim, self.latent_encoder_layers, self.policy_encoder_layers, self.policy_decoder_layers) <= 0:
            raise ValueError("Transformer widths and layer counts must be positive.")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1).")
        if self.learning_rate <= 0.0 or self.backbone_learning_rate <= 0.0:
            raise ValueError("learning_rate and backbone_learning_rate must be positive.")
        if self.weight_decay < 0.0 or self.kl_weight < 0.0:
            raise ValueError("weight_decay and kl_weight must be non-negative.")

    # 返回数据集内某个 split 的目录，例如 datasets/pick/train。
    def split_directory(self, split: str) -> Path:
        """Return the local directory containing one dataset split."""
        if split not in {"train", "eval"}:
            raise ValueError("split must be 'train' or 'eval'.")
        return Path(self.dataset_root) / self.task_name / split


# 读取一个 split 下的全部采集批次，并合并成完整轨迹列表；不修改原始 pkl 文件。
def load_split_trajectories(config: ACTWorkflowConfig, split: str) -> list[dict[str, Any]]:
    """Load every compact demonstration trajectory in the requested split."""
    data_directory = config.split_directory(split)
    paths = sorted(data_directory.glob("*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No demonstration files found in {data_directory}.")

    trajectories: list[dict[str, Any]] = []
    for path in paths:
        with path.open("rb") as file:
            payload = pickle.load(file)
        if not isinstance(payload, Mapping) or "trajectories" not in payload:
            raise ValueError(f"{path} is not a RoboForge compact demonstration file.")
        if payload.get("task") not in (None, config.task_name):
            raise ValueError(f"{path} belongs to task {payload.get('task')!r}, not {config.task_name!r}.")
        if payload.get("split") not in (None, split):
            raise ValueError(f"{path} is marked split={payload.get('split')!r}, not {split!r}.")
        file_trajectories = payload["trajectories"]
        if not isinstance(file_trajectories, list) or not file_trajectories:
            raise ValueError(f"{path} contains no complete trajectories.")
        trajectories.extend(file_trajectories)

    # ChunkBCBuffer 会进一步检查 observation[T+1]、action[T]、done 等轨迹对齐关系。
    if not trajectories:
        raise ValueError(f"No trajectories found in {data_directory}.")
    return trajectories


# 将完整轨迹交给只读训练缓冲区，后续由它随机提供合法的当前观测和动作 chunk。
def make_chunk_buffer(
    trajectories: Sequence[Mapping[str, Any]], *, seed: int | None = None
) -> ChunkBCBuffer:
    """Create an in-memory chunk sampler without altering the source trajectories."""
    if not trajectories:
        raise ValueError("Cannot create a ChunkBCBuffer from zero trajectories.")
    total_transitions = sum(len(trajectory["actions"]) for trajectory in trajectories)
    return ChunkBCBuffer.from_trajectories(
        list(trajectories), capacity=total_transitions, seed=seed
    )


# 从 buffer 随机取一个 raw chunk batch，并删去长度为 1 的观测时间维，得到 Agent 可直接使用的格式。
def sample_act_batch(
    buffer: ChunkBCBuffer, config: ACTWorkflowConfig
) -> tuple[dict[str, np.ndarray | jax.Array], np.ndarray | jax.Array]:
    """Sample and convert one ACT batch with the configured horizons and cameras."""
    chunk_batch = buffer.sample_chunk(
        config.batch_size,
        observation_horizon=config.observation_horizon,
        action_horizon=config.action_horizon,
    )
    return ACTBatchConverter(
        action_horizon=config.action_horizon, image_keys=config.camera_keys
    )(chunk_batch)


# 汇总一个 epoch 内所有 batch 的 BC、KL、total loss；训练和独立评估共用相同归约规则。
def mean_metrics(metric_history: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Return per-key arithmetic means for metrics collected from one epoch."""
    if not metric_history:
        raise ValueError("Cannot average metrics from zero batches.")
    keys = set(metric_history[0])
    if any(set(metrics) != keys for metrics in metric_history):
        raise ValueError("Every metric dictionary in an epoch must have identical keys.")
    return {
        key: float(np.mean([float(np.asarray(metrics[key])) for metrics in metric_history]))
        for key in sorted(keys)
    }


# 只从 train 轨迹拟合 state/action 统计量，禁止把 eval 演示混入标准化统计。
def fit_train_normalizer(trajectories: Sequence[Mapping[str, Any]]) -> DataNormalizer:
    """Fit the ACT normalizer exclusively from complete train trajectories."""
    return DataNormalizer.fit(trajectories)


# 根据配置创建可微调 ResNet-10、CVAE 后验编码器与 ACT policy，再由 Agent 初始化两套参数和优化器。
def create_act_agent(
    config: ACTWorkflowConfig,
    *,
    normalizer: DataNormalizer,
    example_batch: tuple[Mapping[str, Any], Any],
    rng: jax.Array,
    load_pretrained_backbone: bool,
) -> ACTAgent:
    """Build a fresh ACTAgent; optionally replace its new backbone with ResNet-10 weights."""
    if config.resnet_name not in resnetv1_configs:
        raise KeyError(f"Unknown ResNet config {config.resnet_name!r}.")

    # none 保留 4×4 空间特征图；pre_pooling=False 保证主干梯度可参与微调。
    backbone = resnetv1_configs[config.resnet_name](
        pooling_method="none",
        bottleneck_dim=None,
        pre_pooling=False,
        image_size=(config.image_height, config.image_width),
    )
    policy = ACTPolicy(
        backbone=backbone,
        action_horizon=config.action_horizon,
        action_dim=config.action_dim,
        token_dim=config.token_dim,
        num_heads=config.num_heads,
        feedforward_dim=config.feedforward_dim,
        encoder_num_layers=config.policy_encoder_layers,
        decoder_num_layers=config.policy_decoder_layers,
        dropout_rate=config.dropout_rate,
        feature_height=4,
        feature_width=4,
        camera_keys=config.camera_keys,
        backbone_trainable=True,
    )
    latent_encoder = LatentEncoder(
        token_dim=config.token_dim,
        latent_dim=config.latent_dim,
        num_heads=config.num_heads,
        feedforward_dim=config.feedforward_dim,
        num_layers=config.latent_encoder_layers,
        dropout_rate=config.dropout_rate,
    )
    agent = ACTAgent.create(
        policy=policy,
        latent_encoder=latent_encoder,
        normalizer=normalizer,
        example_batch=example_batch,
        rng=rng,
        learning_rate=config.learning_rate,
        backbone_learning_rate=config.backbone_learning_rate,
        weight_decay=config.weight_decay,
        kl_weight=config.kl_weight,
    )

    # 只有全新训练需要覆盖随机主干；评估/执行稍后会从完整 checkpoint 恢复主干。
    if load_pretrained_backbone:
        agent = agent.replace(
            policy_state=load_resnet10_into_policy_state(agent.policy_state)
        )
    return agent


# 将训练可恢复的两个 TrainState、normalizer、epoch 和配置写进单一 checkpoint 文件。
def save_checkpoint(
    checkpoint_path: Path,
    *,
    agent: ACTAgent,
    epoch: int,
    config: ACTWorkflowConfig,
    metrics: Mapping[str, Any] | None = None,
) -> None:
    """Atomically save everything required to resume or deploy an ACT model."""
    if epoch < 0:
        raise ValueError("epoch must be non-negative.")
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "epoch": int(epoch),
        "config": asdict(config),
        "normalizer": agent.normalizer.to_dict(),
        "policy_state": flax.serialization.to_state_dict(agent.policy_state),
        "latent_state": flax.serialization.to_state_dict(agent.latent_state),
        "metrics": dict(metrics or {}),
    }

    # 先写临时文件，再原子替换目标；进程中断不会留下半个 checkpoint 伪装成完整模型。
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    try:
        with temporary_path.open("wb") as file:
            pickle.dump(payload, file)
        os.replace(temporary_path, checkpoint_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


# 读取并验证 checkpoint 的基础字段；评估、执行和恢复训练共用此函数，避免各自解释 pkl 格式。
def _read_checkpoint_payload(checkpoint_path: Path) -> Mapping[str, Any]:
    """Load and validate the common ACT checkpoint envelope."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}.")
    with checkpoint_path.open("rb") as file:
        payload = pickle.load(file)
    required_keys = {"format_version", "epoch", "config", "normalizer", "policy_state", "latent_state"}
    if not isinstance(payload, Mapping) or required_keys - set(payload):
        raise ValueError(f"{checkpoint_path} is not a complete ACT checkpoint.")
    if payload["format_version"] != 1:
        raise ValueError(f"Unsupported ACT checkpoint version: {payload['format_version']!r}.")
    return payload


# 在创建 Agent 前读取 checkpoint 的网络配置与训练期 normalizer，保证评估/执行使用完全相同的数据尺度。
def read_checkpoint_setup(
    checkpoint_path: Path,
) -> tuple[ACTWorkflowConfig, DataNormalizer, dict[str, Any]]:
    """Read the configuration, normalizer, and descriptive metadata from an ACT checkpoint."""
    payload = _read_checkpoint_payload(checkpoint_path)
    config_values = dict(payload["config"])
    config_values["dataset_root"] = Path(config_values["dataset_root"])
    config = ACTWorkflowConfig(**config_values)
    normalizer = DataNormalizer.from_dict(payload["normalizer"])
    metadata = {
        "epoch": int(payload["epoch"]),
        "config": config_values,
        "metrics": dict(payload.get("metrics", {})),
    }
    return config, normalizer, metadata


# 将 checkpoint 的参数、优化器状态和训练期标准化器写回同结构的新 Agent；网络配置必须由调用方先匹配创建。
def load_checkpoint_into_agent(
    checkpoint_path: Path, agent: ACTAgent
) -> tuple[ACTAgent, dict[str, Any]]:
    """Restore a checkpoint into an already-created, shape-compatible ACTAgent."""
    payload = _read_checkpoint_payload(checkpoint_path)

    # from_state_dict 会严格检查参数树结构与形状；配置不匹配时不能静默加载错误权重。
    policy_state = flax.serialization.from_state_dict(agent.policy_state, payload["policy_state"])
    latent_state = flax.serialization.from_state_dict(agent.latent_state, payload["latent_state"])
    restored_agent = agent.replace(
        policy_state=policy_state,
        latent_state=latent_state,
        normalizer=DataNormalizer.from_dict(payload["normalizer"]),
    )
    metadata = {
        "epoch": int(payload["epoch"]),
        "config": dict(payload["config"]),
        "metrics": dict(payload.get("metrics", {})),
    }
    return restored_agent, metadata
