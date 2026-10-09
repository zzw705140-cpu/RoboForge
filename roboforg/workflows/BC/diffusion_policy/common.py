"""Diffusion Policy 训练与评估共用的配置、数据和 checkpoint 工具。"""
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
from flax.core import freeze

from roboforg.agent.diffusion_policy import DataNormalizer, DiffusionPolicy
from roboforg.data.data_buffer import DataBuffer
from roboforg.data.diffusion_policy_data import DiffusionPolicyBatchConverter, DiffusionPolicySampler
from roboforg.workflows.diffusion_policy.train_steps import DiffusionTrainState, create_train_state


@dataclass(frozen=True)
class DiffusionWorkflowConfig:
    dataset_root: Path = Path("datasets")
    task_name: str = "pick"
    train_dir_name: str = "train_1"
    eval_dir_name: str = "eval_1"
    camera_keys: tuple[str, ...] = ("front", "wrist")
    state_dim: int = 23
    action_dim: int = 7
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8
    num_inference_steps: int = 8
    diffusion_step_embedding_dim: int = 128
    down_channels: tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    predict_scale: bool = True
    batch_size: int = 32
    learning_rate: float = 1e-4
    beta1: float = 0.95
    beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 1e-6
    ema_update_after_step: int = 0
    ema_inv_gamma: float = 1.0
    ema_power: float = 0.75
    ema_min_decay: float = 0.0
    ema_max_decay: float = 0.9999

    def __post_init__(self) -> None:
        if self.task_name != "pick":
            raise ValueError("The first Diffusion Policy workflow supports only task_name='pick'.")
        if self.train_dir_name == self.eval_dir_name:
            raise ValueError("Train and eval directories must differ.")
        if self.state_dim != 23 or self.action_dim != 7:
            raise ValueError("The Pick workflow requires state_dim=23 and action_dim=7.")
        if self.n_obs_steps - 1 + self.n_action_steps > self.horizon:
            raise ValueError("The prediction horizon cannot contain the requested execution chunk.")
        if min(self.batch_size, self.num_inference_steps, self.kernel_size, self.n_groups) <= 0:
            raise ValueError("Batch and model dimensions must be positive.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer configuration.")

    def split_directory(self, split: str) -> Path:
        if split not in {"train", "eval"}:
            raise ValueError("split must be 'train' or 'eval'.")
        name = self.train_dir_name if split == "train" else self.eval_dir_name
        return Path(self.dataset_root) / self.task_name / name


def load_split_trajectories(config: DiffusionWorkflowConfig, split: str) -> list[dict[str, Any]]:
    directory = config.split_directory(split)
    paths = sorted(directory.glob("*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No demonstration files found in {directory}.")
    trajectories: list[dict[str, Any]] = []
    for path in paths:
        with path.open("rb") as file:
            payload = pickle.load(file)
        if not isinstance(payload, Mapping) or "trajectories" not in payload:
            raise ValueError(f"{path} is not a RoboForge compact demonstration file.")
        if payload.get("task") not in (None, config.task_name):
            raise ValueError(f"{path} belongs to another task.")
        if payload.get("split") not in (None, split):
            raise ValueError(f"{path} belongs to another split.")
        trajectories.extend(payload["trajectories"])
    return trajectories


def make_sampler(trajectories: Sequence[Mapping[str, Any]], config: DiffusionWorkflowConfig,
                 *, seed: int) -> DiffusionPolicySampler:
    total = sum(len(trajectory["actions"]) for trajectory in trajectories)
    buffer = DataBuffer.from_trajectories(list(trajectories), capacity=total, seed=seed)
    return DiffusionPolicySampler(
        buffer, n_obs_steps=config.n_obs_steps, horizon=config.horizon,
        n_action_steps=config.n_action_steps, seed=seed,
    )


def convert_batch(batch: Mapping[str, Any], config: DiffusionWorkflowConfig):
    return DiffusionPolicyBatchConverter(
        n_obs_steps=config.n_obs_steps, horizon=config.horizon, image_keys=config.camera_keys,
    )(batch)


def create_policy(config: DiffusionWorkflowConfig, *, normalizer: DataNormalizer) -> DiffusionPolicy:
    return DiffusionPolicy(
        camera_keys=config.camera_keys, state_dim=config.state_dim, n_obs_steps=config.n_obs_steps,
        horizon=config.horizon, n_action_steps=config.n_action_steps, action_dim=config.action_dim,
        num_inference_steps=config.num_inference_steps,
        diffusion_step_embedding_dim=config.diffusion_step_embedding_dim,
        down_channels=config.down_channels, kernel_size=config.kernel_size,
        n_groups=config.n_groups, predict_scale=config.predict_scale, normalizer=normalizer,
    )


def initialize_state(policy: DiffusionPolicy, config: DiffusionWorkflowConfig, rng: jax.Array,
                     observations: Mapping[str, Any], *, pretrained_path: Path | None) -> DiffusionTrainState:
    return create_train_state(
        policy, rng, observations, learning_rate=config.learning_rate, beta1=config.beta1,
        beta2=config.beta2, eps=config.adam_epsilon, weight_decay=config.weight_decay,
        pretrained_path=pretrained_path,
    )


def mean_metrics(history: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not history:
        raise ValueError("Cannot average zero training batches.")
    return {
        key: float(np.mean([float(np.asarray(item[key])) for item in history]))
        for key in history[0]
    }


def save_checkpoint(path: Path, *, state: DiffusionTrainState, epoch: int,
                    config: DiffusionWorkflowConfig, normalizer: DataNormalizer,
                    metrics: Mapping[str, Any], save_training_state: bool) -> None:
    """EMA 始终保存；普通参数和优化器状态按配置选择保存。"""
    payload = {
        "format_version": 1,
        "algorithm": "diffusion_policy",
        "epoch": int(epoch),
        "config": asdict(config),
        "normalizer": normalizer.to_dict(),
        "ema_params": flax.serialization.to_state_dict(state.ema_params),
        "training_state": flax.serialization.to_state_dict(state) if save_training_state else None,
        "metrics": dict(metrics),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as file:
            pickle.dump(payload, file)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_checkpoint(path: Path) -> Mapping[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    with path.open("rb") as file:
        payload = pickle.load(file)
    required = {"format_version", "algorithm", "epoch", "config", "normalizer", "ema_params", "training_state"}
    if not isinstance(payload, Mapping) or required - set(payload):
        raise ValueError(f"{path} is not a Diffusion Policy checkpoint.")
    if payload["format_version"] != 1 or payload["algorithm"] != "diffusion_policy":
        raise ValueError("Unsupported Diffusion Policy checkpoint format.")
    return payload


def checkpoint_setup(path: Path) -> tuple[DiffusionWorkflowConfig, DataNormalizer, dict[str, Any]]:
    payload = read_checkpoint(path)
    values = dict(payload["config"])
    values["dataset_root"] = Path(values["dataset_root"])
    values["camera_keys"] = tuple(values["camera_keys"])
    values["down_channels"] = tuple(values["down_channels"])
    return DiffusionWorkflowConfig(**values), DataNormalizer.from_dict(payload["normalizer"]), {
        "epoch": int(payload["epoch"]),
        "metrics": dict(payload.get("metrics", {})),
        "has_training_state": payload["training_state"] is not None,
    }


def restore_ema_params(path: Path, initialized_params: Any) -> Any:
    restored = flax.serialization.from_state_dict(
        initialized_params, read_checkpoint(path)["ema_params"]
    )
    return freeze(restored)


def restore_training_state(path: Path, initialized_state: DiffusionTrainState) -> DiffusionTrainState:
    stored = read_checkpoint(path)["training_state"]
    if stored is None:
        raise ValueError("This checkpoint contains EMA weights only and cannot resume training. "
                         "Set SAVE_TRAINING_STATE=true when creating resumable checkpoints.")
    return flax.serialization.from_state_dict(initialized_state, stored)


__all__ = [
    "DiffusionWorkflowConfig", "checkpoint_setup", "convert_batch", "create_policy",
    "initialize_state", "load_split_trajectories", "make_sampler", "mean_metrics",
    "read_checkpoint", "restore_ema_params", "restore_training_state", "save_checkpoint",
]
