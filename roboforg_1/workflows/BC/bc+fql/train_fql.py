"""FQL 离线联合训练入口, 负责数据, 日志, checkpoint 和周期评估调度."""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from datetime import datetime
from importlib import import_module
import math
import os
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

import jax
import numpy as np

from roboforg_1.utils.checkpoint_untils import create_run_dir, load_run_config, save_run_config

_common = import_module(f"{__package__}.common")
_eval = import_module(f"{__package__}.eval_fql")
FQLWorkflowConfig = _common.FQLWorkflowConfig


class _TrainingProgress:
    """同一行显示累计 epoch 进度, 测评或退出前结束当前行."""

    def __init__(self, total: int):
        self.total = total
        self.active = False

    def show(self, completed: int) -> None:
        width = 30
        filled = completed * width // self.total
        # 最后一格只在全部训练完成后填满.
        bar = "#" * filled + " " * (width - filled)
        print(f"\r训练 {completed}/{self.total} [{bar}]", end="", flush=True)
        self.active = True

    def pause(self) -> None:
        if self.active:
            print(flush=True)
            self.active = False


"""------------------------- 配置与启动参数 -------------------------"""

def parse_args() -> argparse.Namespace:
    """读取模型配置和训练节奏, 模型选项未指定时使用默认值或 checkpoint 配置."""
    parser = argparse.ArgumentParser(description="Train or resume offline FQL.")
    defaults = FQLWorkflowConfig()
    for field in fields(defaults):
        value = getattr(defaults, field.name)
        value_type = str if isinstance(value, (bool, tuple)) else type(value)
        parser.add_argument(f"--{field.name.replace('_', '-')}", type=value_type, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=_common.CHECKPOINT_ROOT)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--num-epochs", type=int, default=1000, help="Target total epochs, including completed epochs on resume.")
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="Random batches per epoch; default ceil(complete chunks / batch size).")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--checkpoint-every-epochs", type=int, default=100)
    parser.add_argument("--checkpoint-keep", type=int, default=2)
    parser.add_argument("--eval-every-epochs", type=int, default=100, help="0 disables periodic evaluation.")
    parser.add_argument("--num-eval-episodes", type=int, default=40)
    parser.add_argument("--final-eval", choices=("true", "false"), default="true")
    parser.add_argument("--show-viewer", choices=("true", "false"), default="false")
    parser.add_argument("--print-policy-output", choices=("true", "false"), default="false", help="Compatibility option; training evaluation always suppresses action output. Use standalone evaluation to print actions.")
    parser.add_argument("--max-eval-episode-steps", type=int, default=None)
    parser.add_argument("--wandb", choices=("true", "false"), default="true")
    parser.add_argument("--wandb-debug", choices=("true", "false"), default="false")
    parser.add_argument("--wandb-project", default="roboforge")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-log-every", type=int, default=10)
    parser.add_argument("--upload-checkpoint-artifact", choices=("true", "false"), default="false")
    args = parser.parse_args()
    for name in ("num_epochs", "checkpoint_every_epochs", "checkpoint_keep", "num_eval_episodes", "wandb_log_every"):
        if getattr(args, name) <= 0:
            parser.error(f"{name.replace('_', '-')} must be positive.")
    for name in ("steps_per_epoch", "max_eval_episode_steps"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error(f"{name.replace('_', '-')} must be positive.")
    if args.eval_every_epochs < 0 or (args.seed is not None and args.seed < 0):
        parser.error("eval-every-epochs and seed must be non-negative.")
    if args.upload_checkpoint_artifact == "true" and args.wandb != "true":
        parser.error("Checkpoint Artifact upload requires --wandb=true.")
    return args


def build_config(args: argparse.Namespace, base: FQLWorkflowConfig | None = None) -> FQLWorkflowConfig:
    """合并显式命令行参数, 续训不允许静默改变保存的数据和算法配置."""
    values = asdict(base or FQLWorkflowConfig())
    for field in fields(FQLWorkflowConfig):
        value = getattr(args, field.name)
        if value is None:
            continue
        if field.name in ("camera_keys", "critic_hidden_dims"):
            value = tuple(part.strip() for part in value.split(",") if part.strip())
            if field.name == "critic_hidden_dims":
                value = tuple(int(part) for part in value)
        elif field.name == "use_temporal_history":
            if value not in {"true", "false"}:
                raise ValueError("use-temporal-history must be true or false.")
            value = value == "true"
        values[field.name] = value
    config = FQLWorkflowConfig(**values)
    if base is not None and config != base:
        raise ValueError("Resume data and model configuration must match the checkpoint.")
    return config


"""------------------------- Batch 更新与模型保存 -------------------------"""

def sample_batch(sampler: Any, config: FQLWorkflowConfig, *, batch_size: int | None = None) -> dict[str, Any]:
    """采样完整 H 步 transition, 使用与算法相同的 discount 并转换为 JAX 数组."""
    batch = sampler.sample_fql_chunk(batch_size or config.batch_size, observation_horizon=config.observation_horizon, action_horizon=config.action_horizon, discount=config.discount)
    return _common.convert_batch(batch, config)


def train_epoch(agent: Any, sampler: Any, config: FQLWorkflowConfig, *, epoch: int, steps_per_epoch: int, seed: int, logger: Any, log_every: int) -> tuple[Any, dict[str, float]]:
    """执行一轮固定次数的有放回随机更新, 每轮采样顺序由 seed 和 epoch 决定."""
    sampler.seed(np.random.SeedSequence([seed, epoch]))
    totals = {}
    for _ in range(steps_per_epoch):
        agent, info = agent.update(sample_batch(sampler, config))
        metrics = {key: float(value) for key, value in jax.device_get(info).items()}
        if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError(f"Non-finite training metrics at epoch {epoch + 1}: {metrics}")
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        step = int(agent.critic.step)
        if logger is not None and step % log_every == 0:
            logger.log({"train_step": step, **{f"train/{key}": value for key, value in metrics.items()}})
    return agent, {key: value / steps_per_epoch for key, value in totals.items()}


def save_training_checkpoint(run_dir: Path, agent: Any, config: FQLWorkflowConfig, observations: Any, *, epoch: int, metrics: dict[str, float], keep: int, logger: Any, upload_artifact: bool, runtime: dict[str, Any]) -> Path:
    """保存完整训练状态, 可选上传 Artifact, 成功后清理旧的周期 checkpoint."""
    path = _common.save_checkpoint(run_dir / f"epoch_{epoch + 1:04d}.ckpt", agent=agent, epoch=epoch, config=config, sample_observations=observations, metrics=metrics, runtime=runtime)
    if upload_artifact and logger is not None:
        import wandb
        artifact = wandb.Artifact(f"{run_dir.name}-checkpoint", type="model", metadata={"epoch": epoch + 1})
        artifact.add_file(str(path))
        logger.run.log_artifact(artifact).wait()
    checkpoints = sorted(run_dir.glob("epoch_*.ckpt"), key=lambda item: int(item.stem.split("_")[-1]))
    for old in checkpoints[:-keep]:
        old.unlink()
    return path


"""------------------------- 新训练与续训调度 -------------------------"""

def run_training(args: argparse.Namespace) -> Path:
    """初始化或恢复 Agent, 执行训练并复用统一评估, 最终关闭日志."""
    resume = args.resume_checkpoint.expanduser().resolve() if args.resume_checkpoint else None
    saved_runtime = {}
    if resume is None:
        config, agent, start_epoch = build_config(args), None, 0
    else:
        agent, saved_config, metadata = _common.load_checkpoint(resume)
        config, start_epoch = build_config(args, saved_config), int(metadata["epoch"]) + 1
        saved_runtime = metadata.get("runtime") or load_run_config(resume.parent)["training"]
    if args.num_epochs <= start_epoch:
        raise ValueError("num-epochs must exceed the completed epoch count.")
    seed = args.seed if args.seed is not None else int(saved_runtime.get("seed", 0))

    # 加载紧凑轨迹, 采样器只选完整动作块, 不使用 BC 尾部填充.
    trajectories = _common.load_split_trajectories(config, "train")
    store = _common.make_store(trajectories)
    sampler = _common.make_bc_sampler(store, seed=seed)
    complete_chunks = int(np.maximum(store.trajectory_lengths - config.action_horizon + 1, 0).sum())
    if complete_chunks == 0:
        raise ValueError("Dataset contains no complete action chunks.")
    steps_per_epoch = args.steps_per_epoch or saved_runtime.get("steps_per_epoch") or math.ceil(complete_chunks / config.batch_size)
    example = sample_batch(sampler, config, batch_size=1)
    observations = jax.tree_util.tree_map(lambda value: value[0], example["observations"])
    if agent is None:
        agent = _common.create_agent(config, observations, example["actions"][0], seed=seed)

    # 新实验创建独立目录, 续训在原目录保存模型并单独记录本次配置.
    run_dir = create_run_dir(args.checkpoint_root, config.task_name, _common.ALGORITHM) if resume is None else resume.parent
    log_dir = run_dir if resume is None else run_dir / "continuations" / f"resume_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
    log_dir.mkdir(parents=True, exist_ok=True)
    runtime = {key: value for key, value in vars(args).items() if key not in {field.name for field in fields(FQLWorkflowConfig)}}
    runtime.update(seed=seed, steps_per_epoch=steps_per_epoch, start_epoch=start_epoch)
    # 记录数据文件列表, 续训采样可复现依赖原数据内容与顺序保持不变.
    run_config = {"schema_version": 1, "algorithm": _common.ALGORITHM, "config": asdict(config), "training": runtime, "dataset_files": [str(path.resolve()) for path in sorted(config.split_directory("train").glob(config.demo_pattern))]}
    save_run_config(log_dir, run_config)
    logger = None
    if args.wandb == "true":
        # 保留平台日志与上传, 隐藏 W&B 的常规终端提示.
        os.environ.setdefault("WANDB_SILENT", "true")
        from roboforg_1.utils.wandb import make_wandb_logger
        variant = {"config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}, "seed": seed, "steps_per_epoch": steps_per_epoch}
        logger = make_wandb_logger(project=args.wandb_project, entity=args.wandb_entity, description="fql", variant=variant, wandb_output_dir=log_dir, run_name=args.wandb_name or log_dir.name, run_id=uuid4().hex, debug=args.wandb_debug == "true")
    progress = _TrainingProgress(args.num_epochs)
    evaluation_count = 0
    try:
        if logger is not None:
            logger.run.define_metric("train_step", hidden=True)
            for name in ("train/*", "epoch/*", "evaluation/*"):
                logger.run.define_metric(name, step_metric="train_step")
            if not logger.run.disabled and not logger.run.offline:
                run_url = logger.run.url
                if run_url:
                    print(f"W&B 训练曲线: {run_url}", flush=True)
        progress.show(start_epoch)
        for epoch in range(start_epoch, args.num_epochs):
            started = time.perf_counter()
            agent, metrics = train_epoch(agent, sampler, config, epoch=epoch, steps_per_epoch=steps_per_epoch, seed=seed, logger=logger, log_every=args.wandb_log_every)
            completed, step = epoch + 1, int(agent.critic.step)
            summary = {"train_step": step, "epoch/completed": completed, "epoch/seconds": time.perf_counter() - started, **{f"epoch/{key}": value for key, value in metrics.items()}}
            if logger is not None:
                logger.log(summary)
            progress.show(completed)

            # 保存先于评估, 即使仿真评估失败也保留本轮训练状态.
            final_epoch = completed == args.num_epochs
            if completed % args.checkpoint_every_epochs == 0 or final_epoch:
                save_training_checkpoint(run_dir, agent, config, observations, epoch=epoch, metrics=metrics, keep=args.checkpoint_keep, logger=logger, upload_artifact=args.upload_checkpoint_artifact == "true", runtime={"seed": seed, "steps_per_epoch": steps_per_epoch})
            if (not final_epoch and args.eval_every_epochs > 0 and completed % args.eval_every_epochs == 0) or (final_epoch and args.final_eval == "true"):
                progress.pause()
                evaluation_count += 1
                print(f"第 {evaluation_count} 次测评 | 进行中...", end="", flush=True)
                try:
                    results = _eval.evaluate_policy(agent, config, num_episodes=args.num_eval_episodes, seed=seed + completed, show_viewer=args.show_viewer == "true", max_episode_steps=args.max_eval_episode_steps, verbose=False)
                except BaseException:
                    print(flush=True)
                    raise
                print(f"\r第 {evaluation_count} 次测评 | 成功率: {results['success_rate']:.1%}", flush=True)
                if logger is not None:
                    logger.log({"train_step": step, **{f"evaluation/{key}": value for key, value in results.items()}})
                if not final_epoch:
                    progress.show(completed)
    finally:
        progress.pause()
        if logger is not None:
            logger.finish()
    return run_dir


def main() -> None:
    """选择计算设备并启动工作流, 不设置资源监控或显存限制."""
    args = parse_args()
    device = jax.devices()[0] if args.device == "auto" else jax.devices(args.device)[0]
    with jax.default_device(device):
        run_training(args)


if __name__ == "__main__":
    main()
