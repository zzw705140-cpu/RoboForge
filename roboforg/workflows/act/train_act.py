"""ACT 的训练入口：按 epoch 更新模型、评估并保存 checkpoint。"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import time
from typing import Any

import jax

from roboforg.workflows.act.common import (
    ACTWorkflowConfig,
    create_act_agent,
    fit_train_normalizer,
    load_checkpoint_into_agent,
    load_split_trajectories,
    make_chunk_buffer,
    mean_metrics,
    read_checkpoint_setup,
    sample_act_batch,
    save_checkpoint,
)
from roboforg.data.act_data.data_conversion import ACTBatchConverter
from roboforg.workflows.act.training_logger import ACTTrainingLogger


# 遍历一次 buffer 的全部合法 chunk 样本；training=True 时才更新 Agent 参数。
def run_epoch(
    *,
    agent,
    buffer,
    config: ACTWorkflowConfig,
    rng: jax.Array,
    training: bool,
    max_batches: int | None = None,
    logger: ACTTrainingLogger | None = None,
) -> tuple[Any, dict[str, float], jax.Array, int]:
    """Run one train or eval pass and return the resulting agent, metrics, RNG, and batch count."""
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided.")

    # 一次 iterator 覆盖每个合法动作 chunk 起点；训练打乱，eval 保持固定顺序以便可比较。
    chunk_iterator = buffer.get_epoch_iterator(
        batch_size=config.batch_size,
        observation_horizon=config.observation_horizon,
        action_horizon=config.action_horizon,
        shuffle=training,
        drop_last=training,
    )
    converter = ACTBatchConverter(config.action_horizon, image_keys=config.camera_keys)
    metric_history: list[dict[str, Any]] = []

    for batch_index, chunk_batch in enumerate(chunk_iterator):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = converter(chunk_batch)
        rng, batch_rng = jax.random.split(rng)
        if training:
            agent, metrics = agent.update(batch, batch_rng)
            # 优化器 step 随 checkpoint 恢复，日志横轴无需另行维护计数器。
            if logger is not None:
                logger.log_train_step(metrics, step=int(agent.policy_state.step))
        else:
            metrics = agent.evaluate(batch, batch_rng)
        metric_history.append(metrics)

    return agent, mean_metrics(metric_history), rng, len(metric_history)


# 从命令行参数生成本次运行使用的 workflow 配置；恢复训练时以 checkpoint 配置为基础，只覆盖用户显式传入的字段。
def build_config(
    args: argparse.Namespace, *, base_config: ACTWorkflowConfig | None = None
) -> ACTWorkflowConfig:
    """Create the ACT configuration used by this training run."""
    base = ACTWorkflowConfig() if base_config is None else base_config
    updates = {
        name: value
        for name, value in {
            "dataset_root": args.dataset_root,
            "batch_size": args.batch_size,
            "action_horizon": args.action_horizon,
            "learning_rate": args.learning_rate,
            "backbone_learning_rate": args.backbone_learning_rate,
            "kl_weight": args.kl_weight,
        }.items()
        if value is not None
    }
    return replace(base, **updates)


# 解析训练所需的路径、超参数和恢复选项；默认训练四轮。
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for ACT training."""
    parser = argparse.ArgumentParser(description="Train RoboForge ACT on Pick demonstrations.")
    parser.add_argument("--dataset-root", type=Path, help="Dataset root; defaults to datasets for a new run.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/act"))
    parser.add_argument("--resume", type=Path, help="Resume parameters and optimizer state from this checkpoint.")
    parser.add_argument("--num-epochs", type=int, default=4, help="Number of additional epochs to train.")
    parser.add_argument("--batch-size", type=int, help="Batch size; checkpoint value is reused on resume.")
    parser.add_argument("--action-horizon", type=int, help="Chunk length; must match the checkpoint on resume.")
    parser.add_argument("--learning-rate", type=float, help="Main-network learning rate.")
    parser.add_argument("--backbone-learning-rate", type=float, help="ResNet fine-tuning learning rate.")
    parser.add_argument("--kl-weight", type=float, help="CVAE KL-loss weight.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1, help="Evaluate every N completed epochs.")
    parser.add_argument("--max-train-batches", type=int, help="Limit train batches per epoch for a smoke test.")
    parser.add_argument("--max-eval-batches", type=int, help="Limit eval batches per pass for a smoke test.")
    parser.add_argument(
        "--no-pretrained-backbone",
        action="store_true",
        help="Start with random ResNet-10 weights; only useful for code smoke tests.",
    )
    # 默认不上传；启用后使用 ACTTrainingLogger 统一管理实验。
    parser.add_argument("--wandb", action="store_true", help="Upload ACT metrics to W&B.")
    parser.add_argument("--wandb-debug", action="store_true", help="Exercise logging without uploading.")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-log-every", type=int, default=10)
    args = parser.parse_args()
    if args.wandb_log_every <= 0:
        parser.error("--wandb-log-every must be positive.")
    if args.num_epochs <= 0 or args.eval_every <= 0:
        parser.error("--num-epochs and --eval-every must be positive.")
    for name in ("batch_size", "action_horizon"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.max_train_batches is not None and args.max_train_batches <= 0:
        parser.error("--max-train-batches must be positive.")
    if args.max_eval_batches is not None and args.max_eval_batches <= 0:
        parser.error("--max-eval-batches must be positive.")
    return args


# 训练总调度：新训练加载预训练 ResNet；恢复训练加载完整 checkpoint，再按 epoch 保存 latest/best。
def main() -> None:
    """Run the complete ACT train/eval/checkpoint loop."""
    args = parse_args()
    if args.resume is not None:
        checkpoint_config, _, _ = read_checkpoint_setup(args.resume)
        if args.action_horizon is not None and args.action_horizon != checkpoint_config.action_horizon:
            raise ValueError(
                "Cannot change action_horizon while resuming: it changes action-query and output shapes."
            )
        config = build_config(args, base_config=checkpoint_config)
    else:
        config = build_config(args)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 只用 train 轨迹拟合 normalizer；eval 轨迹只用于未参与参数更新的模型检查。
    train_trajectories = load_split_trajectories(config, "train")
    eval_trajectories = load_split_trajectories(config, "eval")
    train_buffer = make_chunk_buffer(train_trajectories, seed=args.seed)
    eval_buffer = make_chunk_buffer(eval_trajectories, seed=args.seed)
    normalizer = fit_train_normalizer(train_trajectories)
    example_batch = sample_act_batch(train_buffer, config)

    # 先用示例 batch 初始化正确形状的参数树；恢复时该树作为 checkpoint 严格形状检查的模板。
    rng = jax.random.key(args.seed)
    rng, initialization_rng = jax.random.split(rng)
    agent = create_act_agent(
        config,
        normalizer=normalizer,
        example_batch=example_batch,
        rng=initialization_rng,
        load_pretrained_backbone=not args.no_pretrained_backbone and args.resume is None,
    )
    start_epoch = 0
    best_eval_loss = float("inf")

    # 恢复时保留 checkpoint 的 normalizer、网络权重和 AdamW 状态；采样顺序从当前 seed 重新开始。
    if args.resume is not None:
        agent, metadata = load_checkpoint_into_agent(args.resume, agent)
        start_epoch = metadata["epoch"] + 1
        best_eval_loss = float(
            metadata["metrics"].get(
                "best_eval_loss", metadata["metrics"].get("eval_loss", float("inf"))
            )
        )
        print(f"Resumed {args.resume} after completed epoch {metadata['epoch']}.")

    print(
        f"ACT training: train_trajectories={len(train_trajectories)}, "
        f"eval_trajectories={len(eval_trajectories)}, batch_size={config.batch_size}, "
        f"action_horizon={config.action_horizon}, start_epoch={start_epoch}."
    )

    # 日志在训练配置与 checkpoint 恢复完成后创建；异常退出时同样收尾。
    logger = ACTTrainingLogger.create(
        config=config, checkpoint_dir=args.checkpoint_dir, resume_checkpoint=args.resume,
        enabled=args.wandb or args.wandb_debug, debug=args.wandb_debug,
        project=args.wandb_project, entity=args.wandb_entity, run_name=args.wandb_name,
        log_every=args.wandb_log_every, seed=args.seed,
        train_count=len(train_trajectories), eval_count=len(eval_trajectories),
    )
    try:
        for epoch in range(start_epoch, start_epoch + args.num_epochs):
            epoch_start = time.perf_counter()
            agent, train_metrics, rng, train_batches = run_epoch(
                agent=agent,
                buffer=train_buffer,
                config=config,
                rng=rng,
                training=True,
                max_batches=args.max_train_batches,
                logger=logger,
            )
            summary: dict[str, Any] = {f"train_{key}": value for key, value in train_metrics.items()}

            # 每隔 eval_every 轮检查未见轨迹；best checkpoint 只由 eval total loss 决定。
            if (epoch + 1) % args.eval_every == 0:
                agent, eval_metrics, rng, eval_batches = run_epoch(
                    agent=agent,
                    buffer=eval_buffer,
                    config=config,
                    rng=rng,
                    training=False,
                    max_batches=args.max_eval_batches,
                )
                summary.update({f"eval_{key}": value for key, value in eval_metrics.items()})
                if eval_metrics["loss"] < best_eval_loss:
                    best_eval_loss = eval_metrics["loss"]
                    save_checkpoint(
                        args.checkpoint_dir / "best.ckpt",
                        agent=agent,
                        epoch=epoch,
                        config=config,
                        metrics=summary,
                    )
                    print(f"Saved new best checkpoint (eval_loss={best_eval_loss:.6f}).")
            else:
                eval_batches = 0

            # 即使本轮未做 eval，也把历史最优值存入 latest，恢复训练时不会误覆盖原有 best.ckpt。
            summary["best_eval_loss"] = best_eval_loss

            # 无论 eval 是否运行都保存 latest，便于在服务器中断后恢复本次训练进度。
            save_checkpoint(
                args.checkpoint_dir / "latest.ckpt",
                agent=agent,
                epoch=epoch,
                config=config,
                metrics=summary,
            )
            elapsed = time.perf_counter() - epoch_start
            logger.log_epoch(
                summary, step=int(agent.policy_state.step), epoch=epoch,
                seconds=elapsed, train_batches=train_batches, eval_batches=eval_batches,
            )
            metric_text = ", ".join(f"{key}={value:.6f}" for key, value in sorted(summary.items()))
            print(
                f"epoch={epoch} train_batches={train_batches} eval_batches={eval_batches} "
                f"seconds={elapsed:.2f} {metric_text}"
            )

    finally:
        logger.finish()

if __name__ == "__main__":
    main()
