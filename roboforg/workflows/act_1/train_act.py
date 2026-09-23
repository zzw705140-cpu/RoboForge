# ACT 新训练入口：统一接收参数，创建按时间命名的运行目录，并保存完整训练状态。
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import re
import time
from typing import Any
from uuid import uuid4

import jax

from roboforg.data.act_data.data_conversion import ACTBatchConverter
from roboforg.utils.checkpoint_untils import create_run_dir, save_run_config
from roboforg.utils.wandb import make_wandb_logger
from roboforg.workflows.act.common import (
    ACTWorkflowConfig,
    CHECKPOINT_EVERY_EPOCHS,
    CHECKPOINT_KEEP_LAST,
    DEFAULT_NUM_EPOCHS,
    HEADLESS_NUM_ROLLOUTS,
    ROLLOUT_EVERY_EPOCHS,
    create_act_agent,
    fit_train_normalizer,
    load_split_trajectories,
    make_chunk_buffer,
    mean_metrics,
    sample_act_batch,
    save_checkpoint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints" / "act"


# 训练入口的默认值沿用旧 ACT；启动脚本可覆盖每一项常用配置。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a new ACT run on Pick demonstrations.")

    # 数据与 ACT 模型参数。
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "datasets")
    parser.add_argument("--task-name", default="pick")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--action-horizon", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--kl-weight", type=float, default=10.0)

    # 训练节奏、离屏成功率评估与 checkpoint 保留策略。
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--eval-every-epochs", type=int, default=ROLLOUT_EVERY_EPOCHS)
    parser.add_argument("--num-eval-episodes", type=int, default=HEADLESS_NUM_ROLLOUTS)
    parser.add_argument("--final-eval", choices=("true", "false"), default="true")    # 控制训练最后一轮是否进行离屏测评；默认开启，冒烟测试可关闭。
    parser.add_argument("--checkpoint-every-epochs", type=int, default=CHECKPOINT_EVERY_EPOCHS)
    parser.add_argument("--checkpoint-keep", type=int, default=CHECKPOINT_KEEP_LAST)

    # W&B 使用服务器上的登录凭据；不设置固定 entity 或在代码中存放 API key。
    parser.add_argument("--wandb-project", default="roboforge")
    parser.add_argument("--wandb-log-every", type=int, default=10)

    args = parser.parse_args()
    for name in (
        "batch_size", "action_horizon", "num_epochs", "eval_every_epochs",
        "num_eval_episodes", "checkpoint_every_epochs", "checkpoint_keep",
        "wandb_log_every",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.learning_rate <= 0 or args.backbone_learning_rate <= 0 or args.kl_weight < 0:
        parser.error("learning rates must be positive and KL weight must be non-negative")
    if not args.wandb_project.strip():
        parser.error("--wandb-project must not be empty")
    return args


# 用训练入口参数构建旧 ACT 算法所需的模型配置，未暴露的结构参数沿用原默认值。
def build_config(args: argparse.Namespace) -> ACTWorkflowConfig:
    return replace(
        ACTWorkflowConfig(),
        dataset_root=args.dataset_root.expanduser().resolve(),
        task_name=args.task_name,
        batch_size=args.batch_size,
        action_horizon=args.action_horizon,
        learning_rate=args.learning_rate,
        backbone_learning_rate=args.backbone_learning_rate,
        kl_weight=args.kl_weight,
    )


# 每轮遍历训练集的全部合法动作块，逐 batch 更新 ACT 参数并记录 W&B 指标。
def train_epoch(agent: Any, buffer: Any, config: ACTWorkflowConfig, rng: jax.Array, logger: Any, log_every: int):
    converter = ACTBatchConverter(config.action_horizon, image_keys=config.camera_keys)
    history = []
    for chunk_batch in buffer.get_epoch_iterator(
        batch_size=config.batch_size,
        observation_horizon=config.observation_horizon,
        action_horizon=config.action_horizon,
        shuffle=True,
        drop_last=True,
    ):
        rng, update_rng = jax.random.split(rng)
        agent, metrics = agent.update(converter(chunk_batch), update_rng)
        history.append(metrics)
        step = int(agent.policy_state.step)
        if step % log_every == 0:
            logger.log({"train_step": step, "train": {key: float(value) for key, value in metrics.items()}})
    return agent, mean_metrics(history), rng, len(history)


# 按完成的 epoch 保存完整 ACT checkpoint，再只保留最近的指定份数。
def save_training_checkpoint(run_dir: Path, agent: Any, epoch: int, config: ACTWorkflowConfig,
                             metrics: dict[str, float], keep: int) -> Path:
    path = run_dir / f"epoch_{epoch + 1:04d}.ckpt"
    save_checkpoint(path, agent=agent, epoch=epoch, config=config, metrics=metrics)
    numbered = sorted(
        ((int(match.group(1)), item) for item in run_dir.iterdir()
         if item.is_file() and (match := re.fullmatch(r"epoch_(\d+)\.ckpt", item.name))),
        key=lambda item: item[0],
    )
    for _, expired in numbered[:-keep]:
        expired.unlink()
    return path


# 新训练：建目录和配置，准备数据与模型，开启 W&B，再训练、评估并保存 checkpoint。
def main() -> None:
    args = parse_args()
    config = build_config(args)
    run_dir = create_run_dir(CHECKPOINT_ROOT, config.task_name, "act")
    wandb_run_id = uuid4().hex
    config_values = asdict(config)
    config_values["dataset_root"] = str(config.dataset_root)
    save_run_config(run_dir, {
        "schema_version": 1,
        "algorithm": "act",
        "config": config_values,
        "training": {
            "seed": args.seed,
            "num_epochs": args.num_epochs,
            "eval_every_epochs": args.eval_every_epochs,
            "final_eval": args.final_eval,                           # 记录本次训练是否启用最终测评，便于之后查看实验设置。
            "num_eval_episodes": args.num_eval_episodes,
            "checkpoint_every_epochs": args.checkpoint_every_epochs,
            "checkpoint_keep": args.checkpoint_keep,
            "wandb_log_every": args.wandb_log_every,
        },
        "wandb": {"project": args.wandb_project, "run_id": wandb_run_id},
    })

    trajectories = load_split_trajectories(config, "train")
    buffer = make_chunk_buffer(trajectories, seed=args.seed)
    normalizer = fit_train_normalizer(trajectories)
    example_batch = sample_act_batch(buffer, config)
    rng = jax.random.key(args.seed)
    rng, initialization_rng = jax.random.split(rng)
    agent = create_act_agent(
        config, normalizer=normalizer, example_batch=example_batch,
        rng=initialization_rng, load_pretrained_backbone=True,
    )

    # W&B 初始化失败直接终止训练；run ID 已写入本地配置，API key 从服务器读取。
    logger = make_wandb_logger(
        project=args.wandb_project,
        description="act",
        variant={"act_config": config_values, "seed": args.seed},
        run_name=run_dir.name,
        run_id=wandb_run_id,
        wandb_output_dir=run_dir,
    )
    try:
        print(f"ACT run directory: {run_dir}", flush=True)
        print(f"W&B destination: {logger.run.path}; URL: {logger.run.url}", flush=True)
        logger.run.define_metric("train_step", hidden=True)
        logger.run.define_metric("train/*", step_metric="train_step")
        logger.run.define_metric("epoch/*", step_metric="train_step")
        logger.run.define_metric("evaluation/*", step_metric="train_step")

        for epoch in range(args.num_epochs):
            started = time.perf_counter()
            agent, train_metrics, rng, batch_count = train_epoch(
                agent, buffer, config, rng, logger, args.wandb_log_every,
            )
            completed = epoch + 1
            step = int(agent.policy_state.step)
            logger.log({
                "train_step": step,
                "epoch": {"completed": completed, "batches": batch_count,
                          "seconds": time.perf_counter() - started,
                          **{f"train_{key}": value for key, value in train_metrics.items()}},
            })
            print(f"epoch={completed} batches={batch_count} step={step} metrics={train_metrics}", flush=True)

            final_epoch = completed == args.num_epochs
            if completed % args.checkpoint_every_epochs == 0 or final_epoch:
                path = save_training_checkpoint(
                    run_dir, agent, epoch, config, train_metrics, args.checkpoint_keep,
                )
                print(f"Saved checkpoint: {path}", flush=True)
            if (                                                     # 中途按设定间隔测评；最后一轮是否测评由 final_eval 单独控制。
                (not final_epoch and completed % args.eval_every_epochs == 0)
                or (final_epoch and args.final_eval == "true")
            ):
                from roboforg.workflows.act_1.eval_act import evaluate_policy
                results = evaluate_policy(agent, num_episodes=args.num_eval_episodes,
                                          seed=args.seed + completed)
                logger.log({"train_step": step, "evaluation": {"epoch": completed, **results}})
                print(f"Evaluation after epoch {completed}: {results}", flush=True)
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
