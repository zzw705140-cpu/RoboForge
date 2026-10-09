"""ACT 离线评估入口：在 eval 演示上计算 loss，但绝不更新参数。"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax

from roboforg.data.act_data.data_conversion import ACTBatchConverter
from roboforg.workflows.act.common import (
    ACTWorkflowConfig,
    create_act_agent,
    load_checkpoint_into_agent,
    load_split_trajectories,
    make_chunk_buffer,
    mean_metrics,
    read_checkpoint_setup,
    sample_act_batch,
)


# 逐 batch 调用 Agent 的只读 evaluate 接口，汇总整个 eval split 的 BC、KL 和 total loss。
def evaluate_dataset(
    *,
    agent: Any,
    buffer: Any,
    config: ACTWorkflowConfig,
    rng: jax.Array,
    max_batches: int | None = None,
) -> tuple[dict[str, float], int]:
    """Evaluate every requested batch without modifying the Agent parameter states."""
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided.")

    # eval 不打乱，且保留最后不足 batch_size 的小批次，使每个合法 eval 样本都参与统计。
    chunk_iterator = buffer.get_epoch_iterator(
        batch_size=config.batch_size,
        observation_horizon=config.observation_horizon,
        action_horizon=config.action_horizon,
        shuffle=False,
        drop_last=False,
    )
    converter = ACTBatchConverter(config.action_horizon, image_keys=config.camera_keys)
    metric_history: list[dict[str, Any]] = []
    for batch_index, chunk_batch in enumerate(chunk_iterator):
        if max_batches is not None and batch_index >= max_batches:
            break
        rng, batch_rng = jax.random.split(rng)
        # evaluate 不进行 apply_gradients，传入的 agent 参数和优化器状态保持完全不变。
        metric_history.append(agent.evaluate(converter(chunk_batch), batch_rng))
    return mean_metrics(metric_history), len(metric_history)


# 解析评估目标 checkpoint 和可选的 eval 数据路径、batch 大小覆盖项。
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for standalone ACT evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a RoboForge ACT checkpoint on Pick eval demonstrations.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, help="Override the dataset root stored in the checkpoint.")
    parser.add_argument("--batch-size", type=int, help="Override only evaluation batch size.")
    parser.add_argument("--max-batches", type=int, help="Evaluate only this many batches for a smoke test.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.max_batches is not None and args.max_batches <= 0:
        parser.error("--max-batches must be positive.")
    return args


# 评估总调度：从 checkpoint 还原结构与统计量，创建同形 Agent，加载权重，再扫描 eval 数据。
def main() -> None:
    """Load one checkpoint and report its offline eval losses."""
    args = parse_args()

    # checkpoint 是配置的唯一来源，避免把 action_horizon、层数等结构参数手工写错。
    config, normalizer, metadata = read_checkpoint_setup(args.checkpoint)
    if args.dataset_root is not None:
        config = replace(config, dataset_root=args.dataset_root)
    if args.batch_size is not None:
        config = replace(config, batch_size=args.batch_size)

    # 使用 eval 的一个 batch 只为初始化同形参数树；normalizer 必须使用 checkpoint 中的 train 统计量。
    eval_trajectories = load_split_trajectories(config, "eval")
    eval_buffer = make_chunk_buffer(eval_trajectories, seed=args.seed)
    example_batch = sample_act_batch(eval_buffer, config)
    rng = jax.random.key(args.seed)
    rng, initialization_rng = jax.random.split(rng)
    agent = create_act_agent(
        config,
        normalizer=normalizer,
        example_batch=example_batch,
        rng=initialization_rng,
        load_pretrained_backbone=False,
    )
    agent, _ = load_checkpoint_into_agent(args.checkpoint, agent)

    metrics, batch_count = evaluate_dataset(
        agent=agent,
        buffer=eval_buffer,
        config=config,
        rng=rng,
        max_batches=args.max_batches,
    )
    metric_text = ", ".join(f"{key}={value:.6f}" for key, value in sorted(metrics.items()))
    print(
        f"ACT eval: checkpoint_epoch={metadata['epoch']}, trajectories={len(eval_trajectories)}, "
        f"batches={batch_count}, {metric_text}"
    )


if __name__ == "__main__":
    main()
