"""ACT 端到端冒烟测试：用极少真实数据验证训练、评估与仿真执行链路。"""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import shutil
from typing import Any

import jax

#######################################
#              冒烟测试                #
#######################################


# 冒烟测试不打开 MuJoCo 窗口；在无桌面或服务器上默认使用 EGL 离屏渲染。
# 必须在导入会加载 MuJoCo 的 run_act 前设置，用户已显式设置时则尊重其选择。
os.environ.setdefault("MUJOCO_GL", "egl")

from roboforg.workflows.act.common import (
    ACTWorkflowConfig,
    create_act_agent,
    fit_train_normalizer,
    load_checkpoint_into_agent,
    load_split_trajectories,
    make_chunk_buffer,
    read_checkpoint_setup,
    sample_act_batch,
    save_checkpoint,
)
from roboforg.workflows.act.eval_act import evaluate_dataset
from roboforg.workflows.act.run_act import (
    batch_observation,
    make_evaluation_environment,
    run_rollout,
)
from roboforg.workflows.act.train_act import run_epoch
from roboforg.agent.act import ActionEnsemble


# 将正式 ACT 配置缩小为一次真实训练、评估和执行都可快速跑完的固定配置。
def make_smoke_config(dataset_root: Path) -> ACTWorkflowConfig:
    """Return the deliberately small configuration used only by this smoke test."""
    return replace(
        ACTWorkflowConfig(dataset_root=dataset_root),
        batch_size=1,  # 单个样本足以验证前向、反向和优化器更新。
    )


# 运行完整链路：读取真实轨迹、加载预训练视觉主干、更新一次、保存/读取、评估一次、仿真执行一步。
def run_smoke_test(
    *,
    dataset_root: Path,
    output_dir: Path,
    seed: int,
    rollout_steps: int,
    keep_checkpoint: bool,
) -> None:
    """Run the smallest meaningful end-to-end ACT test without touching formal checkpoints."""
    config = make_smoke_config(dataset_root)
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / "smoke.ckpt"

    # 冒烟目录必须是新目录，避免误覆盖正式训练的 checkpoint。
    if output_dir.exists():
        raise FileExistsError(
            f"Smoke output directory already exists: {output_dir}. "
            "Choose another --output-dir or remove this old smoke directory first."
        )
    output_dir.mkdir(parents=True)

    try:
        # 1) 读取真实 train/eval 演示，并仅用 train 拟合标准化统计量。
        train_trajectories = load_split_trajectories(config, "train")
        eval_trajectories = load_split_trajectories(config, "eval")
        train_buffer = make_chunk_buffer(train_trajectories, seed=seed)
        eval_buffer = make_chunk_buffer(eval_trajectories, seed=seed)
        normalizer = fit_train_normalizer(train_trajectories)
        example_batch = sample_act_batch(train_buffer, config)
        print(
            f"[1/5] data ready: train_trajectories={len(train_trajectories)}, "
            f"eval_trajectories={len(eval_trajectories)}, batch_size={config.batch_size}."
        )

        # 2) 初始化完整 Agent，并严格加载本地缓存的预训练 ResNet-10。
        rng = jax.random.key(seed)
        rng, initialization_rng = jax.random.split(rng)
        agent = create_act_agent(
            config,
            normalizer=normalizer,
            example_batch=example_batch,
            rng=initialization_rng,
            load_pretrained_backbone=True,
        )
        print("[2/5] ACT agent and pretrained ResNet-10 ready.")

        # 3) 只更新一个真实 chunk batch；这会覆盖 CVAE、策略、Transformer 和可微调视觉主干。
        agent, train_metrics, rng, train_batches = run_epoch(
            agent=agent,
            buffer=train_buffer,
            config=config,
            rng=rng,
            training=True,
            max_batches=1,
        )
        if train_batches != 1:
            raise RuntimeError("Smoke training did not process exactly one batch.")
        print(f"[3/5] one training update passed: {train_metrics}.")

        # 4) 经 checkpoint 往返后，对一批未参与训练的 eval 演示做只读损失计算。
        save_checkpoint(
            checkpoint_path,
            agent=agent,
            epoch=0,
            config=config,
            metrics={"smoke_train_loss": train_metrics["loss"]},
        )
        restored_config, restored_normalizer, _ = read_checkpoint_setup(checkpoint_path)
        eval_example_batch = sample_act_batch(eval_buffer, restored_config)
        rng, restore_rng = jax.random.split(rng)
        restored_agent = create_act_agent(
            restored_config,
            normalizer=restored_normalizer,
            example_batch=eval_example_batch,
            rng=restore_rng,
            load_pretrained_backbone=False,
        )
        restored_agent, _ = load_checkpoint_into_agent(checkpoint_path, restored_agent)
        rng, eval_rng = jax.random.split(rng)
        eval_metrics, eval_batches = evaluate_dataset(
            agent=restored_agent,
            buffer=eval_buffer,
            config=restored_config,
            rng=eval_rng,
            max_batches=1,
        )
        if eval_batches != 1:
            raise RuntimeError("Smoke evaluation did not process exactly one batch.")
        print(f"[4/5] checkpoint round-trip and one eval batch passed: {eval_metrics}.")

        # 5) 创建真实 Pick 环境，预测 action chunk、做 temporal ensemble，并向环境执行至少一步。
        env = make_evaluation_environment(show_viewer=False)
        try:
            ensemble = ActionEnsemble(
                restored_config.action_horizon,
                action_dim=restored_config.action_dim,
                decay=0.01,
            )
            rollout_result = run_rollout(
                env=env,
                agent=restored_agent,
                ensemble=ensemble,
                seed=seed,
                max_steps=rollout_steps,
                query_every=1,
            )
        finally:
            env.close()
        print(f"[5/5] simulator rollout passed: {rollout_result}.")
        print("ACT end-to-end smoke test PASSED.")
    finally:
        # 默认清理临时模型，保证它绝不与正式训练 checkpoint 混在一起。
        if output_dir.exists() and not keep_checkpoint:
            shutil.rmtree(output_dir)


# 解析固定冒烟测试仅需的路径和随机种子；不暴露正式训练超参数，防止误把它当成长训练入口。
def parse_args() -> argparse.Namespace:
    """Parse arguments for the fixed end-to-end ACT smoke test."""
    parser = argparse.ArgumentParser(description="Run one end-to-end RoboForge ACT smoke test.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/act_smoke"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout-steps", type=int, default=1)
    parser.add_argument(
        "--keep-checkpoint",
        action="store_true",
        help="Keep the temporary smoke checkpoint for manual inspection.",
    )
    args = parser.parse_args()
    if args.rollout_steps <= 0:
        parser.error("--rollout-steps must be positive.")
    return args


# 命令行入口：只运行一次端到端检查，成功后打印明确标记并默认删除临时 checkpoint。
def main() -> None:
    """Run the configured ACT end-to-end smoke test."""
    args = parse_args()
    run_smoke_test(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        seed=args.seed,
        rollout_steps=args.rollout_steps,
        keep_checkpoint=args.keep_checkpoint,
    )


if __name__ == "__main__":
    main()
