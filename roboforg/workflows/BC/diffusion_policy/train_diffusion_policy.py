"""Diffusion Policy 完整训练入口。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import re
import time
from uuid import uuid4

import jax

from roboforg.agent.diffusion_policy import DataNormalizer
from roboforg.utils.checkpoint_untils import create_run_dir, save_run_config
from roboforg.utils.wandb import make_wandb_logger
from roboforg.workflows.diffusion_policy.common import (
    DiffusionWorkflowConfig,
    checkpoint_setup,
    convert_batch,
    create_policy,
    initialize_state,
    load_split_trajectories,
    make_sampler,
    mean_metrics,
    restore_training_state,
    save_checkpoint,
)
from roboforg.workflows.diffusion_policy.train_steps import train_step


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints" / "diffusion_policy"


# 读取命令行中的数据、模型、优化器、训练节奏与日志配置，并检查取值是否合法。
def parse_args() -> argparse.Namespace:
    # 创建参数解析器；这些配置主要由 train_diffusion_policy.sh 传入。
    parser = argparse.ArgumentParser(description="Train or resume Diffusion Policy.")

    # 数据集、任务、随机种子、续训权重和视觉主干预训练权重。
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "datasets")
    parser.add_argument("--task-name", default="pick")
    parser.add_argument("--train-dir-name", default="train_1")
    parser.add_argument("--eval-dir-name", default="eval_1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--pretrained-path", type=Path,
                        default=PROJECT_ROOT / ".resnet_params" / "resnet10_params.pkl")

    # batch、观测/动作时间窗口以及一维 U-Net 的结构参数。
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--down-channels", default="256,512,1024")
    parser.add_argument("--diffusion-step-embedding-dim", type=int, default=128)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--n-groups", type=int, default=8)
    parser.add_argument("--predict-scale", choices=("true", "false"), default="true")

    # AdamW 优化器和 EMA 参数平滑配置。
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.95)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--ema-update-after-step", type=int, default=0)
    parser.add_argument("--ema-inv-gamma", type=float, default=1.0)
    parser.add_argument("--ema-power", type=float, default=0.75)
    parser.add_argument("--ema-min-decay", type=float, default=0.0)
    parser.add_argument("--ema-max-decay", type=float, default=0.9999)

    # 总训练轮数、仿真评估、checkpoint 保存和 W&B 日志配置。
    parser.add_argument("--num-epochs", type=int, default=700)
    parser.add_argument("--eval-every-epochs", type=int, default=100)
    parser.add_argument("--num-eval-episodes", type=int, default=40)
    parser.add_argument("--final-eval", choices=("true", "false"), default="true")
    parser.add_argument("--checkpoint-every-epochs", type=int, default=200)
    parser.add_argument("--checkpoint-keep", type=int, default=2)
    parser.add_argument("--save-training-state", choices=("true", "false"), default="false")
    parser.add_argument("--wandb-project", default="roboforge")
    parser.add_argument("--wandb-log-every", type=int, default=200)
    args = parser.parse_args()

    # 所有表示数量或间隔的参数都必须是正整数。
    integer_fields = (
        "batch_size", "n_obs_steps", "horizon", "n_action_steps", "num_inference_steps",
        "diffusion_step_embedding_dim", "kernel_size", "n_groups", "num_epochs",
        "eval_every_epochs", "num_eval_episodes", "checkpoint_every_epochs",
        "checkpoint_keep", "wandb_log_every",
    )
    for field in integer_fields:
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive.")

    # 随机种子和 EMA 延迟步数允许为零，但不能为负数。
    if args.seed < 0 or args.ema_update_after_step < 0:
        parser.error("seed and ema-update-after-step must be non-negative.")

    # 将逗号分隔的通道字符串转换成 U-Net 使用的整数元组。
    try:
        args.down_channels = tuple(int(value.strip()) for value in args.down_channels.split(","))
    except ValueError:
        parser.error("--down-channels must be comma-separated integers.")
    if len(args.down_channels) < 2 or min(args.down_channels) <= 0:
        parser.error("--down-channels requires at least two positive values.")
    return args


# 将命令行参数整理为训练、评估和 checkpoint 共用的 workflow 配置。
def build_config(args: argparse.Namespace) -> DiffusionWorkflowConfig:
    return DiffusionWorkflowConfig(
        dataset_root=args.dataset_root.expanduser().resolve(), task_name=args.task_name,
        train_dir_name=args.train_dir_name, eval_dir_name=args.eval_dir_name,
        n_obs_steps=args.n_obs_steps, horizon=args.horizon, n_action_steps=args.n_action_steps,
        num_inference_steps=args.num_inference_steps,
        diffusion_step_embedding_dim=args.diffusion_step_embedding_dim,
        down_channels=args.down_channels, kernel_size=args.kernel_size, n_groups=args.n_groups,
        predict_scale=args.predict_scale == "true", batch_size=args.batch_size,
        learning_rate=args.learning_rate, beta1=args.beta1, beta2=args.beta2,
        adam_epsilon=args.adam_epsilon, weight_decay=args.weight_decay,
        ema_update_after_step=args.ema_update_after_step, ema_inv_gamma=args.ema_inv_gamma,
        ema_power=args.ema_power, ema_min_decay=args.ema_min_decay,
        ema_max_decay=args.ema_max_decay,
    )


# 完整遍历一轮训练集，逐 batch 更新普通参数和 EMA 参数并汇总指标。
def train_epoch(state, policy, sampler, config, rng, logger, log_every):
    # 保存本轮各 batch 指标，供 epoch 结束时求平均值。
    history = []
    batches = 0

    # 无放回遍历合法时间窗口，丢弃不足一个完整 batch 的尾部样本。
    for raw_batch in sampler.get_epoch_iterator(
        batch_size=config.batch_size, shuffle=True, drop_last=True,
    ):
        # 将采样结果转换成策略需要的观测字典和动作序列。
        observations, actions = convert_batch(raw_batch, config)

        # 拆出当前更新使用的随机键，再计算梯度、更新 AdamW 和 EMA。
        rng, update_rng = jax.random.split(rng)
        state, metrics = train_step(
            state, observations, actions, update_rng, policy=policy,
            update_after_step=config.ema_update_after_step,
            inv_gamma=config.ema_inv_gamma, power=config.ema_power,
            min_decay=config.ema_min_decay, max_decay=config.ema_max_decay,
        )
        history.append(metrics)
        batches += 1
        step = int(state.step)

        # 按参数更新次数向 W&B 记录 batch 级训练指标。
        if step % log_every == 0:
            logger.log({"train_step": step, "train": {
                key: float(value) for key, value in metrics.items()
            }})
    return state, mean_metrics(history), rng, batches


# 保存当前 epoch，并按照保留数量清理更早的 checkpoint。
def save_training_checkpoint(run_dir: Path, state, epoch: int, config, normalizer,
                             metrics, *, keep: int, save_training_state: bool) -> Path:
    # EMA 始终保存；是否额外保存普通参数和优化器状态由开关决定。
    path = run_dir / f"epoch_{epoch + 1:04d}.ckpt"
    save_checkpoint(
        path, state=state, epoch=epoch, config=config, normalizer=normalizer,
        metrics=metrics, save_training_state=save_training_state,
    )

    # 只匹配当前实验目录下的 epoch_XXXX.ckpt，并按 epoch 编号排序。
    numbered = sorted(
        ((int(match.group(1)), item) for item in run_dir.iterdir()
         if item.is_file() and (match := re.fullmatch(r"epoch_(\d+)\.ckpt", item.name))),
        key=lambda pair: pair[0],
    )

    # 删除超出保留数量的最早 checkpoint。
    for _, expired in numbered[:-keep]:
        expired.unlink()
    return path


# 组织新训练或续训的完整流程：建目录、训练、评估、保存并关闭日志。
def main() -> None:
    # 读取参数并建立统一配置。
    args = parse_args()
    config = build_config(args)
    resume = args.resume_checkpoint.expanduser().resolve() if args.resume_checkpoint else None

    # 未指定恢复点时，按照当前时间建立新的实验目录。
    if resume is None:
        start_epoch = 0
        normalizer = DataNormalizer()
        run_dir = create_run_dir(CHECKPOINT_ROOT, config.task_name, "diffusion_policy")
        logger_dir = run_dir
    else:
        # 续训读取原配置和 normalizer，并拒绝不匹配或仅含 EMA 的 checkpoint。
        saved_config, normalizer, metadata = checkpoint_setup(resume)
        if config != saved_config:
            raise ValueError("Resume configuration must exactly match the checkpoint configuration.")
        if not metadata["has_training_state"]:
            raise ValueError("Cannot resume from an EMA-only checkpoint.")
        start_epoch = metadata["epoch"] + 1
        if args.num_epochs <= start_epoch:
            raise ValueError("num-epochs is the target total and must exceed completed epochs.")
        run_dir = resume.parent

        # 续训沿用原权重目录，但为本次日志创建独立子目录。
        logger_dir = run_dir / "continuations" / datetime.now().strftime("resume_%Y%m%d_%H%M%S")
        logger_dir.mkdir(parents=True, exist_ok=False)

    # 将配置转换成可写入 JSON 和 W&B 的普通数据。
    config_values = asdict(config)
    config_values["dataset_root"] = str(config.dataset_root)
    wandb_run_id = uuid4().hex

    # 保存本次运行的模型、训练、checkpoint 和 W&B 配置。
    save_run_config(logger_dir, {
        "schema_version": 1,
        "algorithm": "diffusion_policy",
        "config": config_values,
        "training": {
            "seed": args.seed, "num_epochs": args.num_epochs,
            "eval_every_epochs": args.eval_every_epochs,
            "num_eval_episodes": args.num_eval_episodes, "final_eval": args.final_eval,
            "checkpoint_every_epochs": args.checkpoint_every_epochs,
            "checkpoint_keep": args.checkpoint_keep,
            "save_training_state": args.save_training_state,
            "wandb_log_every": args.wandb_log_every,
        },
        "resume_checkpoint": None if resume is None else str(resume),
        "wandb": {"project": args.wandb_project, "run_id": wandb_run_id},
    })

    # 加载训练轨迹，创建时间窗口采样器，并抽取初始化模型所需的样例观测。
    trajectories = load_split_trajectories(config, "train")
    sampler = make_sampler(trajectories, config, seed=args.seed)
    example_observations, _ = convert_batch(sampler.sample(1), config)

    # 创建策略和训练状态；只有新训练会加载 ResNet-10 预训练权重。
    policy = create_policy(config, normalizer=normalizer)
    rng = jax.random.key(args.seed)
    rng, initialization_rng = jax.random.split(rng)
    state = initialize_state(
        policy, config, initialization_rng, example_observations,
        pretrained_path=None if resume is not None else args.pretrained_path.expanduser().resolve(),
    )

    # 续训时恢复普通参数、EMA、优化器状态和全局更新步数。
    if resume is not None:
        state = restore_training_state(resume, state)

    # 创建 W&B run；认证信息由当前机器保存的 wandb 登录凭据提供。
    logger = make_wandb_logger(
        project=args.wandb_project, description="diffusion_policy",
        variant={"diffusion_policy_config": config_values, "seed": args.seed},
        run_id=wandb_run_id, wandb_output_dir=logger_dir, run_name=logger_dir.name,
    )
    try:
        # 让训练、epoch 和评估曲线统一使用 train_step 作为横轴。
        logger.run.define_metric("train_step", hidden=True)
        logger.run.define_metric("train/*", step_metric="train_step")
        logger.run.define_metric("epoch/*", step_metric="train_step")
        logger.run.define_metric("evaluation/*", step_metric="train_step")
        print(f"Diffusion Policy checkpoint directory: {run_dir}", flush=True)

        # 从起始 epoch 训练到目标总 epoch；续训不会重复已经完成的轮次。
        for epoch in range(start_epoch, args.num_epochs):
            started = time.perf_counter()

            # 完成一次训练集遍历，并取得该轮平均指标。
            state, metrics, rng, batch_count = train_epoch(
                state, policy, sampler, config, rng, logger, args.wandb_log_every,
            )
            completed = epoch + 1
            step = int(state.step)

            # 记录本轮耗时、batch 数量和训练指标。
            logger.log({"train_step": step, "epoch": {
                "completed": completed, "batches": batch_count,
                "seconds": time.perf_counter() - started,
                **{f"train_{key}": value for key, value in metrics.items()},
            }})
            print(f"epoch={completed} batches={batch_count} step={step} metrics={metrics}", flush=True)

            # 到达保存间隔或最后一轮时保存 checkpoint。
            final_epoch = completed == args.num_epochs
            if completed % args.checkpoint_every_epochs == 0 or final_epoch:
                path = save_training_checkpoint(
                    run_dir, state, epoch, config, normalizer, metrics,
                    keep=args.checkpoint_keep,
                    save_training_state=args.save_training_state == "true",
                )
                print(f"Saved checkpoint: {path}", flush=True)

            # 中途按间隔评估；最后一轮是否评估由 FINAL_EVAL 控制。
            if ((not final_epoch and completed % args.eval_every_epochs == 0)
                    or (final_epoch and args.final_eval == "true")):
                # 直接复用独立评估页的函数，避免维护两套 rollout 逻辑。
                from roboforg.workflows.diffusion_policy.eval_diffusion_policy import evaluate_policy
                results = evaluate_policy(
                    policy, state.ema_params, num_episodes=args.num_eval_episodes,
                    seed=args.seed + completed,
                )
                logger.log({"train_step": step, "evaluation": {"epoch": completed, **results}})
                print(f"Evaluation after epoch {completed}: {results}", flush=True)
    finally:
        # 正常结束或出现异常时都关闭 W&B，处理尚未上传的缓存日志。
        logger.finish()


if __name__ == "__main__":
    main()
