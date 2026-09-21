"""ACT 的实验记录规则；通用 W&B 上传功能复用 utils.wandb。"""

from dataclasses import asdict
import json
from pathlib import Path
from uuid import uuid4


# 管理 ACT 的配置、指标和实验 ID；未启用时所有记录调用均为空操作。
class ACTTrainingLogger:
    # 保存通用 logger 和 batch 指标记录间隔。
    def __init__(self, logger=None, *, log_every: int = 10):
        if log_every <= 0:
            raise ValueError("log_every must be positive.")
        self.logger = logger
        self.log_every = log_every

    # 初始化或恢复实验；恢复信息随 checkpoint 目录保存，迁移服务器时需一并复制。
    @classmethod
    def create(cls, *, config, checkpoint_dir, resume_checkpoint=None,
               enabled=False, debug=False, project=None, entity=None,
               run_name=None, log_every=10, seed=0, train_count=0, eval_count=0):
        if not enabled:
            return cls(log_every=log_every)
        from roboforg.utils.wandb import make_wandb_logger

        directory = Path(checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        identity_path = directory / "wandb_run.json"
        previous = None
        # 禁用上传的测试不读取或写入线上实验 ID，避免污染正式恢复记录。
        if not debug and resume_checkpoint is not None:
            source = Path(resume_checkpoint).parent / "wandb_run.json"
            if source.exists():
                previous = json.loads(source.read_text())
                if project is not None and project != previous["project"]:
                    raise ValueError("W&B project differs from the resumed experiment.")
                if entity is not None and entity != previous["entity"]:
                    raise ValueError("W&B entity differs from the resumed experiment.")
            else:
                print("No saved W&B identity: creating a new experiment for resumed training.")
        if not debug and identity_path.exists() and previous is None:
            raise FileExistsError("This directory already has a W&B run; resume it or choose a new checkpoint directory.")
        if previous is not None and identity_path.exists():
            if json.loads(identity_path.read_text()) != previous:
                raise ValueError("Destination directory belongs to another W&B experiment.")

        # Path 转为字符串；只上传明确列出的训练配置，不读取或记录登录密钥。
        variant = {key: str(value) if isinstance(value, Path) else value
                   for key, value in asdict(config).items()}
        variant.update(seed=seed, train_trajectories=train_count, eval_trajectories=eval_count)
        logger = make_wandb_logger(
            project=previous["project"] if previous else (project or "roboforge"),
            entity=previous["entity"] if previous else entity,
            description="act", variant=variant, debug=debug,
            wandb_output_dir=directory, run_name=run_name,
            run_id=previous["id"] if previous else uuid4().hex,
            resume="must" if previous else None,
        )
        try:
            # 用显式训练步数作横轴；W&B 内部提交步数自增，避免同一步的 batch/epoch 日志冲突。
            logger.run.define_metric("train_step")
            for pattern in ("train/*", "epoch_train/*", "eval/*", "progress/*"):
                logger.run.define_metric(pattern, step_metric="train_step")
            if not debug:
                identity = {"id": logger.run.id, "project": logger.run.project,
                            "entity": logger.run.entity}
                temporary = identity_path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(identity, indent=2))
                temporary.replace(identity_path)
        except BaseException:
            logger.finish()
            raise
        return cls(logger, log_every=log_every)

    # 每隔指定更新次数上传一批损失；转换标量也会等待 JAX 异步计算完成。
    def log_train_step(self, metrics, *, step):
        if self.logger is not None and step % self.log_every == 0:
            self.logger.log({"train_step": step,
                             "train": {key: float(value) for key, value in metrics.items()}})

    # 记录整轮平均指标与耗时；epoch 对用户显示为从 1 开始的已完成轮数。
    def log_epoch(self, summary, *, step, epoch, seconds, train_batches, eval_batches):
        if self.logger is None:
            return
        data = {"train_step": step, "epoch_train": {}, "eval": {},
                "progress": {"epoch": epoch + 1, "seconds": seconds,
                             "train_batches": train_batches, "eval_batches": eval_batches}}
        for key, value in summary.items():
            if key.startswith("train_"):
                data["epoch_train"][key[6:]] = float(value)
            elif key.startswith("eval_"):
                data["eval"][key[5:]] = float(value)
            elif key == "best_eval_loss":
                data["progress"][key] = float(value)
        self.logger.log(data)

    # 完成上传并释放 run；重复调用不会重复关闭。
    def finish(self):
        if self.logger is not None:
            self.logger.finish()
            self.logger = None
