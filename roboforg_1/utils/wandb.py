from __future__ import annotations

import datetime
import tempfile
from collections.abc import Mapping
from copy import copy
from pathlib import Path
from socket import gethostname
from typing import Any

import absl.flags as flags
import ml_collections
import wandb

###################################################
#                日志/实验管理工具                  #
###################################################
#工具层代码，对接Weights & Biases（wandb）
#wandb是训练实验记录平台，用来记录各种曲线和配置，以及实验对比


def _recursive_flatten_dict(data: Mapping[str, Any]) -> dict[str, Any]:
    """将嵌套指标字典展平为 WandB 使用的 ``a/b/c`` 键名"""

    flattened: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, Mapping):
            nested = _recursive_flatten_dict(value)
            flattened.update({f"{key}/{nested_key}": nested_value for nested_key, nested_value in nested.items()})
        else:
            flattened[str(key)] = value
    return flattened


class WandBLogger:
    """封装 WandB run 的初始化、配置同步和指标记录"""

    @staticmethod
    def get_default_config() -> ml_collections.ConfigDict:
        """返回默认 WandB 配置"""

        config = ml_collections.ConfigDict()
        config.project = "sepo"
        config.entity = ml_collections.config_dict.FieldReference(None, field_type=str)
        config.exp_descriptor = ""
        config.unique_identifier = ""
        config.group = None
        config.tag = None
        return config

    def __init__(
        self,
        wandb_config: ml_collections.ConfigDict,
        variant: Mapping[str, Any],
        wandb_output_dir: str | Path | None = None,
        debug: bool = False,
        run_name: str | None = None,
        run_id: str | None = None,
        resume: str | None = None,
    ) -> None:
        self.config = wandb_config

        # 未指定唯一标识时使用当前时间，避免不同训练任务使用相同 run id。
        if self.config.unique_identifier == "":
            self.config.unique_identifier = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        self.experiment_id = run_id or f"{self.config.exp_descriptor}_{self.config.unique_identifier}"
        self.config.experiment_id = self.experiment_id

        if wandb_output_dir is None:
            wandb_output_dir = tempfile.mkdtemp(prefix="sepo_wandb_")
        else:
            output_path = Path(wandb_output_dir).expanduser()
            output_path.mkdir(parents=True, exist_ok=True)
            wandb_output_dir = str(output_path)

        self._variant = copy(dict(variant))
        self._variant.setdefault("hostname", gethostname())

        # debug=True 时禁用网络上传，但保留与在线模式相同的调用路径。
        mode = "disabled" if debug else "online"
        tags = None if self.config.tag is None else [str(self.config.tag)]

        self.run = wandb.init(
            config=self._variant,
            project=self.config.project,
            entity=self.config.entity,
            group=self.config.group,
            tags=tags,
            dir=wandb_output_dir,
            id=self.experiment_id,
            name=self.experiment_id if run_name is None else run_name,  # run_name 只控制 WandB 页面上的显示名；带时间戳的 experiment_id 仍用作唯一 id，避免多次评估相互覆盖
            save_code=True,
            mode=mode,
            resume=resume,
        )

        # absl flags 已完成解析时，把训练命令行参数同步保存到 WandB config
        if flags.FLAGS.is_parsed():
            flag_dict = {key: getattr(flags.FLAGS, key) for key in flags.FLAGS}
        else:
            flag_dict = {}

        for key, value in flag_dict.items():
            if isinstance(value, ml_collections.ConfigDict):
                flag_dict[key] = value.to_dict()

        if flag_dict:
            self.run.config.update(flag_dict, allow_val_change=True)

    def log(self, data: Mapping[str, Any], step: int | None = None) -> None:
        """记录指标；嵌套字典会自动转换为斜杠分隔的扁平键"""

        self.run.log(_recursive_flatten_dict(data), step=step)

    def finish(self) -> None:
        """等待剩余指标写入完成，并正常结束当前 WandB run"""

        self.run.finish()


def make_wandb_logger(
    project: str = "gym-hil",
    description: str = "offline_iql",
    *,
    debug: bool = False,
    entity: str | None = None,
    group: str | None = None,
    variant: Mapping[str, Any] | None = None,
    wandb_output_dir: str | Path | None = None,
    run_name: str | None = None,
    run_id: str | None = None,
    resume: str | None = None,
) -> WandBLogger:
    """创建 WandBLogger"""

    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": project,
            "entity": entity,
            "group": group,
            "exp_descriptor": description,
            "tag": description,
        }
    )

    return WandBLogger(
        wandb_config=wandb_config,
        variant={} if variant is None else variant,
        wandb_output_dir=wandb_output_dir,
        debug=debug,
        run_name=run_name,
        run_id=run_id,
        resume=resume,
    )


__all__ = ["WandBLogger", "make_wandb_logger"]
