# 管理本地训练目录、配置文件和 checkpoint 路径。
# 模型状态由算法代码保存，W&B 由训练入口管理。

from __future__ import annotations

import datetime
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


_CONFIG_NAME = "run_config.json"
_RUN_TIMESTAMP = r"\d{8}_\d{6}"
_CHECKPOINT_PATTERNS = (
    re.compile(r"checkpoint_(\d+)$"),  # Sepo2 的命名方式
    re.compile(r"epoch_(\d+)\.ckpt$"),  # 现有 ACT 的命名方式
)

#------------------------------------------------------#
#               new / resume 启动时：内部辅助函数                
#------------------------------------------------------#

# 验证任务和算法名称能够安全地用于目录名。
def _name_component(value: str, field: str) -> str:
    if (not isinstance(value, str) or "__" in value or
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value)):
        raise ValueError(f"{field} must use letters, digits, '_' or '-', without '__': {value!r}")
    return value


# 生成任务和算法前缀，用于区分 root 下的运行目录。
def _run_prefix(task: str, algorithm: str) -> str:
    return f"{_name_component(task, 'task')}__{_name_component(algorithm, 'algorithm')}__"


#------------------------------------------------------#
#                       new 启动时                        
#------------------------------------------------------#

# new 启动时调用：在 root 下按本地时间创建运行目录，同秒重名时加序号。
# root 由训练入口传入，例如 RoboForge/checkpoints/act；已有目录不会被覆盖。
def create_run_dir(root: str | Path, task: str, algorithm: str) -> Path:
    root = Path(root).expanduser().resolve()
    prefix = _run_prefix(task, algorithm)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    suffix = 0
    while True:
        name = f"{prefix}{timestamp}" + (f"_{suffix:02d}" if suffix else "")
        run_dir = root / name
        try:
            run_dir.mkdir(exist_ok=False)
            return run_dir
        except FileExistsError:
            suffix += 1


#------------------------------------------------------#
#                      resume 启动时                      
#------------------------------------------------------#

# resume 启动时调用：按时间戳和同秒序号找最近的运行目录。
# 即使该目录没有 checkpoint 也返回它，由后续检查报错。
def find_latest_run_dir(root: str | Path, task: str, algorithm: str) -> Path:
    root = Path(root).expanduser().resolve()
    prefix = _run_prefix(task, algorithm)
    pattern = re.compile(rf"{re.escape(prefix)}({_RUN_TIMESTAMP})(?:_(\d+))?$")
    if not root.is_dir():
        raise FileNotFoundError(f"Training root does not exist: {root}")
    runs: list[tuple[str, int, Path]] = []
    for path in root.iterdir():
        match = pattern.fullmatch(path.name)
        if path.is_dir() and match:
            runs.append((match.group(1), int(match.group(2) or 0), path))
    if not runs:
        raise FileNotFoundError(f"No run found for task={task!r}, algorithm={algorithm!r} in {root}")
    return max(runs, key=lambda item: (item[0], item[1]))[2]


#------------------------------------------------------#
#                      找到要恢复的目录后                      
#------------------------------------------------------#

# 找到恢复目录后调用：按编号选择最新 checkpoint；没有则报错。
# 只检查目录的直接子项；支持 checkpoint_N 和 epoch_N.ckpt 两种命名。
# 两种命名同时出现时报错，避免混用 step 和 epoch 编号。
def find_latest_checkpoint(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    matches: list[list[tuple[int, Path]]] = [[], []]
    for path in run_dir.iterdir():
        for index, pattern in enumerate(_CHECKPOINT_PATTERNS):
            match = pattern.fullmatch(path.name)
            if match and (path.is_file() or path.is_dir()):
                matches[index].append((int(match.group(1)), path))
                break
    if all(matches):
        raise ValueError(f"Mixed checkpoint naming formats in: {run_dir}")
    checkpoints = matches[0] or matches[1]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in latest run: {run_dir}")
    return max(checkpoints, key=lambda item: item[0])[1]


#------------------------------------------------------#
#            保存配置或检查配置时：内部辅助函数                 
#------------------------------------------------------#

# 将常见配置值转成 JSON 数据；未知类型直接报错。
def _json_config(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_config(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_dict"):
        return _json_config(value.to_dict())
    if hasattr(value, "item"):
        return _json_config(value.item())
    raise TypeError(f"Unsupported configuration value: {type(value).__name__}")


#------------------------------------------------------#
#                   新训练建立目录后                      
#------------------------------------------------------#

# 新建运行目录后调用：将训练配置写入 run_config.json。
# 先写临时文件再原子替换；已有配置时报错，不覆盖原始配置。
def save_run_config(run_dir: str | Path, config: Mapping[str, Any]) -> Path:
    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    data = _json_config(config)
    target = run_dir / _CONFIG_NAME
    if target.exists():
        raise FileExistsError(f"Run config already exists: {target}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{_CONFIG_NAME}.", dir=run_dir)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


#------------------------------------------------------#
#                      恢复训练时                       
#------------------------------------------------------#

# resume 启动时调用：读取已有运行的训练配置。
def load_run_config(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir).expanduser().resolve() / _CONFIG_NAME
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Run config must contain a JSON object: {path}")
    return config


#------------------------------------------------------#
#                      恢复模型前                       
#------------------------------------------------------#

# 恢复模型前调用：检查调用方指定的不可变配置是否一致。
# 模型结构、动作长度等由算法列入 immutable_fields；训练轮数不列入。
def check_resume_config(
    saved: Mapping[str, Any],
    current: Mapping[str, Any],
    immutable_fields: Sequence[str],
) -> None:
    differences = []
    for field in immutable_fields:
        if field not in saved or field not in current:
            differences.append(f"{field}: missing from saved or current config")
        elif _json_config(saved[field]) != _json_config(current[field]):
            differences.append(f"{field}: saved={saved[field]!r}, current={current[field]!r}")
    if differences:
        raise ValueError("Resume configuration is incompatible: " + "; ".join(differences))


__all__ = [
    "create_run_dir",
    "find_latest_run_dir",
    "find_latest_checkpoint",
    "save_run_config",
    "load_run_config",
    "check_resume_config",
]
