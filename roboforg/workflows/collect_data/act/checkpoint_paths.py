"""轻量 checkpoint 路径选择；可由 shell 直接调用，无需初始化 JAX。"""

import argparse
from pathlib import Path
import re


def numbered_checkpoints(directory: Path) -> list[Path]:
    paths = [p for p in directory.glob("epoch_*.ckpt")
             if p.is_file() and re.fullmatch(r"epoch_[0-9]+\.ckpt", p.name)]
    return sorted(paths, key=lambda p: int(p.stem.split("_")[1]))


def latest_checkpoint(directory: Path) -> Path:
    paths = numbered_checkpoints(directory)
    if paths:
        return paths[-1]
    # 兼容本次改动前保存的实验；不删除其历史文件。
    legacy = directory / "latest.ckpt"
    if legacy.is_file():
        return legacy
    raise FileNotFoundError(f"No checkpoint found in {directory}")


def resolve_run(root: Path, run: int) -> Path:
    directories = sorted(p for p in root.glob(f"act_k*_b*_run{run}") if p.is_dir())
    if len(directories) != 1:
        raise ValueError(f"Expected one directory for run{run}, found {len(directories)}: "
                         + ", ".join(map(str, directories)))
    return latest_checkpoint(directories[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", type=int)
    group.add_argument("--directory", type=Path)
    parser.add_argument("--root", type=Path, default=Path("checkpoints/act"))
    args = parser.parse_args()
    if args.run is not None and args.run <= 0:
        parser.error("--run must be positive")
    try:
        path = resolve_run(args.root, args.run) if args.run is not None else latest_checkpoint(args.directory)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(path)


if __name__ == "__main__":
    main()
