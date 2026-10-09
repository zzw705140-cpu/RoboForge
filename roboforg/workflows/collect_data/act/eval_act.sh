#!/usr/bin/env bash
# 启动 ACT 离线评估；第一个可选参数为 checkpoint 路径。

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

CHECKPOINT="${1:-checkpoints/act/best.ckpt}"
if [ "$#" -gt 0 ]; then
  shift
fi

if [ ! -f "$CHECKPOINT" ]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

# 其余参数原样传给 eval_act.py，例如 --max-batches=1。
python -m roboforg.workflows.act.eval_act --checkpoint="$CHECKPOINT" "$@"
