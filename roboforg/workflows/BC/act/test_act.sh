#!/usr/bin/env bash
# 手动仿真：默认离屏 40 回合；--show-viewer 开屏观察 10 回合。
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

RUN_INDEX=""
ROLLOUT_ARGS=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --run|--num-rollouts)
      if [[ "$#" -lt 2 || ! "$2" =~ ^[1-9][0-9]*$ ]]; then
        echo "$1 must be followed by a positive integer." >&2
        exit 2
      fi
      if [[ "$1" == "--run" ]]; then
        RUN_INDEX="$2"
      else
        ROLLOUT_ARGS+=(--num-rollouts "$2")
      fi
      shift 2
      ;;
    --show-viewer)
      ROLLOUT_ARGS+=(--show-viewer)
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done
if [[ -z "$RUN_INDEX" ]]; then
  echo "Usage: bash roboforg/workflows/act/test_act.sh --run N [--num-rollouts N] [--show-viewer]" >&2
  exit 2
fi

CHECKPOINT="$(python roboforg/workflows/act/checkpoint_paths.py --run "$RUN_INDEX")"
echo "ACT simulation: checkpoint=${CHECKPOINT}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python -m roboforg.workflows.act.run_act --checkpoint "$CHECKPOINT" "${ROLLOUT_ARGS[@]}"

# bash roboforg/workflows/act/test_act.sh --run 1
# bash roboforg/workflows/act/test_act.sh --run 1 --show-viewer
