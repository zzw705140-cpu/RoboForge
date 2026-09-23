#!/usr/bin/env bash
# 独立评估 ACT checkpoint：可在下方指定某个 epoch 文件；留空则选择最新运行的最新文件。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

CHECKPOINT_PATH=""
NUM_EPISODES=40
SEED=0

export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false

if [[ -z "$CHECKPOINT_PATH" ]]; then
  CHECKPOINT_PATH="$(python - <<'PYTHON'
from pathlib import Path
from roboforg.utils.checkpoint_untils import find_latest_checkpoint, find_latest_run_dir
root = Path.cwd() / "checkpoints" / "act"
print(find_latest_checkpoint(find_latest_run_dir(root, "pick", "act")))
PYTHON
)"
fi

python -m roboforg.workflows.act_1.eval_act \
  --checkpoint="$CHECKPOINT_PATH" \
  --num-episodes="$NUM_EPISODES" \
  --seed="$SEED"


# conda activate gym_hil
# cd ~/project/RoboForge
# bash roboforg/workflows/act_1/eval_act.sh

# python -m roboforg.workflows.act_1.eval_act \
#   --checkpoint="checkpoints/act/<运行目录>/epoch_0500.ckpt" \
#   --num-episodes=40