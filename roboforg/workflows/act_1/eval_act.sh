#!/usr/bin/env bash
# 独立测评 ACT checkpoint：关屏统计成功率，开屏只观察动作。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

CHECKPOINT_PATH=""
SHOW_VIEWER=false                 # true：本地开屏观察；false：关屏统计成功率
SEED=0

# 开屏默认观察 10 回合；关屏默认统计 40 回合。
if [[ "$SHOW_VIEWER" == "true" ]]; then
  NUM_EPISODES=10
  unset MUJOCO_GL
elif [[ "$SHOW_VIEWER" == "false" ]]; then
  NUM_EPISODES=40
  export MUJOCO_GL=egl
else
  echo "SHOW_VIEWER must be true or false" >&2
  exit 2
fi
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
  --seed="$SEED" \
  --show-viewer="$SHOW_VIEWER"


# conda activate gym_hil
# cd ~/project/RoboForge
# bash roboforg/workflows/act_1/eval_act.sh

# python -m roboforg.workflows.act_1.eval_act \
#   --checkpoint="checkpoints/act/<运行目录>/epoch_0500.ckpt" \
#   --show-viewer=true \
#   --num-episodes=40 \
#   --print-policy-output true

