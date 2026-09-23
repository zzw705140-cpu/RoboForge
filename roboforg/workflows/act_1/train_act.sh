#!/usr/bin/env bash
# ACT 新训练入口：修改下面配置后运行本脚本；每次运行自动创建时间目录。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

# 数据与 ACT 参数：沿用旧 ACT 训练脚本的常用值。
DATASET_ROOT="$PROJECT_ROOT/datasets"
TASK_NAME="pick"
SEED=0
BATCH_SIZE=32
ACTION_HORIZON=4
LEARNING_RATE=1e-4
BACKBONE_LEARNING_RATE=1e-5
KL_WEIGHT=10.0

# 训练节奏与离屏成功率评估。
NUM_EPOCHS=1000
EVAL_EVERY_EPOCHS=500
NUM_EVAL_EPISODES=40
FINAL_EVAL=true                      # 冒烟测试跳过最后测评；正式训练改为 true

# checkpoint 和 W&B；W&B 使用服务器上 wandb login 保存的身份。
CHECKPOINT_EVERY_EPOCHS=200
CHECKPOINT_KEEP=5
WANDB_PROJECT="roboforge"
WANDB_LOG_EVERY=10

export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -m roboforg.workflows.act_1.train_act \
  --dataset-root="$DATASET_ROOT" \
  --task-name="$TASK_NAME" \
  --seed="$SEED" \
  --batch-size="$BATCH_SIZE" \
  --action-horizon="$ACTION_HORIZON" \
  --learning-rate="$LEARNING_RATE" \
  --backbone-learning-rate="$BACKBONE_LEARNING_RATE" \
  --kl-weight="$KL_WEIGHT" \
  --num-epochs="$NUM_EPOCHS" \
  --eval-every-epochs="$EVAL_EVERY_EPOCHS" \
  --final-eval="$FINAL_EVAL" \
  --num-eval-episodes="$NUM_EVAL_EPISODES" \
  --checkpoint-every-epochs="$CHECKPOINT_EVERY_EPOCHS" \
  --checkpoint-keep="$CHECKPOINT_KEEP" \
  --wandb-project="$WANDB_PROJECT" \
  --wandb-log-every="$WANDB_LOG_EVERY"


# conda activate gym_hil
# cd ~/project/RoboForge
# bash roboforg/workflows/act_1/train_act.sh
