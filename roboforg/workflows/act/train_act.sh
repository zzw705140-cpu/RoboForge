#!/usr/bin/env bash
# ACT 统一训练入口：new 新建实验，resume 从该实验的 latest.ckpt 继续训练。

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

##############################################################################
# 训练配置：需要调整参数时，只修改这一段。
##############################################################################
NUM_EPOCHS=4                 # 本次训练轮数；resume 时表示额外增加的轮数
INITIAL_EPOCHS=4             # 实验名称中的 ep 标签，创建实验后不要修改
BATCH_SIZE=32                # 每个 batch 的样本数
ACTION_HORIZON=4             # 每次预测的连续动作步数
LEARNING_RATE=1e-4           # Transformer、CVAE 等主网络学习率
BACKBONE_LEARNING_RATE=1e-5  # 预训练 ResNet-10 的微调学习率
KL_WEIGHT=10.0               # CVAE KL loss 权重
SEED=0                       # 随机种子

# W&B 配置：训练时自动上传指标；API key 由服务器上的 wandb login 管理。
WANDB_PROJECT="roboforge"
WANDB_ENTITY="zzw705140-"
WANDB_LOG_EVERY=10

# 关闭 JAX 的大块显存预留，按实际需要申请显存。
export XLA_PYTHON_CLIENT_PREALLOCATE=false

##############################################################################
# 启动参数：第一个参数为 new/resume，第二个参数为正整数 run 编号。
##############################################################################
MODE="${1:-}"
RUN_INDEX="${2:-}"
if [[ "$MODE" != "new" && "$MODE" != "resume" ]]; then
  echo "Usage: bash roboforg/workflows/act/train_act.sh {new|resume} RUN_INDEX" >&2
  exit 2
fi
if [[ ! "$RUN_INDEX" =~ ^[1-9][0-9]*$ ]]; then
  echo "RUN_INDEX must be a positive integer, for example: new 1 or resume 2." >&2
  exit 2
fi

# W&B 名称包含关键参数；checkpoint 目录不含 ep，保证追加轮数后路径保持不变。
RUN_NAME="act_k${ACTION_HORIZON}_b${BATCH_SIZE}_ep${INITIAL_EPOCHS}_run${RUN_INDEX}"
CHECKPOINT_DIR="checkpoints/act/act_k${ACTION_HORIZON}_b${BATCH_SIZE}_run${RUN_INDEX}"
LATEST_CHECKPOINT="${CHECKPOINT_DIR}/latest.ckpt"
RESUME_ARGS=()

if [[ "$MODE" == "new" ]]; then
  # 新实验禁止复用已有目录，避免覆盖旧模型或混入旧的 W&B Run。
  if [[ -e "$CHECKPOINT_DIR" ]]; then
    echo "New run refused: ${CHECKPOINT_DIR} already exists. Use resume or choose another RUN_INDEX." >&2
    exit 1
  fi
else
  # 恢复实验必须存在完整的 latest.ckpt；它包含权重、优化器和已完成 epoch。
  if [[ ! -f "$LATEST_CHECKPOINT" ]]; then
    echo "Resume failed: checkpoint not found: ${LATEST_CHECKPOINT}" >&2
    exit 1
  fi
  RESUME_ARGS+=(--resume="$LATEST_CHECKPOINT")
fi

echo "ACT ${MODE}: name=${RUN_NAME}, epochs=${NUM_EPOCHS}, batch=${BATCH_SIZE}, chunk=${ACTION_HORIZON}"

python -m roboforg.workflows.act.train_act \
  --dataset-root=datasets \
  --checkpoint-dir="$CHECKPOINT_DIR" \
  --num-epochs="$NUM_EPOCHS" \
  --batch-size="$BATCH_SIZE" \
  --action-horizon="$ACTION_HORIZON" \
  --learning-rate="$LEARNING_RATE" \
  --backbone-learning-rate="$BACKBONE_LEARNING_RATE" \
  --kl-weight="$KL_WEIGHT" \
  --seed="$SEED" \
  --wandb \
  --wandb-project="$WANDB_PROJECT" \
  --wandb-entity="$WANDB_ENTITY" \
  --wandb-name="$RUN_NAME" \
  --wandb-log-every="$WANDB_LOG_EVERY" \
  "${RESUME_ARGS[@]}"


# conda activate gym_hil
# cd ~/project/RoboForge
# bash roboforg/workflows/act/train_act.sh new 1
# bash roboforg/workflows/act/train_act.sh resume 1
