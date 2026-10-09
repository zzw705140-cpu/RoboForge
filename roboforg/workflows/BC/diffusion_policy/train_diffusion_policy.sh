#!/usr/bin/env bash
# Diffusion Policy 训练入口：恢复路径留空时新建实验，填写完整 checkpoint 路径时继续原实验。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

# 数据、任务与初始化权重。
DATASET_ROOT="$PROJECT_ROOT/datasets"                              # 数据集根目录
TASK_NAME="pick"                                                   # 当前任务名称
TRAIN_DIR_NAME="train_1"                                           # datasets/pick 下的训练数据目录
EVAL_DIR_NAME="eval_1"                                             # datasets/pick 下的评估数据目录
PRETRAINED_PATH="$PROJECT_ROOT/.resnet_params/resnet10_params.pkl" # ResNet-10 预训练权重
SEED=0                                                             # 数据打乱、初始化和评估随机种子

# Diffusion Policy 模型与数据窗口。
BATCH_SIZE=32                                               # 每次参数更新使用的样本数
N_OBS_STEPS=2                                                       # 输入的连续观测帧数
HORIZON=16                                                  # 一次预测的动作序列长度
N_ACTION_STEPS=8                                            # 每次预测后连续执行的动作数
NUM_INFERENCE_STEPS=8                                               # DDIM 推理去噪次数
DOWN_CHANNELS="256,512,1024"                                        # 一维 U-Net 各层通道数
DIFFUSION_STEP_EMBEDDING_DIM=128                                    # 扩散步编码维度
KERNEL_SIZE=5                                                       # 一维卷积核大小，必须为奇数
N_GROUPS=8                                                          # GroupNorm 分组数
PREDICT_SCALE=true                                                  # 条件层是否同时预测缩放与偏移

# AdamW 与 EMA；评估和默认 checkpoint 使用 EMA 参数。
LEARNING_RATE=1e-4                                                  # AdamW 学习率
BETA1=0.95                                                          # AdamW 一阶动量系数
BETA2=0.999                                                         # AdamW 二阶动量系数
ADAM_EPSILON=1e-8                                                   # AdamW 数值稳定项
WEIGHT_DECAY=1e-6                                                   # 权重衰减强度
EMA_UPDATE_AFTER_STEP=0                                             # 延迟多少次更新后开始 EMA
EMA_INV_GAMMA=1.0                                                   # EMA warmup 速度参数
EMA_POWER=0.75                                                      # EMA 衰减曲线指数
EMA_MIN_DECAY=0.0                                                   # EMA 最小衰减率
EMA_MAX_DECAY=0.9999                                                # EMA 最大衰减率

# 训练节奏与仿真成功率评估。
NUM_EPOCHS=700                                              # 目标总训练轮数
EVAL_EVERY_EPOCHS=100                                       # 每隔多少轮进行一次仿真评估
NUM_EVAL_EPISODES=40                                                # 训练期间每次离屏评估的回合数
FINAL_EVAL=true                                            # 最后一轮是否进行评估

# checkpoint 与 W&B。
CHECKPOINT_EVERY_EPOCHS=200                                         # 每隔多少轮保存一次 checkpoint
CHECKPOINT_KEEP=2                                                     # 只保留最新的若干 checkpoint
SAVE_TRAINING_STATE=false                                           # false：只存 EMA；true：额外保存续训状态
RESUME_CHECKPOINT=""                                                # 续训 checkpoint；仅支持包含训练状态的文件
WANDB_PROJECT="roboforge"                                           # W&B 项目名称
WANDB_LOG_EVERY=200                                         # 每多少次参数更新记录一次 batch 指标

export XLA_PYTHON_CLIENT_PREALLOCATE=false

RESUME_ARGS=()
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  RESUME_ARGS+=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi

python -m roboforg.workflows.diffusion_policy.train_diffusion_policy "${RESUME_ARGS[@]}" \
  --dataset-root="$DATASET_ROOT" \
  --task-name="$TASK_NAME" \
  --train-dir-name="$TRAIN_DIR_NAME" \
  --eval-dir-name="$EVAL_DIR_NAME" \
  --pretrained-path="$PRETRAINED_PATH" \
  --seed="$SEED" \
  --batch-size="$BATCH_SIZE" \
  --n-obs-steps="$N_OBS_STEPS" \
  --horizon="$HORIZON" \
  --n-action-steps="$N_ACTION_STEPS" \
  --num-inference-steps="$NUM_INFERENCE_STEPS" \
  --down-channels="$DOWN_CHANNELS" \
  --diffusion-step-embedding-dim="$DIFFUSION_STEP_EMBEDDING_DIM" \
  --kernel-size="$KERNEL_SIZE" \
  --n-groups="$N_GROUPS" \
  --predict-scale="$PREDICT_SCALE" \
  --learning-rate="$LEARNING_RATE" \
  --beta1="$BETA1" \
  --beta2="$BETA2" \
  --adam-epsilon="$ADAM_EPSILON" \
  --weight-decay="$WEIGHT_DECAY" \
  --ema-update-after-step="$EMA_UPDATE_AFTER_STEP" \
  --ema-inv-gamma="$EMA_INV_GAMMA" \
  --ema-power="$EMA_POWER" \
  --ema-min-decay="$EMA_MIN_DECAY" \
  --ema-max-decay="$EMA_MAX_DECAY" \
  --num-epochs="$NUM_EPOCHS" \
  --eval-every-epochs="$EVAL_EVERY_EPOCHS" \
  --num-eval-episodes="$NUM_EVAL_EPISODES" \
  --final-eval="$FINAL_EVAL" \
  --checkpoint-every-epochs="$CHECKPOINT_EVERY_EPOCHS" \
  --checkpoint-keep="$CHECKPOINT_KEEP" \
  --save-training-state="$SAVE_TRAINING_STATE" \
  --wandb-project="$WANDB_PROJECT" \
  --wandb-log-every="$WANDB_LOG_EVERY"
