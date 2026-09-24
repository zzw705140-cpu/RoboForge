#!/usr/bin/env bash
# ACT 新训练入口：修改下面配置后运行本脚本；每次运行自动创建时间目录。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

# 数据与 ACT 参数：沿用旧 ACT 训练脚本的常用值。
DATASET_ROOT="$PROJECT_ROOT/datasets"           # 训练数据集的根目录
TASK_NAME="pick"                                # 任务名称
TRAIN_DIR_NAME="train_1"                        # datasets/pick 下用于训练的文件夹；   旧数据用 train
EVAL_DIR_NAME="eval_1"                          # datasets/pick 下用于离线评估的文件夹；旧数据用 eval
SEED=0                                          # 随机种子，控制数据打乱和模型初始化
BATCH_SIZE=32                                   # 每次参数更新使用的样本数
ACTION_HORIZON=4                                # 每次预测的连续动作步数
LEARNING_RATE=1e-4                              # 非视觉网络参数的学习率
BACKBONE_LEARNING_RATE=1e-5                     # 视觉主干网络参数的学习率
KL_WEIGHT=10.0                                  # 总损失中 KL 损失的权重

# 训练节奏与离屏成功率评估。
NUM_EPOCHS=700                                    # 训练轮数
EVAL_EVERY_EPOCHS=200                           # 测评间隔
NUM_EVAL_EPISODES=40                            # 一次测评进行40回合测试
FINAL_EVAL=true                           # 冒烟测试跳过最后测评；正式训练改为 true

# checkpoint 和 W&B；W&B 使用服务器上 wandb login 保存的身份。
CHECKPOINT_EVERY_EPOCHS=200                     # 权重参数保存间隔
CHECKPOINT_KEEP=1                               # 只保留最新的一个权重参数
WANDB_PROJECT="roboforge"
WANDB_LOG_EVERY=10                              # 每10次更新记录一次训练指标

export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -m roboforg.workflows.act_1.train_act \
  --dataset-root="$DATASET_ROOT" \
  --task-name="$TASK_NAME" \
  --train-dir-name="$TRAIN_DIR_NAME" \
  --eval-dir-name="$EVAL_DIR_NAME" \
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
