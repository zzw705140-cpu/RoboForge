#!/usr/bin/env bash
# FQL 离线联合训练, 命令末尾可追加参数覆盖以下设置.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$PROJECT_ROOT"

# ------------------------- 任务与数据 ------------------------- #
TASK_NAME=pick                             # 任务名称, 对应 examples/mappings.py 中的环境映射
DATASET_ROOT=datasets                      # 数据集根目录
TRAIN_DIR_NAME=train_1                       # 训练数据子目录, 路径为 数据集根目录/任务/子目录
EVAL_DIR_NAME=eval_1                         # 保留的评估数据目录配置, 当前周期评估使用环境 rollout
CAMERA_KEYS=front,wrist                    # 输入相机名称及顺序, 多路相机用逗号分隔
BATCH_SIZE=32                                     # 每次参数更新采样的完整动作块数量
OBSERVATION_HORIZON=1                      # 观测历史长度 T, 包含当前帧
ACTION_HORIZON=4                           # 动作块长度 H, 评估时每次连续执行 H 步
USE_TEMPORAL_HISTORY=true                  # 是否保留观测历史维, false 时 T 必须为 1

# ------------------------- Encoder 与 Behavior Flow ------------------------- #
ENCODER_LR=0.0003                          # 观测编码器学习率, ResNet 主干保持冻结
NETWORK_TYPE=condition                     # 条件主干类型, 当前仅支持 condition
HIDDEN_DIM=512                             # Behavior Flow 和 Actor 的隐藏层宽度
NUM_BLOCKS=4                               # Behavior Flow 和 Actor 的隐藏层数量
TIME_EMBED_DIM=32                          # Behavior Flow 时间编码维度, 至少为 4 的偶数
DROPOUT_RATE=0.0                           # Behavior Flow 和 Actor 的 Dropout 比例, 0 表示关闭
NUM_ODE_STEPS=10                           # Behavior Flow 教师生成蒸馏目标时的 ODE 积分步数
BEHAVIOR_FLOW_LR=0.0003                    # Behavior Flow 速度网络学习率

# ------------------------- Actor 与 Critic ------------------------- #
ONE_STEP_ACTOR_LR=0.0003                   # One-step Actor 学习率
CRITIC_HIDDEN_DIMS=512,512,512,512         # 各 Q head 的隐藏层宽度, 用逗号分隔
NUM_QS=2                                   # 参数独立的 Q head 数量
CRITIC_LR=0.0003                           # 在线 Critic 学习率
DISCOUNT=0.99                              # 单个环境步的折扣因子, chunk bootstrap 使用其 H 次方
TAU=0.005                                  # Target Critic 软更新系数, 在线参数占比
ALPHA=100.0                                # Actor 蒸馏损失权重, 总损失为 alpha * 蒸馏损失 - Q
Q_AGGREGATION=mean                         # Q head 聚合方式, mean 取均值或 min 取最小值

# ------------------------- 训练与 checkpoint ------------------------- #
NUM_EPOCHS=700                                      # 目标总轮数, 包含续训前已完成轮数
STEPS_PER_EPOCH=""                         # 每轮随机更新次数, 留空自动计算或沿用续训记录
SEED=""                                    # 随机种子, 留空时新训练默认 0, 续训使用保存值
DEVICE=auto                                # 计算设备, auto 自动选择, 也可指定 cpu 或 gpu
CHECKPOINT_ROOT=checkpoints/fql            # 新训练的 checkpoint 根目录, 续训沿用原实验目录
RESUME_CHECKPOINT=""                       # 续训 checkpoint 路径, 留空创建新实验
CHECKPOINT_EVERY_EPOCHS=100                # 每隔多少轮保存完整训练状态, 最后一轮也会保存
CHECKPOINT_KEEP=2                          # 仅保留最近多少个 checkpoint, 保存成功后删除旧文件

# ------------------------- 统一评估 ------------------------- #
EVAL_EVERY_EPOCHS=100                      # 训练期间每隔多少轮评估, 0 关闭周期评估
NUM_EVAL_EPISODES=40                       # 每次训练期间评估的回合数
FINAL_EVAL=true                                    # 最后一轮是否评估, 独立于周期评估间隔
SHOW_VIEWER=false                          # 训练期间评估是否打开仿真窗口
PRINT_POLICY_OUTPUT=false                  # 训练期间评估是否打印每步发送给环境的动作

# ------------------------- W&B ------------------------- #
WANDB=true                                 # 是否启用 W&B 指标记录和上传
WANDB_DEBUG=false                          # true 使用禁用上传的调试模式
WANDB_PROJECT=roboforge                    # W&B 项目名称
WANDB_LOG_EVERY=200                                  # 每隔多少次参数更新记录一次 batch 训练指标
UPLOAD_CHECKPOINT_ARTIFACT=false           # 是否额外上传模型 Artifact, 默认关闭

OPTIONAL_ARGS=()
if [[ -n "$RESUME_CHECKPOINT" ]]; then
    OPTIONAL_ARGS+=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi
if [[ -n "$STEPS_PER_EPOCH" ]]; then
    OPTIONAL_ARGS+=(--steps-per-epoch "$STEPS_PER_EPOCH")
fi
if [[ -n "$SEED" ]]; then
    OPTIONAL_ARGS+=(--seed "$SEED")
fi

# 续训时模型和数据配置默认从 checkpoint 读取, 不传入脚本的新训练默认值.
MODEL_ARGS=()
RESUMING=false
if [[ -n "$RESUME_CHECKPOINT" ]]; then
    RESUMING=true
fi
for ARGUMENT in "$@"; do
    if [[ "$ARGUMENT" == --resume-checkpoint || "$ARGUMENT" == --resume-checkpoint=* ]]; then
        RESUMING=true
    fi
done
if [[ "$RESUMING" == false ]]; then
    MODEL_ARGS+=(
        --task-name="$TASK_NAME" --dataset-root="$DATASET_ROOT"
        --train-dir-name="$TRAIN_DIR_NAME" --eval-dir-name="$EVAL_DIR_NAME"
        --camera-keys="$CAMERA_KEYS" --batch-size="$BATCH_SIZE"
        --observation-horizon="$OBSERVATION_HORIZON" --action-horizon="$ACTION_HORIZON"
        --use-temporal-history="$USE_TEMPORAL_HISTORY"
        --encoder-lr="$ENCODER_LR" --network-type="$NETWORK_TYPE"
        --hidden-dim="$HIDDEN_DIM" --num-blocks="$NUM_BLOCKS"
        --time-embed-dim="$TIME_EMBED_DIM" --dropout-rate="$DROPOUT_RATE"
        --num-ode-steps="$NUM_ODE_STEPS" --behavior-flow-lr="$BEHAVIOR_FLOW_LR"
        --one-step-actor-lr="$ONE_STEP_ACTOR_LR" --critic-hidden-dims="$CRITIC_HIDDEN_DIMS"
        --num-qs="$NUM_QS" --critic-lr="$CRITIC_LR"
        --discount="$DISCOUNT" --tau="$TAU" --alpha="$ALPHA" --q-aggregation="$Q_AGGREGATION"
    )
fi
exec python -m 'roboforg_1.workflows.BC.bc+fql.train_fql' \
    "${MODEL_ARGS[@]}" "${OPTIONAL_ARGS[@]}" \
    --num-epochs="$NUM_EPOCHS" --device="$DEVICE" \
    --checkpoint-root="$CHECKPOINT_ROOT" \
    --checkpoint-every-epochs="$CHECKPOINT_EVERY_EPOCHS" --checkpoint-keep="$CHECKPOINT_KEEP" \
    --eval-every-epochs="$EVAL_EVERY_EPOCHS" --num-eval-episodes="$NUM_EVAL_EPISODES" \
    --final-eval="$FINAL_EVAL" --show-viewer="$SHOW_VIEWER" --print-policy-output="$PRINT_POLICY_OUTPUT" \
    --wandb="$WANDB" --wandb-debug="$WANDB_DEBUG" --wandb-project="$WANDB_PROJECT" \
    --wandb-log-every="$WANDB_LOG_EVERY" --upload-checkpoint-artifact="$UPLOAD_CHECKPOINT_ARTIFACT" "$@"
