#!/usr/bin/env bash
# Collect one batch of successful human Pick demonstrations.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

# Change only these values between collection runs.
# train: 20 runs × 10 demos = 200 demonstrations.
# eval:   3 runs × 10 demos = 30 demonstrations.
SPLIT="train"                                           # 数据用途：train 或 eval
OUTPUT_DIR_NAME="train_1"                               # datasets/pick 下的保存文件夹；采 eval 时改为 eval_1
SUCCESSES_NEEDED=10
INPUT_DEVICE="keyboard"
SEED=0
# 接管后首次有效机械臂动作的六维范数阈值；夹爪目标切换也会开始录制。
START_MOTION_THRESHOLD=0.000001

python -m roboforg.workflows.collect_data.collect_data \
    --split="${SPLIT}" \
    --output-dir-name="${OUTPUT_DIR_NAME}" \
    --successes-needed="${SUCCESSES_NEEDED}" \
    --input-device="${INPUT_DEVICE}" \
    --seed="${SEED}" \
    --start-motion-threshold="${START_MOTION_THRESHOLD}"


# conda activate gym_hil
# cd ~/project/RoboForge
# bash roboforg/workflows/collect_data/collect_data.sh