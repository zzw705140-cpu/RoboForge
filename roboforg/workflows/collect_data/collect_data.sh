#!/usr/bin/env bash
# Collect one batch of successful human Pick demonstrations.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

# Change only these values between collection runs.
# train: 20 runs × 10 demos = 200 demonstrations.
# eval:   3 runs × 10 demos = 30 demonstrations.
SPLIT="train"                                           #eval为评估所用的数据，直接改就行
SUCCESSES_NEEDED=10
INPUT_DEVICE="keyboard"
SEED=0

python -m roboforg.workflows.collect_data.collect_data \
    --split="${SPLIT}" \
    --successes-needed="${SUCCESSES_NEEDED}" \
    --input-device="${INPUT_DEVICE}" \
    --seed="${SEED}"
