#!/usr/bin/env bash
# 独立评估 Diffusion Policy：路径留空时自动使用最新 checkpoint。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

CHECKPOINT_PATH=""                              # 留空：最新权重；填写完整路径：指定权重
SHOW_VIEWER=false                              # true：打开仿真窗口；false：离屏评估
NUM_EPISODES=40                                # 开屏建议 10，离屏建议 40
PRINT_POLICY_OUTPUT=false                      # 是否打印每一步实际发送给环境的动作
SEED=0                                         # 环境和扩散采样随机种子

export XLA_PYTHON_CLIENT_PREALLOCATE=false

CHECKPOINT_ARGS=()
if [[ -n "$CHECKPOINT_PATH" ]]; then
  CHECKPOINT_ARGS+=(--checkpoint "$CHECKPOINT_PATH")
fi

python -m roboforg.workflows.diffusion_policy.eval_diffusion_policy "${CHECKPOINT_ARGS[@]}" \
  --show-viewer="$SHOW_VIEWER" \
  --num-episodes="$NUM_EPISODES" \
  --print-policy-output="$PRINT_POLICY_OUTPUT" \
  --seed="$SEED"
