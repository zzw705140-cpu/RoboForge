#!/usr/bin/env bash
# FQL 独立评估入口, 所有关键选项允许由命令行覆盖.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$PROJECT_ROOT"

# ------------------------- 模型来源 ------------------------- #
CHECKPOINT_PATH=""                 # 指定 checkpoint, 留空时按实验目录或最新实验查找
RUN_DIR=""                         # 指定实验目录, 与 CHECKPOINT_PATH 二选一
CHECKPOINT_ROOT="checkpoints/fql"
TASK_NAME=""                       # 自动查找默认 pick, 指定其他任务时填写映射名称

# ------------------------- 评估选项 ------------------------- #
NUM_EPISODES=40
SHOW_VIEWER=false
PRINT_POLICY_OUTPUT=false
SEED=0

SOURCE_ARGS=()
if [[ -n "$CHECKPOINT_PATH" ]]; then
    SOURCE_ARGS+=(--checkpoint "$CHECKPOINT_PATH")
fi
if [[ -n "$RUN_DIR" ]]; then
    SOURCE_ARGS+=(--run-dir "$RUN_DIR")
fi
if [[ -n "$TASK_NAME" ]]; then
    SOURCE_ARGS+=(--task-name "$TASK_NAME")
fi

exec python -m 'roboforg_1.workflows.BC.bc+fql.eval_fql' \
    "${SOURCE_ARGS[@]}" \
    --checkpoint-root="$CHECKPOINT_ROOT" \
    --num-episodes="$NUM_EPISODES" \
    --show-viewer="$SHOW_VIEWER" \
    --print-policy-output="$PRINT_POLICY_OUTPUT" \
    --seed="$SEED" "$@"
