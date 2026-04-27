#!/bin/bash
# End-to-end binary tree kernel RL data-flow validation.
# Depth=2: 1 root → 2 children → 4 leaves, 6 exportable training nodes.
#
# Usage:
#   bash scripts/kernel_rl/run_e2e.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

NOW=$(date +%Y%m%d_%H%M%S)
LOG_FILE="scripts/kernel_rl/e2e_run_${NOW}.log"

echo "============================================"
echo "Kernel RL E2E — Binary Tree Depth=2"
echo "Project : $PROJECT_ROOT"
echo "GPUs    : $(nvidia-smi -L 2>/dev/null | wc -l) available"
echo "Log     : $LOG_FILE"
echo "============================================"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
export RAY_memory_monitor_refresh_ms=0
export RAY_DEDUP_LOGS=0

echo ""
echo "Starting kernel PPO trainer (Ray will auto-init)..."
python -m verl.trainer.main_ppo_kernel \
    --config verl/trainer/config/ppo_trainer_kernel.yaml \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "E2E run PASSED (exit 0). Full log: $LOG_FILE"
else
    echo ""
    echo "E2E run FAILED (exit $EXIT_CODE). Check log: $LOG_FILE"
    exit $EXIT_CODE
fi
