#!/bin/bash
# Multi-step RL training — 50 steps with 24 seed tasks (Levels 1, 2, 3).
# Monitors compile rate, correctness, speedup over time.
#
# Usage:
#   bash scripts/kernel_rl/run_multi_step.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

NOW=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/tmp/kernel_rl_run_${NOW}"
LOG_FILE="${LOG_DIR}/train.log"
METRICS_FILE="${LOG_DIR}/step_metrics.jsonl"

mkdir -p "$LOG_DIR"

echo "============================================"
echo "Kernel RL Multi-Step — 50 steps, 24 tasks"
echo "Project : $PROJECT_ROOT"
echo "GPUs    : $(nvidia-smi -L 2>/dev/null | wc -l) available"
echo "Log dir : $LOG_DIR"
echo "============================================"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
export RAY_memory_monitor_refresh_ms=0
export RAY_DEDUP_LOGS=0
export HYDRA_OUTPUT_DIR=/tmp/hydra_outputs_${NOW}

echo ""
echo "Starting multi-step kernel PPO trainer..."
python -m verl.trainer.main_ppo_kernel \
    hydra.run.dir="${LOG_DIR}/hydra" \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=== Scorer logs ==="
echo ""

# Collect per-step stats from scorer logs
for f in /tmp/kernel_rl_checkpoints/*/events.out.tfevents.* 2>/dev/null; do
    if [ -f "$f" ]; then
        echo "TensorBoard events: $f"
    fi
done

# Summarize scorer logs
SCORER_DIR="./outputs/kernel_scorer_logs"
if [ -d "$SCORER_DIR" ]; then
    echo ""
    echo "=== Scorer results summary ==="
    for logfile in "$SCORER_DIR"/scorer_pid*.jsonl; do
        if [ -f "$logfile" ]; then
            total=$(wc -l < "$logfile")
            compiled=$(python3 -c "
import json
count = 0
total = 0
with open('$logfile') as f:
    for line in f:
        r = json.loads(line)
        total += 1
        if r.get('metrics',{}).get('compiled'): count += 1
print(f'{count}/{total}')
" 2>/dev/null)
            echo "  $logfile: $compiled compiled"
        fi
    done
fi

# Cleanup large temp files
rm -rf "${HYDRA_OUTPUT_DIR}" 2>/dev/null || true

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Multi-step RL PASSED (exit 0)."
    echo "Log: $LOG_FILE"
    echo "Scorer: $SCORER_DIR"
    echo "Tree: ./outputs/kernel_tree_logs"
else
    echo ""
    echo "Multi-step RL FAILED (exit $EXIT_CODE). Check log: $LOG_FILE"
fi

exit $EXIT_CODE
