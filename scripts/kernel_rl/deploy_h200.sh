#!/bin/bash
# ============================================================================
# One-click H200 deployment for multi-step Tree-GRPO kernel RL training.
#
# Usage (on the H200 machine):
#   export MODEL_PATH=/path/to/Qwen2.5-Coder-7B-Instruct-SFT-kernel-v3
#   export RUN_DIR=/path/to/output_dir
#   bash scripts/kernel_rl/deploy_h200.sh
#
# Output structure (all under $RUN_DIR):
#   train.log          — full training stdout/stderr
#   progress.json      — step-level metrics, scorer stats, updated each step
#   tree_logs/         — per-node full prompt + response + scorer feedback
#   scorer_logs/       — per-node JSONL scoring results (compile/speedup/rewards)
#   samples/           — model-generated code extracted every N steps
#   checkpoints/       — FSDP model checkpoints (saved every 10 steps)
#   deploy.log         — deployment script output
# ============================================================================

set -euo pipefail

# ── User-configurable paths ───────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-/inspire/qb-ilm/project/wuliqifa/public/sdt/models/Qwen2.5-Coder-7B-Instruct-SFT-kernel-v3}"
RUN_DIR="${RUN_DIR:-$(pwd)/outputs/h200_run_$(date +%Y%m%d_%H%M%S)}"
TOTAL_STEPS="${TOTAL_STEPS:-50}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-10}"

# ── Fixed paths ───────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Create output structure ───────────────────────────────────────────
mkdir -p "$RUN_DIR"/{tree_logs,scorer_logs,samples,checkpoints}

DEPLOY_LOG="$RUN_DIR/deploy.log"
TRAIN_LOG="$RUN_DIR/train.log"
PROGRESS_FILE="$RUN_DIR/progress.json"

# ── Log header ────────────────────────────────────────────────────────
exec > >(tee -a "$DEPLOY_LOG") 2>&1

echo "=============================================================================="
echo "Tree-GRPO Kernel RL — H200 Deployment"
echo "=============================================================================="
echo "Started:     $(date)"
echo "Model:       $MODEL_PATH"
echo "Run dir:     $RUN_DIR"
echo "Steps:       $TOTAL_STEPS"
echo "GPUs:        $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -8
echo "=============================================================================="
echo ""
echo "Output structure:"
echo "  Training log:    $TRAIN_LOG"
echo "  Progress JSON:   $PROGRESS_FILE"
echo "  Tree logs:       $RUN_DIR/tree_logs/"
echo "  Scorer logs:     $RUN_DIR/scorer_logs/"
echo "  Code samples:    $RUN_DIR/samples/"
echo "  Checkpoints:     $RUN_DIR/checkpoints/"
echo "=============================================================================="
echo ""

# ── Verify prerequisites ──────────────────────────────────────────────
echo "=== Checking prerequisites ==="

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Set MODEL_PATH=/path/to/Qwen2.5-Coder-7B-Instruct-SFT-kernel-v3"
    exit 1
fi
echo "  [OK] Model: $MODEL_PATH"

if [ ! -f "$PROJECT_ROOT/data/kernel_rl/train.parquet" ]; then
    echo "ERROR: Training data not found. Run: python scripts/kernel_rl/build_seed_tasks.py"
    exit 1
fi
echo "  [OK] Training data: 24 tasks"

python3 -c "import torch; print(f'  [OK] PyTorch {torch.__version__} + CUDA {torch.version.cuda}')" 2>/dev/null
python3 -c "import triton; print(f'  [OK] Triton')" 2>/dev/null || echo "  [WARN] Triton not found"
python3 -c "import ray; print(f'  [OK] Ray {ray.__version__}')" 2>/dev/null || echo "  [WARN] Ray not found"
python3 -c "import vllm; print(f'  [OK] vLLM')" 2>/dev/null || echo "  [WARN] vLLM not found"

echo ""

# ── Environment variables ─────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
export RAY_memory_monitor_refresh_ms=0
export RAY_DEDUP_LOGS=0
export HYDRA_OUTPUT_DIR="$RUN_DIR/hydra"

# Inject paths into config
export MODEL_PATH
export RUN_DIR

# ── Start progress tracker (background) ───────────────────────────────
echo "=== Starting progress tracker ==="
python3 "$PROJECT_ROOT/scripts/kernel_rl/track_progress.py" \
    --run-dir "$RUN_DIR" \
    --sample-interval "$SAMPLE_INTERVAL" \
    --max-samples 8 &
TRACKER_PID=$!
echo "  Tracker PID: $TRACKER_PID"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "=== Shutting down ==="
    kill $TRACKER_PID 2>/dev/null || true
    ray stop 2>/dev/null || true
    echo "=== Done ==="
}
trap cleanup EXIT INT TERM

# ── Launch training ───────────────────────────────────────────────────
echo "=== Starting training ($TOTAL_STEPS steps) ==="
echo ""

cd "$PROJECT_ROOT"

python -m verl.trainer.main_ppo_kernel \
    --config-name ppo_trainer_kernel_h200 \
    hydra.run.dir="$RUN_DIR/hydra" \
    2>&1 | tee "$TRAIN_LOG"

TRAIN_EXIT=${PIPESTATUS[0]}

echo ""
echo "=============================================================================="
if [ $TRAIN_EXIT -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training exited with code $TRAIN_EXIT — check $TRAIN_LOG for errors."
fi
echo "Run directory: $RUN_DIR"
echo "=============================================================================="

exit $TRAIN_EXIT
