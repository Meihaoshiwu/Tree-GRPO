#!/bin/bash
# SFT training on Qwen2.5-Coder-7B-Instruct
# Full-parameter fine-tuning with FSDP on 4 GPUs
set -euo pipefail

MODEL_PATH="/inspire/qb-ilm/project/wuliqifa/public/sdt/models/Qwen2.5-Coder-7B-Instruct"
DATA_PATH="/inspire/qb-ilm/project/wuliqifa/public/sdt/Tree-GRPO/data/kernel_rl/sft_train_v2.jsonl"
OUTPUT_DIR="/inspire/qb-ilm/project/wuliqifa/public/sdt/models/Qwen2.5-Coder-7B-Instruct-SFT-kernel-v2"
LOG_DIR="/inspire/qb-ilm/project/wuliqifa/public/sdt/Tree-GRPO/scripts/kernel_rl"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/sft_train_${TIMESTAMP}.log"

echo "=== SFT Training ===" | tee -a "$LOG_FILE"
echo "Model: $MODEL_PATH" | tee -a "$LOG_FILE"
echo "Data:  $DATA_PATH" | tee -a "$LOG_FILE"
echo "Output: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun \
    --nproc_per_node=4 \
    --master_port=29501 \
    /inspire/qb-ilm/project/wuliqifa/public/sdt/Tree-GRPO/scripts/kernel_rl/train_sft.py \
    --model_name_or_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --bf16 \
    --logging_steps 1 \
    --save_steps 100 \
    --save_total_limit 2 \
    --fsdp "full_shard auto_wrap" \
    2>&1 | tee -a "$LOG_FILE"

echo "=== SFT Complete ===" | tee -a "$LOG_FILE"
echo "Model saved to: $OUTPUT_DIR" | tee -a "$LOG_FILE"
