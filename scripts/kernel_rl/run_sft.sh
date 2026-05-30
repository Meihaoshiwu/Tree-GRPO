#!/bin/bash
# SFT training on Qwen2.5-Coder-7B-Instruct
# Full-parameter fine-tuning with FSDP on 4 GPUs
set -euo pipefail

MODEL_PATH="/inspire/qb-ilm/project/wuliqifa/public/sdt/models/Qwen2.5-Coder-7B-Instruct"
DATA_PATH="/inspire/qb-ilm/project/wuliqifa/public/sdt/Tree-GRPO/data/kernel_rl/sft_train_v3.jsonl"
TMP_OUTPUT="/tmp/sft_v3_output"
FINAL_OUTPUT="/inspire/qb-ilm/project/wuliqifa/public/sdt/models/Qwen2.5-Coder-7B-Instruct-SFT-kernel-v3"
LOG_DIR="/inspire/qb-ilm/project/wuliqifa/public/sdt/Tree-GRPO/scripts/kernel_rl"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/sft_train_${TIMESTAMP}.log"

echo "=== SFT Training ===" | tee -a "$LOG_FILE"
echo "Model: $MODEL_PATH" | tee -a "$LOG_FILE"
echo "Data:  $DATA_PATH" | tee -a "$LOG_FILE"
echo "Output: $FINAL_OUTPUT (tmp: $TMP_OUTPUT)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun \
    --nproc_per_node=4 \
    --master_port=29501 \
    /inspire/qb-ilm/project/wuliqifa/public/sdt/Tree-GRPO/scripts/kernel_rl/train_sft.py \
    --model_name_or_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$TMP_OUTPUT" \
    --num_train_epochs 10 \
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

echo "=== Copying model to GPFS ===" | tee -a "$LOG_FILE"
mkdir -p "$FINAL_OUTPUT"
LAST_CKPT=$(ls -d "$TMP_OUTPUT"/checkpoint-* 2>/dev/null | tail -1)
if [ -n "$LAST_CKPT" ] && [ -f "$LAST_CKPT/model.safetensors" ]; then
    cp "$LAST_CKPT/model.safetensors" "$FINAL_OUTPUT/"
    cp "$LAST_CKPT/config.json" "$FINAL_OUTPUT/"
    cp "$LAST_CKPT/tokenizer.json" "$FINAL_OUTPUT/"
    cp "$LAST_CKPT/tokenizer_config.json" "$FINAL_OUTPUT/"
    cp "$LAST_CKPT/chat_template.jinja" "$FINAL_OUTPUT/" 2>/dev/null
    cp "$LAST_CKPT/generation_config.json" "$FINAL_OUTPUT/" 2>/dev/null
    echo "Model copied to $FINAL_OUTPUT" | tee -a "$LOG_FILE"
fi
rm -rf "$TMP_OUTPUT"
echo "=== SFT Complete ===" | tee -a "$LOG_FILE"
