"""Full-parameter SFT for Qwen2.5-Coder-7B-Instruct with FSDP.

Usage:
    torchrun --nproc_per_node=4 train_sft.py \
        --model_name_or_path /path/to/model \
        --data_path /path/to/sft_train.jsonl \
        --output_dir /path/to/output \
        --num_train_epochs 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint


# Known TrainingArguments fields (from transformers source)
_TRAINER_KNOWN = {
    "output_dir", "overwrite_output_dir", "do_train", "do_eval", "do_predict",
    "eval_strategy", "prediction_loss_only", "per_device_train_batch_size",
    "per_device_eval_batch_size", "gradient_accumulation_steps",
    "learning_rate", "weight_decay", "adam_beta1", "adam_beta2", "adam_epsilon",
    "max_grad_norm", "num_train_epochs", "max_steps", "lr_scheduler_type",
    "warmup_ratio", "warmup_steps", "logging_strategy", "logging_steps",
    "logging_first_step", "save_strategy", "save_steps", "save_total_limit",
    "save_safetensors", "save_on_each_node", "save_only_model", "seed",
    "bf16", "fp16", "fp16_opt_level", "bf16_full_eval", "fp16_full_eval",
    "tf32", "local_rank", "ddp_backend", "ddp_find_unused_parameters",
    "dataloader_num_workers", "dataloader_pin_memory",
    "dataloader_persistent_workers", "run_name", "report_to",
    "hub_model_id", "hub_token", "push_to_hub", "hub_strategy",
    "gradient_checkpointing", "gradient_checkpointing_kwargs",
    "fsdp", "fsdp_config", "fsdp_min_num_params",
    "fsdp_transformer_layer_cls_to_wrap", "fsdp_forward_prefetch",
    "fsdp_use_orig_params", "deepspeed", "label_smoothing_factor",
    "optim", "group_by_length", "dataloader_drop_last",
    "ignore_data_skip", "include_inputs_for_metrics",
    "remove_unused_columns", "disable_tqdm", "resume_from_checkpoint",
    "skip_memory_metrics",
}


class KernelSFTDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: List[Dict[str, str]] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        prompt = ex["prompt"]
        completion = ex["completion"]

        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
        completion_ids = self.tokenizer(
            completion + self.tokenizer.eos_token, add_special_tokens=False
        ).input_ids

        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids

        if len(input_ids) > self.max_length:
            overflow = len(input_ids) - self.max_length
            input_ids = input_ids[overflow:]
            labels = labels[overflow:]
            remaining = max(0, len(prompt_ids) - overflow)
            for i in range(min(remaining, len(labels))):
                labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    max_len = max(len(b["input_ids"]) for b in batch)
    pad = 0
    return {
        "input_ids": torch.tensor(
            [b["input_ids"] + [pad] * (max_len - len(b["input_ids"])) for b in batch],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [b["attention_mask"] + [0] * (max_len - len(b["attention_mask"])) for b in batch],
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            [b["labels"] + [-100] * (max_len - len(b["labels"])) for b in batch],
            dtype=torch.long,
        ),
    }


def _filter_trainer_args(argv: List[str]) -> List[str]:
    """Keep only args that are known TrainingArguments fields."""
    out = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg.startswith("--"):
            key = arg[2:].replace("-", "_")
            if key in _TRAINER_KNOWN:
                out.append(arg)
                if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                    out.append(argv[i + 1])
                    i += 1
            elif arg in ("--help", "-h"):
                out.append(arg)
            # else: silently skip unknown args
        i += 1
    return out


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Parse script-only args
    script_parser = argparse.ArgumentParser()
    script_parser.add_argument("--model_name_or_path", type=str, required=True)
    script_parser.add_argument("--data_path", type=str, required=True)
    script_ns, _ = script_parser.parse_known_args(sys.argv[1:])

    if local_rank == 0:
        print(f"Model: {script_ns.model_name_or_path}")
        print(f"Data: {script_ns.data_path}")
        print(f"GPUs: {world_size}")

    # Parse trainer args (filtered)
    trainer_argv = _filter_trainer_args(sys.argv[1:])
    if local_rank == 0:
        print(f"Trainer args: {trainer_argv}")

    parser = HfArgumentParser(TrainingArguments)
    training_args = parser.parse_args_into_dataclasses(trainer_argv)[0]

    # Set fsdp_transformer_layer_cls_to_wrap via fsdp_config
    if training_args.fsdp_config is None:
        training_args.fsdp_config = {}
    if isinstance(training_args.fsdp_config, str):
        import json
        training_args.fsdp_config = json.loads(training_args.fsdp_config)
    if not training_args.fsdp_config.get("transformer_layer_cls_to_wrap"):
        training_args.fsdp_config["transformer_layer_cls_to_wrap"] = "Qwen2DecoderLayer"

    if local_rank == 0:
        print(f"Output dir: {training_args.output_dir}")
        print(f"Epochs: {training_args.num_train_epochs}")

    tokenizer = AutoTokenizer.from_pretrained(script_ns.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = KernelSFTDataset(
        data_path=script_ns.data_path,
        tokenizer=tokenizer,
        max_length=2048,
    )
    if local_rank == 0:
        print(f"Train examples: {len(train_dataset)}")

    model = AutoModelForCausalLM.from_pretrained(
        script_ns.model_name_or_path,
        dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    model.gradient_checkpointing_enable()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )
    # Set processing_class (tokenizer) for save
    trainer.processing_class = tokenizer

    trainer.train()

    if local_rank == 0:
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        print(f"Saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
