from __future__ import annotations

from typing import Iterable

import torch

from verl import DataProto
from verl.trainer.ppo import core_algos

from .schema import KERNEL_SECTION_NAMES


def compute_multi_section_advantages(
    batch: DataProto,
    adv_estimator: str,
    section_names: Iterable[str] = KERNEL_SECTION_NAMES,
) -> DataProto:
    """
    Compute section-wise advantages and a merged fallback advantage tensor.

    The merged ``advantages`` field keeps the rest of the training stack
    compatible. The real section-specific training signal is stored in
    ``design_advantages / code_advantages / predict_advantages``.
    """
    responses = batch.batch["responses"]
    response_length = responses.shape[-1]
    response_mask = batch.batch["attention_mask"][:, -response_length:].float()

    merged_scores = torch.zeros_like(response_mask, dtype=torch.float32)
    merged_advantages = torch.zeros_like(response_mask, dtype=torch.float32)
    merged_returns = torch.zeros_like(response_mask, dtype=torch.float32)
    merged_loss_mask = torch.zeros_like(response_mask, dtype=torch.float32)

    group_index = batch.non_tensor_batch["uid"]

    for section in section_names:
        mask_key = f"{section}_mask"
        score_key = f"{section}_token_scores"

        if mask_key not in batch.batch.keys() or score_key not in batch.batch.keys():
            continue

        section_mask = batch.batch[mask_key].float()
        section_scores = batch.batch[score_key].float() * section_mask
        merged_scores = merged_scores + section_scores
        merged_loss_mask = torch.maximum(merged_loss_mask, section_mask)

        if adv_estimator == "grpo":
            section_advantages, section_returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=section_scores,
                eos_mask=section_mask,
                index=group_index,
            )
        elif adv_estimator == "grpo_per_token":
            section_advantages, section_returns = core_algos.compute_grpo_per_token_advantage(
                token_level_rewards=section_scores,
                eos_mask=section_mask,
                index=group_index,
            )
        elif adv_estimator == "no_estimator":
            section_advantages = section_scores
            section_returns = section_scores
        else:
            raise NotImplementedError(f"Unsupported kernel advantage estimator: {adv_estimator}")

        batch.batch[f"{section}_advantages"] = section_advantages.float()
        batch.batch[f"{section}_returns"] = section_returns.float()
        merged_advantages = merged_advantages + section_advantages.float() * section_mask
        merged_returns = merged_returns + section_returns.float() * section_mask

    batch.batch["token_level_scores"] = merged_scores.float()
    batch.batch["token_level_rewards"] = merged_scores.float()
    batch.batch["loss_mask"] = merged_loss_mask.long()
    batch.batch["advantages"] = merged_advantages.float()
    batch.batch["returns"] = merged_returns.float()
    return batch
