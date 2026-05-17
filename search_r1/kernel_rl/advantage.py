"""Section-wise advantage computation with GRPO + generational signals.

Two sources of advantage:
  1. GRPO same-depth: (r_i - group_mean) / (group_std + eps)
  2. Generational:     (r_child - r_parent) / (r_parent + eps)

Combined: total_adv = w_grpo * adv_grpo + w_gen * adv_gen
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch

from verl import DataProto
from verl.trainer.ppo import core_algos

from .schema import KERNEL_SECTION_NAMES


def compute_multi_section_advantages(
    batch: DataProto,
    adv_estimator: str,
    section_names: Iterable[str] = KERNEL_SECTION_NAMES,
    grpo_weight: float = 0.5,
    gen_weight: float = 0.5,
) -> DataProto:
    """Compute section-wise advantages combining GRPO and generational signals.

    Args:
        grpo_weight: weight for GRPO same-depth advantage (default 0.5)
        gen_weight: weight for generational (child-vs-parent) advantage (default 0.5)
    """
    responses = batch.batch["responses"]
    response_length = responses.shape[-1]
    response_mask = batch.batch["attention_mask"][:, -response_length:].float()

    merged_scores = torch.zeros_like(response_mask, dtype=torch.float32)
    merged_advantages = torch.zeros_like(response_mask, dtype=torch.float32)
    merged_returns = torch.zeros_like(response_mask, dtype=torch.float32)
    merged_loss_mask = torch.zeros_like(response_mask, dtype=torch.float32)

    group_index = batch.non_tensor_batch["uid"]
    depth_info = batch.non_tensor_batch.get("depth", None)

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
            raise NotImplementedError(f"Unknown advantage estimator: {adv_estimator}")

        # Add generational advantage signal if depth info is available
        if depth_info is not None:
            gen_adv = _compute_generational_advantages(
                section_scores, section_mask, group_index, depth_info
            )
            # Combine: weighted sum, preserving scale
            section_advantages = (
                grpo_weight * section_advantages.float()
                + gen_weight * gen_adv.float() * section_mask
            )

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


def _compute_generational_advantages(
    token_scores: torch.Tensor,
    section_mask: torch.Tensor,
    group_index: torch.Tensor,
    depth_info: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute child-vs-parent advantage: (r_child - r_parent) / (r_parent + eps).

    Each sample has a node-level reward = sum(token_scores * mask) / sum(mask).
    Generational advantage uses the difference between a sample and its parent
    (identified by same group_index, shallower depth).
    """
    batch_size = token_scores.shape[0]
    gen_adv = torch.zeros_like(token_scores, dtype=torch.float32)

    # Compute per-sample scalar rewards
    mask_sum = section_mask.sum(dim=1).clamp(min=1)
    sample_rewards = (token_scores * section_mask).sum(dim=1) / mask_sum

    # For each sample, find parent reward (same group, depth-1)
    for i in range(batch_size):
        my_uid = group_index[i].item() if isinstance(group_index[i], torch.Tensor) else group_index[i]
        my_depth = depth_info[i].item() if isinstance(depth_info[i], torch.Tensor) else depth_info[i]

        # Find parent: same group_uid, depth = my_depth - 1
        parent_r = None
        for j in range(batch_size):
            parent_uid = group_index[j].item() if isinstance(group_index[j], torch.Tensor) else group_index[j]
            parent_depth = depth_info[j].item() if isinstance(depth_info[j], torch.Tensor) else depth_info[j]
            if parent_uid == my_uid and parent_depth == my_depth - 1:
                parent_r = sample_rewards[j].item()
                break

        if parent_r is not None and parent_r > eps:
            gen_adv[i] = (sample_rewards[i].item() - parent_r) / (parent_r + eps)

    return gen_adv
