"""Section-wise advantage computation with GRPO + generational signals.

Two sources of advantage, combined with adaptive normalization:
  1. GRPO same-depth: (r_i - group_mean) / (group_std + eps)
  2. Generational:     (r_child - r_parent) / 0.3  (absolute ÷ baseline)

Adaptive merge: both normalized to unit variance, then equal-weighted.
Fallback: when both fail (all-zero group, no parent), use absolute baseline.
"""

from __future__ import annotations

from typing import Iterable

import torch

from verl import DataProto
from verl.trainer.ppo import core_algos

from .schema import KERNEL_SECTION_NAMES

# Absolute baseline: "compiled but incorrect" reward floor
ABSOLUTE_BASELINE = 0.15
GENERATIONAL_BASELINE = 0.3  # code_reward at speedup=1.0

# Thresholds
GRPO_MIN_STD = 0.05  # below this, GRPO considered invalid
GEN_MIN_VAR = 1e-8   # below this, generational considered invalid


def compute_multi_section_advantages(
    batch: DataProto,
    adv_estimator: str,
    section_names: Iterable[str] = KERNEL_SECTION_NAMES,
) -> DataProto:
    """Compute section-wise advantages with adaptive normalization."""
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

        # ── Step 1: base advantage from estimator ──────────────────
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
            raise NotImplementedError(f"Unknown: {adv_estimator}")

        # ── Step 2: generational advantage ─────────────────────────
        if depth_info is not None:
            gen_adv = _compute_generational_advantages(
                section_scores, section_mask, group_index, depth_info
            )

            # ── Step 3: adaptive normalization ────────────────────
            adv_grpo = section_advantages.float()
            gen = gen_adv.float()

            grpo_std = adv_grpo.std().clamp(min=1e-6)
            gen_std = gen.std().clamp(min=1e-6)

            grpo_valid = grpo_std > GRPO_MIN_STD
            gen_valid = gen_std > GEN_MIN_VAR

            if grpo_valid and gen_valid:
                # Normalize both to unit variance → equal weight merge
                adv_combined = adv_grpo / grpo_std + gen / gen_std
            elif grpo_valid:
                adv_combined = adv_grpo
            elif gen_valid:
                adv_combined = gen
            else:
                # Both invalid → absolute baseline fallback
                mask_sum = section_mask.sum(dim=1).clamp(min=1)
                sample_rewards = (section_scores * section_mask).sum(dim=1) / mask_sum
                abs_adv = (sample_rewards - ABSOLUTE_BASELINE) / GENERATIONAL_BASELINE
                adv_combined = abs_adv.unsqueeze(-1).expand_as(section_mask)

            section_advantages = adv_combined
        # ── endif depth_info ────────────────────────────────────────

        batch.batch[f"{section}_advantages"] = section_advantages.float()
        batch.batch[f"{section}_returns"] = section_returns.float()

        # Single mask application at merge time (no double masking)
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
    group_index: list,
    depth_info: list,
) -> torch.Tensor:
    """Generational advantage: (r_child - r_parent) / baseline.

    Uses absolute difference normalized by GENERATIONAL_BASELINE (0.3),
    which is the code_reward at speedup=1.0 — a meaningful reference point.
    """
    batch_size = token_scores.shape[0]
    gen_adv = torch.zeros(batch_size, dtype=torch.float32, device=token_scores.device)

    # Per-sample scalar rewards
    mask_sum = section_mask.sum(dim=1).clamp(min=1)
    sample_rewards = (token_scores * section_mask).sum(dim=1) / mask_sum

    # Build parent map: for each depth, find parent (same tree_uid, depth-1)
    # group_index encodes tree_uid, depth_info is per-sample depth
    parent_rewards = torch.zeros(batch_size, dtype=torch.float32, device=token_scores.device)
    has_parent = torch.zeros(batch_size, dtype=torch.bool, device=token_scores.device)

    for i in range(batch_size):
        my_uid = group_index[i].item() if hasattr(group_index[i], 'item') else group_index[i]
        my_depth = depth_info[i].item() if hasattr(depth_info[i], 'item') else depth_info[i]
        if my_depth <= 0:
            continue
        for j in range(batch_size):
            p_uid = group_index[j].item() if hasattr(group_index[j], 'item') else group_index[j]
            p_depth = depth_info[j].item() if hasattr(depth_info[j], 'item') else depth_info[j]
            if p_uid == my_uid and p_depth == my_depth - 1:
                parent_rewards[i] = sample_rewards[j]
                has_parent[i] = True
                break

    # Absolute difference ÷ baseline
    valid = has_parent & (parent_rewards > 1e-8)
    gen_adv[valid] = (sample_rewards[valid] - parent_rewards[valid]) / GENERATIONAL_BASELINE

    return gen_adv
