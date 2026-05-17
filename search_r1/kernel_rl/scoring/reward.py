"""Reward functions for Triton kernel evaluation — Phase 2 design.

Three independent section rewards:
  - code:    speedup-based S-curve (no penalty for mediocre, 2 penalty exceptions)
  - design:  strategy-implementation consistency
  - predict: per-metric binary accuracy

Plus anti-cheat detection: Python-wrapping, suspiciously short code.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from .evaluator import EvalResult


# ══════════════════════════════════════════════════════════════════════
# Anti-cheat detection
# ══════════════════════════════════════════════════════════════════════

def detect_python_cheating(code_text: str) -> bool:
    """Return True if the code uses PyTorch ops without calling Triton kernel."""
    if not code_text or len(code_text.strip()) < 5:
        return False
    has_triton_kernel = bool(re.search(r'@triton\.jit', code_text))
    if not has_triton_kernel:
        return False  # not even trying to be Triton — handled by compile fail

    # Check: Is there a triton kernel launch in a non-decorated function?
    triton_launch = bool(re.search(
        r'\w+\s*\[grid\]\(|kernel\s*\[', code_text
    ))

    # Check: Are there PyTorch op calls in the wrapper?
    pytorch_ops = bool(re.search(
        r'torch\.(nn\.functional\.|Tensor\.|sum\(|mean\(|matmul\(|add\()',
        code_text
    ))

    # Cheating: has @triton.jit but wrapper calls PyTorch instead of launching it
    if pytorch_ops and not triton_launch:
        return True
    return False


def is_too_short(code_text: str, reference_python: str) -> bool:
    """Return True if code is suspiciously shorter than reference."""
    code_lines = [l for l in code_text.split('\n')
                  if l.strip() and not l.strip().startswith('#')]
    ref_lines = [l for l in (reference_python or '').split('\n') if l.strip()]
    if not ref_lines:
        return len(code_lines) < 3
    return len(code_lines) < len(ref_lines)


# ══════════════════════════════════════════════════════════════════════
# Code Reward — speedup-based S-curve
# ══════════════════════════════════════════════════════════════════════

def code_reward(
    compiled: bool,
    correctness: bool,
    speedup: float,
    is_cheating: bool = False,
    code_too_short: bool = False,
) -> float:
    """Speedup-based S-curve. No penalty for mediocre code, two exceptions.

    Penalty exceptions:
      1. is_cheating: compiled OK but calls PyTorch, never launches Triton → -0.3
      2. compile_failed + code_too_short: didn't really try → -0.2

    Normal reward:
      - compile failed but tried: 0.05
      - compiled + incorrect: 0.15
      - compiled + correct: 0.2 (slow) to 0.70 (max) via S-curve
    """
    # ── Penalty path ──────────────────────────────────────────────
    if is_cheating:
        return -0.3

    if not compiled and code_too_short:
        return -0.2

    # ── Effort path (compile failed but tried) ─────────────────────
    if not compiled:
        return 0.05

    # ── Compile OK but wrong ──────────────────────────────────────
    if not correctness:
        return 0.15

    # ── Compile OK + correct → speedup-based ──────────────────────
    if speedup <= 0.5:
        return 0.2
    elif speedup < 0.8:
        return 0.25
    else:
        # S-curve: flat below target, steep around midpoint, capped
        k = 4.0
        midpoint = 1.2
        sigmoid = 1.0 / (1.0 + math.exp(-k * (speedup - midpoint)))
        return 0.3 + sigmoid * 0.4


# ══════════════════════════════════════════════════════════════════════
# Design Reward — strategy vs implementation consistency
# ══════════════════════════════════════════════════════════════════════

def design_reward(
    design_text: str,
    code_text: str,
    speedup: float = 0.0,
) -> float:
    """Check if claimed design strategy matches actual implementation.

    Heuristics (best-effort, not perfect):
      1. Block size claim matches code (0.1)
      2. Memory pattern claim matches code (0.1)
      3. Has specific (non-generic) content (0.1)
      4. Performance was better than baseline (0.1 bonus)
    """
    score = 0.0

    # 1. Block size consistency
    bs_design = _extract_num(design_text, [r'BLOCK[_\s]*SIZE\s*[=:]\s*(\d+)',
                                            r'block\s*size\s*(?:of\s*)?(\d+)'])
    bs_code = _extract_num(code_text, [r'BLOCK[_\s]*SIZE\s*[=:]\s*(\d+)',
                                        r'BLOCK\s*=\s*(\d+)'])
    if bs_design and bs_code and bs_design == bs_code:
        score += 0.1

    # 2. Memory pattern hint
    design_lower = design_text.lower()
    if "coalesced" in design_lower and "stride" not in code_text.lower():
        if re.search(r'offsets|pid\s*\*\s*BLOCK', code_text):
            score += 0.1

    # 3. Has specific content (not just generic template)
    specific_markers = [
        r'because', r'since', r'due to', r'reason',
        r'\d+\s*(GB|MB|KB|bytes|elements)',
        r'occupancy', r'register pressure', r'bank conflict',
        r'tensor core', r'warp',
    ]
    if any(re.search(m, design_lower) for m in specific_markers):
        score += 0.1

    # 4. Performance bonus
    if speedup >= 1.0:
        score += 0.1

    return min(score, 0.4)


def _extract_num(text: str, patterns: list) -> Optional[int]:
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


# ══════════════════════════════════════════════════════════════════════
# Predict Reward — per-metric binary accuracy
# ══════════════════════════════════════════════════════════════════════

def predict_reward(
    predict_text: str,
    compiled: bool,
    speedup: float,
    correctness: bool,
) -> float:
    """Score prediction accuracy. Binary per metric, penalty for wildly wrong."""
    score = 0.0
    pred_lower = predict_text.lower()

    # Metric 1: compile prediction
    pred_will_compile = _parse_compile_prediction(pred_lower)
    if pred_will_compile is not None:
        if pred_will_compile == compiled:
            score += 0.1
        elif not compiled and pred_will_compile:
            # Said would compile but didn't → minor penalty
            score -= 0.03

    # Metric 2: speedup prediction
    pred_sp = _parse_speedup_prediction(pred_lower)
    if pred_sp is not None and speedup > 0:
        error = abs(pred_sp - speedup) / max(pred_sp, speedup, 1.0)
        if error < 0.2:
            score += 0.15       # very accurate
        elif error < 0.5:
            score += 0.08       # moderately accurate
        elif error > 2.0:
            score -= 0.05       # wildly wrong

    # Metric 3: bottleneck identification
    if speedup > 0 and correctness:
        actual_bn = "compute" if speedup < 0.8 else ("memory" if speedup < 1.5 else "balanced")
        if actual_bn in pred_lower or "memory-bound" in pred_lower or "compute-bound" in pred_lower:
            score += 0.05

    return max(min(score, 0.3), -0.1)


def _parse_compile_prediction(text: str) -> Optional[bool]:
    if re.search(r'will\s+compile|should\s+compile|compiles', text):
        return True
    if re.search(r'fail\s+to\s+compile|won\'?t\s+compile|compilation\s+error', text):
        return False
    return None


def _parse_speedup_prediction(text: str) -> Optional[float]:
    m = re.search(r'speedup\s*(?:of\s*)?(?:~|about\s*|approximately\s*)?(\d+\.?\d*)\s*[x×]', text)
    if m:
        return float(m.group(1))
    m = re.search(r'(\d+\.?\d*)\s*[x×]\s*(?:faster|speedup)', text)
    if m:
        return float(m.group(1))
    return None


# ══════════════════════════════════════════════════════════════════════
# Combined Reward
# ══════════════════════════════════════════════════════════════════════

def compute_rewards(
    result: EvalResult,
    design_text: str = "",
    code_text: str = "",
    predict_text: str = "",
    reference_python: str = "",
    target_speedup: float = 2.0,
) -> Dict[str, float]:
    """Compute three independent section rewards from eval result.

    Returns {"design": ..., "code": ..., "predict": ...}
    """
    # Anti-cheat checks
    cheating = detect_python_cheating(code_text) if code_text else False
    too_short = is_too_short(code_text, reference_python) if code_text else False

    r_code = code_reward(
        compiled=result.compiled,
        correctness=result.correctness,
        speedup=result.speedup,
        is_cheating=cheating,
        code_too_short=too_short,
    )

    r_design = design_reward(
        design_text=design_text,
        code_text=code_text,
        speedup=result.speedup,
    )

    r_predict = predict_reward(
        predict_text=predict_text,
        compiled=result.compiled,
        speedup=result.speedup,
        correctness=result.correctness,
    )

    return {
        "design": round(r_design, 4),
        "code": round(r_code, 4),
        "predict": round(r_predict, 4),
    }


# ══════════════════════════════════════════════════════════════════════
# Registry
# ══════════════════════════════════════════════════════════════════════

RewardFn = Callable[..., Dict[str, float]]

_REWARD_REGISTRY: Dict[str, RewardFn] = {
    "phase2": compute_rewards,
}


def register_reward(name: str, fn: RewardFn) -> None:
    _REWARD_REGISTRY[name] = fn


def get_reward_fn(name: str = "phase2") -> RewardFn:
    if name not in _REWARD_REGISTRY:
        raise KeyError(f"Unknown: {name}. Available: {list(_REWARD_REGISTRY)}")
    return _REWARD_REGISTRY[name]
