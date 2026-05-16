"""Reward functions for Triton kernel evaluation.

Each reward function receives the raw EvalResult (and optionally the PerfResult
tail) and returns a ``{design, code, predict}`` reward dict.  The interface is
intentionally simple so users can swap in custom reward logic without touching
the evaluator internals.

Interface (callable):
    def my_reward(result: EvalResult, target_speedup: float = 2.0) -> Dict[str, float]:
        ...
        return {"design": ..., "code": ..., "predict": ...}

To use a custom reward function:
    evaluator = KernelEvaluator()
    result = evaluator.evaluate(...)
    rewards = my_reward(result, target_speedup=2.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from .evaluator import EvalResult


# ---------------------------------------------------------------------------
# Default reward function (inspired by TritonForge's three-component reward)
# ---------------------------------------------------------------------------

def default_reward(result: EvalResult, target_speedup: float = 2.0) -> Dict[str, float]:
    """Three-component reward: compile + correctness + performance.

    Weights (tuneable per task difficulty):
        w_compile    = 0.3   — bare compilation success
        w_correct    = 0.4   — numerical correctness
        w_perf       = 0.3   — speedup (capped)

    For difficult tasks, users may want to increase w_compile and reduce w_perf
    early in training, then gradually shift.
    """
    if not result.compiled:
        return {"design": 0.0, "code": 0.1, "predict": 0.0}

    r_code = 0.3
    r_design = 0.0
    r_predict = 0.0

    if result.correctness:
        r_code += 0.4
        if result.speedup > 0.0:
            cap = target_speedup * 2.0
            perf_score = min(result.speedup, cap) / cap
            r_code += 0.3 * perf_score
            r_predict = 0.3 * perf_score
            r_design = 0.3 * perf_score

    return {
        "design": round(r_design, 4),
        "code": round(r_code, 4),
        "predict": round(r_predict, 4),
    }


# ---------------------------------------------------------------------------
# Weighted reward — all knobs exposed
# ---------------------------------------------------------------------------

@dataclass
class RewardWeights:
    """Tuneable weights for the three-component reward.

    All values should be non-negative floats.  They do NOT need to sum to 1.0;
    the reward function applies them as absolute additive contributions.
    """
    compile_pass: float = 0.3
    correct_pass: float = 0.4
    perf_scale: float = 0.3       # multiplied by capped speedup ratio
    speedup_cap_mult: float = 2.0  # cap = target_speedup * speedup_cap_mult


def weighted_reward(
    result: EvalResult,
    target_speedup: float = 2.0,
    weights: Optional[RewardWeights] = None,
) -> Dict[str, float]:
    """Reward with configurable weights.

    >>> weights = RewardWeights(compile_pass=0.5, correct_pass=0.3, perf_scale=0.2)
    >>> weighted_reward(result, target_speedup=2.0, weights=weights)
    """
    w = weights or RewardWeights()

    if not result.compiled:
        return {"design": 0.0, "code": 0.1, "predict": 0.0}

    r_design = 0.0
    r_predict = 0.0
    r_code = w.compile_pass

    if result.correctness:
        r_code += w.correct_pass
        if result.speedup > 0.0:
            cap = target_speedup * w.speedup_cap_mult
            perf_score = min(result.speedup, cap) / cap
            perf_bonus = w.perf_scale * perf_score
            r_code += perf_bonus
            r_predict = perf_bonus
            r_design = perf_bonus

    return {
        "design": round(r_design, 4),
        "code": round(r_code, 4),
        "predict": round(r_predict, 4),
    }


# ---------------------------------------------------------------------------
# Registry — allows plugging in custom reward functions by name
# ---------------------------------------------------------------------------

RewardFn = Callable[[EvalResult, float], Dict[str, float]]

_REWARD_REGISTRY: Dict[str, RewardFn] = {
    "default": default_reward,
    "weighted": weighted_reward,
}


def register_reward(name: str, fn: RewardFn) -> None:
    """Register a custom reward function for retrieval by name."""
    _REWARD_REGISTRY[name] = fn


def get_reward_fn(name: str = "default") -> RewardFn:
    """Look up a registered reward function by name."""
    if name not in _REWARD_REGISTRY:
        raise KeyError(f"Unknown reward function '{name}'. Available: {list(_REWARD_REGISTRY)}")
    return _REWARD_REGISTRY[name]
