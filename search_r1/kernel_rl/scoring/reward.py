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

def analyze_code_structure(code_text: str) -> dict:
    """AST-based structural analysis of generated Triton code.

    Returns a dict with:
      - has_triton_jit: bool — is there a @triton.jit decorated function?
      - triton_launched: bool — is any @triton.jit kernel called via [...] launch?
      - pytorch_called: bool — are PyTorch ops called in wrapper code?
      - is_cheating: bool — has @triton.jit but calls PyTorch without launching Triton
      - is_parasitic: bool — has @triton.jit but the kernel body is trivially empty (<3 lines)

    Single AST parse replaces both string checks and regex detection.
    """
    result = {
        "has_triton_jit": False,
        "triton_launched": False,
        "pytorch_called": False,
        "is_cheating": False,
        "is_parasitic": False,
    }

    if not code_text or len(code_text.strip()) < 10:
        return result

    try:
        import ast
        tree = ast.parse(code_text)
    except SyntaxError:
        return result  # syntax error → compile will fail anyway

    # Find @triton.jit decorated functions and their body sizes
    triton_functions = {}  # name -> body_line_count
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                dec_str = ast.unparse(dec) if hasattr(ast, 'unparse') else ast.dump(dec)
                if 'triton.jit' in dec_str or 'jit' in dec_str:
                    # Count non-empty body lines
                    body_lines = [l for l in ast.unparse(node).split('\n')
                                  if l.strip() and not l.strip().startswith('#')]
                    triton_functions[node.name] = len(body_lines)

    if triton_functions:
        result["has_triton_jit"] = True

    # Find function calls in non-decorated (wrapper) code
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            call_str = ast.unparse(node) if hasattr(ast, 'unparse') else ast.dump(node)
            # Triton kernel launch: fn_name[grid](...)
            for fn_name in triton_functions:
                if fn_name in call_str and ('[' in call_str or 'grid' in call_str.lower()):
                    result["triton_launched"] = True
            # PyTorch ops in wrapper
            if any(op in call_str for op in [
                'torch.nn.functional.', 'torch.relu', 'torch.gelu',
                'torch.softmax', 'torch.matmul', 'torch.sum(', 'torch.mean(',
                'F.relu', 'F.gelu', 'F.softmax',
            ]):
                result["pytorch_called"] = True

    # Cheating: has @triton.jit but never launches it, uses PyTorch instead
    if result["has_triton_jit"] and result["pytorch_called"] and not result["triton_launched"]:
        result["is_cheating"] = True

    # Parasitic: has @triton.jit but kernel body is trivially small (empty shell)
    if result["has_triton_jit"]:
        min_body = min(triton_functions.values()) if triton_functions else 0
        if min_body < 3:
            result["is_parasitic"] = True

    return result


def detect_python_cheating(code_text: str) -> bool:
    """Legacy wrapper — use analyze_code_structure() directly for full info."""
    return analyze_code_structure(code_text)["is_cheating"]


# ══════════════════════════════════════════════════════════════════════
# Code Reward — speedup-based S-curve
# ══════════════════════════════════════════════════════════════════════

def code_reward(
    compiled: bool,
    correctness: bool,
    speedup: float,
    correctness_ratio: float = 0.0,
    is_cheating: bool = False,
    has_triton_jit: bool = True,
    is_parasitic: bool = False,
) -> float:
    """Speedup-based S-curve + correctness proportion. No penalty for mediocre code.

    Penalty exceptions:
      1. is_cheating: AST detects @triton.jit defined but never launched,
         PyTorch called instead → -0.3
      2. no @triton.jit at all + compile failed → -0.2 (didn't even try)
      3. is_parasitic: @triton.jit exists but kernel body is trivially
         short (<3 lines) → treated as "didn't try" (0.05 max)

    Normal reward:
      - compile failed but tried: 0.05
      - compiled + incorrect: 0.15
      - compiled + correct: 0.2 (slow) to 0.70 (max) via S-curve
    """
    # ── Penalty path ──────────────────────────────────────────────
    if is_cheating:
        return -0.3

    if not compiled and (not has_triton_jit or is_parasitic):
        return -0.2

    # ── Effort path (compile failed but tried) ─────────────────────
    if not compiled:
        return 0.05

    # ── Compile OK but wrong ──────────────────────────────────────
    if not correctness:
        return 0.15

    # ── Compile OK + correct → speedup-based + correctness proportion ──
    if speedup <= 0.5:
        base = 0.2
    elif speedup < 0.8:
        base = 0.25
    else:
        k = 4.0
        midpoint = 1.2
        sigmoid = 1.0 / (1.0 + math.exp(-k * (speedup - midpoint)))
        base = 0.3 + sigmoid * 0.4
    # Modest correctness proportion bonus (0-0.05)
    return base + 0.05 * correctness_ratio


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
    """Compute three independent section rewards (Phase 2 design).

    Returns {"design": ..., "code": ..., "predict": ...}
    """
    # Unified AST analysis — one parse, all structural facts
    ast_info = analyze_code_structure(code_text) if code_text else {}
    cheating = ast_info.get("is_cheating", False)
    has_triton_jit = ast_info.get("has_triton_jit", False)
    is_parasitic = ast_info.get("is_parasitic", False)

    # Correctness proportion
    total_trials = max(result.num_correct_trials, 1)
    correctness_ratio = result.num_passed_trials / total_trials if result.compiled else 0.0

    r_code = code_reward(
        compiled=result.compiled,
        correctness=result.correctness,
        speedup=result.speedup,
        correctness_ratio=correctness_ratio,
        is_cheating=cheating,
        has_triton_jit=has_triton_jit,
        is_parasitic=is_parasitic,
    )

    r_design = design_reward(
        design_text=design_text,
        code_text=code_text,
        speedup=result.speedup,
    )

    # Predict: always active (three sections independent)
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
