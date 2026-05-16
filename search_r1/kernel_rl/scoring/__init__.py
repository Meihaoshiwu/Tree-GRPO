"""
Kernel scoring module adapted from KernelBench (ICML'25).

Three-step evaluation pipeline:
    1. Compilation  — write to tempfile, importlib-load, catch errors
    2. Correctness   — N random-input trials, torch.allclose vs PyTorch reference
    3. Performance   — CUDA-event wall-clock timing, compute speedup

SM-level profiling is available via the optional profiler module (requires ncu).
"""

from .evaluator import KernelEvaluator, EvalConfig, EvalResult

__all__ = ["KernelEvaluator", "EvalConfig", "EvalResult"]
