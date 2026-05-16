"""Main kernel evaluator: compile → correctness → performance.

Adapted from KernelBench ``eval_kernel_against_ref`` and TritonForge's
multi-component reward design.  Compilation runs in a subprocess for safety;
correctness and performance run in-process after compilation succeeds.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


@dataclass
class EvalConfig:
    compile_timeout_s: int = 120
    num_correct_trials: int = 3
    atol: float = 1e-4
    rtol: float = 1e-4
    dtype: str = "float32"
    measure_perf: bool = True
    num_warmup: int = 5
    num_perf_trials: int = 20
    discard_first: int = 2
    excessive_speedup_threshold: float = 10.0
    seed: int = 42
    verbose: bool = False


@dataclass
class EvalResult:
    compiled: bool = False
    correctness: bool = False
    timed_out: bool = False
    status: str = ""
    compile_time_s: float = 0.0
    runtime_ms: float = 0.0
    ref_runtime_ms: float = 0.0
    speedup: float = 0.0
    num_correct_trials: int = 0
    num_passed_trials: int = 0
    max_diff: float = 0.0
    error_type: str = ""
    error_msg: str = ""
    error_traceback: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class KernelEvaluator:
    """Evaluate a generated Triton kernel against a PyTorch reference."""

    def __init__(self, config: Optional[EvalConfig] = None):
        self.config = config or EvalConfig()
        self._loaded_modules: List[Any] = []  # track for cleanup

    # ------------------------------------------------------------------
    def evaluate(
        self,
        ref_code: str,
        new_code: str,
        problem_format: str = "kernelbench",
        device_id: int = 0,
    ) -> EvalResult:
        cfg = self.config
        device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        dtype = _str_to_dtype(cfg.dtype)

        # ── Step 0: Compile new_code in subprocess (crash isolation) ───
        from .compiler import compile_in_subprocess

        comp = compile_in_subprocess(code=new_code, timeout_s=cfg.compile_timeout_s)
        if not comp.success:
            return EvalResult(
                compiled=False,
                status="compile_error" if comp.error_type != "timeout" else "timeout",
                timed_out=(comp.error_type == "timeout"),
                compile_time_s=comp.elapsed_s,
                error_type=comp.error_type,
                error_msg=_truncate(comp.stderr, 2000),
                metadata={"compile_stdout": comp.stdout[:1000]},
            )

        # ── Step 1: Load in-process (importlib for Triton, exec for ref) ──
        try:
            from .compiler import compile_in_process, cleanup_compiled_module

            new_comp = compile_in_process(new_code)
            if not new_comp.success:
                return EvalResult(
                    compiled=False,
                    status="compile_error",
                    compile_time_s=comp.elapsed_s + new_comp.elapsed_s,
                    error_type=new_comp.error_type,
                    error_msg=_truncate(new_comp.stderr, 2000),
                )

            mod = __import__(new_comp.module_name)
            ref_fn, new_fn, input_gen = _extract_functions(ref_code, mod, problem_format)

            ref_fn = _to_device_fn(ref_fn, device, dtype)
            new_fn = _to_device_fn(new_fn, device, dtype)

            self._loaded_modules.append((new_comp.module_name, new_comp.tempfile_path))

        except Exception as exc:
            return EvalResult(
                compiled=False,
                status="compile_error",
                compile_time_s=comp.elapsed_s,
                error_type="load_error",
                error_msg=_truncate(f"{type(exc).__name__}: {exc}", 2000),
                error_traceback=_truncate(traceback.format_exc(), 4000),
            )

        # ── Step 2: Correctness ────────────────────────────────────────
        from .correctness import check_correctness

        corr_result = check_correctness(
            ref_fn=ref_fn,
            new_fn=new_fn,
            input_gen=input_gen,
            num_trials=cfg.num_correct_trials,
            rtol=cfg.rtol,
            atol=cfg.atol,
            device=device,
            dtype=dtype,
            seed=cfg.seed,
            verbose=cfg.verbose,
        )

        if not corr_result.passed:
            return EvalResult(
                compiled=True,
                correctness=False,
                status="incorrect",
                compile_time_s=comp.elapsed_s,
                error_type=corr_result.error_type,
                error_msg=corr_result.error_msg,
                num_correct_trials=corr_result.trials_total,
                num_passed_trials=corr_result.trials_passed,
                max_diff=corr_result.max_diff,
            )

        # ── Step 3: Performance ─────────────────────────────────────────
        if not cfg.measure_perf:
            return EvalResult(
                compiled=True,
                correctness=True,
                status="correct",
                compile_time_s=comp.elapsed_s,
                num_correct_trials=corr_result.trials_total,
                num_passed_trials=corr_result.trials_passed,
            )

        try:
            from .benchmark import measure_performance

            def _ref_closure():
                torch.manual_seed(cfg.seed)
                inp = input_gen()
                torch.manual_seed(cfg.seed)
                # ref_fn already wrapped with _to_device_fn, so pass raw inputs
                return ref_fn(*inp)

            def _new_closure():
                torch.manual_seed(cfg.seed)
                inp = input_gen()
                torch.manual_seed(cfg.seed)
                return new_fn(*inp)

            perf_result = measure_performance(
                kernel_fn=_new_closure,
                args=[],
                ref_kernel_fn=_ref_closure,
                num_warmup=cfg.num_warmup,
                num_trials=cfg.num_perf_trials,
                discard_first=cfg.discard_first,
                device=device,
                verbose=cfg.verbose,
            )

            metadata: Dict[str, Any] = {
                "hardware": perf_result.hardware,
                "num_perf_trials": perf_result.trials,
                "ref_runtime_ms": perf_result.ref_runtime_ms,
                "runtime_std_ms": perf_result.runtime_std_ms,
            }
            if perf_result.speedup > cfg.excessive_speedup_threshold:
                metadata["excessive_speedup"] = True
                metadata["excessive_speedup_value"] = perf_result.speedup

            return EvalResult(
                compiled=True,
                correctness=True,
                status="correct",
                compile_time_s=comp.elapsed_s,
                runtime_ms=perf_result.runtime_ms,
                ref_runtime_ms=perf_result.ref_runtime_ms,
                speedup=perf_result.speedup,
                num_correct_trials=corr_result.trials_total,
                num_passed_trials=corr_result.trials_passed,
                metadata=metadata,
            )
        except Exception as exc:
            return EvalResult(
                compiled=True,
                correctness=True,
                status="correct",
                compile_time_s=comp.elapsed_s,
                error_type="perf_error",
                error_msg=_truncate(str(exc), 1000),
                num_correct_trials=corr_result.trials_total,
                num_passed_trials=corr_result.trials_passed,
            )

    # ------------------------------------------------------------------
    @staticmethod
    def compute_rewards(
        result: EvalResult,
        target_speedup: float = 2.0,
        reward_fn: Optional[str] = None,
    ) -> Dict[str, float]:
        """Compute scalar rewards using the configured or default reward function.

        Args:
            result: Completed EvalResult from ``evaluate()``.
            target_speedup: Target speedup for performance-scaling.
            reward_fn: Name of a registered reward function, or None for default.
        """
        from .reward import get_reward_fn
        fn = get_reward_fn(reward_fn or "default")
        return fn(result, target_speedup)

    @staticmethod
    def build_feedback(result: EvalResult) -> str:
        lines = []
        if not result.compiled:
            lines.append("[COMPILE ERROR]")
            lines.append(f"  error_type: {result.error_type}")
            lines.append(f"  message: {result.error_msg[:500]}")
            return "\n".join(lines)
        lines.append("[COMPILED] OK")
        if not result.correctness:
            lines.append(f"[CORRECTNESS] FAILED ({result.num_passed_trials}/{result.num_correct_trials} trials)")
            if result.max_diff > 0:
                lines.append(f"  max_diff: {result.max_diff:.6f}")
            if result.error_msg:
                lines.append(f"  error: {result.error_msg[:500]}")
            return "\n".join(lines)
        lines.append(f"[CORRECTNESS] PASS ({result.num_passed_trials}/{result.num_correct_trials} trials)")
        if result.speedup > 0:
            lines.append(f"[PERFORMANCE] runtime={result.runtime_ms:.4f}ms  ref={result.ref_runtime_ms:.4f}ms  speedup={result.speedup:.2f}x")
        else:
            lines.append("[PERFORMANCE] not measured")
        return "\n".join(lines)

    def cleanup(self):
        """Remove tempfiles and sys.modules entries."""
        from .compiler import cleanup_compiled_module
        for module_name, tempfile_path in self._loaded_modules:
            cleanup_compiled_module(module_name, tempfile_path)
        self._loaded_modules.clear()


# ---------------------------------------------------------------------------
# Internal: code loading
# ---------------------------------------------------------------------------

def _extract_functions(
    ref_code: str,
    new_module,
    problem_format: str,
) -> Tuple[Callable, Callable, Callable]:
    """Extract ref_fn, new_fn, input_gen from ref_code and compiled new_module.

    *ref_code* is loaded via ``exec()`` (pure PyTorch — no Triton decorators).
    *new_module* was already loaded via importlib (handles @triton.jit correctly).
    """
    ref_ns: Dict[str, Any] = {}
    exec(ref_code, ref_ns)

    # Find reference function
    ref_fn = ref_ns.get("ref_fn")
    if ref_fn is None:
        # Look for any callable that looks like a kernel function
        for name, obj in ref_ns.items():
            if callable(obj) and not name.startswith("_") and not isinstance(obj, type):
                if name not in ("gen_inputs", "get_inputs", "get_init_inputs"):
                    ref_fn = obj
                    break
    if ref_fn is None:
        raise ValueError("Could not find ref_fn in ref_code")

    # Find input generator
    gen_inputs = ref_ns.get("gen_inputs") or ref_ns.get("get_inputs")
    if gen_inputs is None:
        raise ValueError("Could not find gen_inputs or get_inputs in ref_code")

    # Find new function from loaded module
    new_fn = getattr(new_module, "new_fn", None)
    if new_fn is None:
        # Look for any callable
        for name in dir(new_module):
            obj = getattr(new_module, name)
            if callable(obj) and not name.startswith("_") and not isinstance(obj, type):
                if name != "ref_fn":
                    new_fn = obj
                    break
    if new_fn is None:
        raise ValueError(f"Could not find new_fn in compiled module. Available: {dir(new_module)}")

    return ref_fn, new_fn, gen_inputs


def _to_device_fn(fn: Callable, device: torch.device, dtype: torch.dtype):
    def _wrapped(*args):
        moved = []
        for a in args:
            if isinstance(a, torch.Tensor):
                moved.append(a.to(device=device, dtype=dtype))
            else:
                moved.append(a)
        return fn(*moved)
    return _wrapped


def _str_to_dtype(s: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}.get(
        s, torch.float32
    )


def _truncate(s: str, max_len: int) -> str:
    return s if len(s) <= max_len else s[: max_len - 3] + "..."
