"""SM-level kernel profiling (optional, requires NVIDIA Nsight Compute).

Adapted from KernelBench ``profile.py``.  Provides hardware-level metrics:
  - ``sm__cycles_active.avg``   — average active cycles per SM
  - ``sm__cycles_elapsed.sum``  — total SM cycles elapsed
  - ``gpu__time_duration.sum``  — total GPU time (ns)
  - ``smsp__inst_executed``     — instructions executed per SM partition

Requirements:
  - NVIDIA GPU only
  - ``nsight-python`` package installed
  - ``ncu`` CLI available in PATH (usually requires sudo for hardware counters)

If prerequisites are not met, ``profile_kernel`` returns empty results gracefully
rather than crashing the caller.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch

# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------

_NSIGHT_AVAILABLE: Optional[bool] = None


def _check_nsight() -> bool:
    global _NSIGHT_AVAILABLE
    if _NSIGHT_AVAILABLE is None:
        try:
            import nsight  # noqa: F401
            _NSIGHT_AVAILABLE = True
        except ImportError:
            _NSIGHT_AVAILABLE = False
    return _NSIGHT_AVAILABLE


def _check_ncu() -> bool:
    return subprocess.run(["which", "ncu"], capture_output=True).returncode == 0


@dataclass
class ProfileResult:
    available: bool = False
    metrics: Dict[str, float] = field(default_factory=dict)
    error: str = ""


def profile_kernel(
    kernel_fn: Callable[[], Any],
    metrics: Optional[List[str]] = None,
    num_trials: int = 1,
    verbose: bool = False,
) -> ProfileResult:
    """Profile a kernel closure with Nsight hardware counters.

    Args:
        kernel_fn: Zero-arg closure that runs the kernel forward pass.
        metrics: Nsight metric names.  Defaults to ``['gpu__time_duration.sum']``.
        num_trials: Number of profiling runs.
        verbose: Print progress.

    Returns:
        ProfileResult with ``available=True`` and ``metrics`` populated on success,
        or ``available=False`` with ``error`` set if nsight is not available.
    """
    if not _check_nsight():
        return ProfileResult(
            available=False,
            error="nsight-python not installed.  pip install nsight-python",
        )
    if not _check_ncu():
        return ProfileResult(
            available=False,
            error="ncu CLI not found in PATH.  Install NVIDIA Nsight Compute.",
        )
    if not torch.cuda.is_available():
        return ProfileResult(available=False, error="CUDA not available.")

    import nsight

    if metrics is None:
        metrics = ["gpu__time_duration.sum"]
    elif isinstance(metrics, str):
        metrics = [metrics]

    @nsight.analyze.kernel(
        metrics=metrics,
        runs=num_trials,
        configs=[(0,)],
        combine_kernel_metrics=lambda a, b: (0 if a is None else a) + (0 if b is None else b),
    )
    def _profiled(_):
        with nsight.annotate("kernel"):
            return kernel_fn()

    try:
        result = _profiled()
        df = result.to_dataframe() if result else None
        if df is None or df.empty:
            return ProfileResult(available=True, metrics={m: float("nan") for m in metrics})

        metric_col = next((c for c in df.columns if c.lower() == "metric"), None)
        value_col = next((c for c in df.columns if "value" in c.lower()), None)

        if not metric_col or not value_col:
            return ProfileResult(available=True, metrics={m: float("nan") for m in metrics})

        metric_dict = {row[metric_col]: float(row[value_col]) for _, row in df.iterrows()}
        return ProfileResult(available=True, metrics={m: metric_dict.get(m, float("nan")) for m in metrics})

    except Exception as exc:
        if verbose:
            print(f"[Profiler] Error: {exc}")
        return ProfileResult(available=True, error=str(exc), metrics={})


# ---------------------------------------------------------------------------
# Convenience: known-useful metric sets
# ---------------------------------------------------------------------------

PERF_METRICS = [
    "gpu__time_duration.sum",
    "sm__cycles_elapsed.sum",
]

OCCUPANCY_METRICS = [
    "sm__cycles_active.avg",
    "sm__cycles_elapsed.sum",
    "sm__warps_active.avg",
    "sm__maximum_warps_per_cycle.avg",
]

MEMORY_METRICS = [
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__bytes_read.sum",
    "lts__bytes_write.sum",
]

COMPUTE_METRICS = [
    "smsp__inst_executed_pipe_tensor_op_hmma.sum",  # Tensor Core
    "smsp__inst_executed_pipe_fp64.sum",             # FP64
    "smsp__inst_executed_pipe_fp32.sum",             # FP32
    "smsp__inst_executed_pipe_alu.sum",               # INT
]
