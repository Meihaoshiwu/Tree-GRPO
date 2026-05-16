"""Performance measurement for Triton kernels.

Adapted from KernelBench ``timing.py`` — CUDA-event wall-clock timing with L2 cache
clearing between each trial to measure cold-cache performance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch


@dataclass
class PerfResult:
    runtime_ms: float = 0.0          # mean across trials
    runtime_std_ms: float = 0.0      # std across trials
    runtime_min_ms: float = 0.0
    runtime_max_ms: float = 0.0
    ref_runtime_ms: float = 0.0      # PyTorch baseline mean
    ref_runtime_std_ms: float = 0.0
    speedup: float = 1.0             # ref_runtime / runtime
    trials: int = 0
    hardware: str = ""
    device_id: int = 0
    raw_times_ms: List[float] = field(default_factory=list)


def measure_performance(
    kernel_fn: Callable,
    args: List[Any],
    ref_kernel_fn: Optional[Callable] = None,
    num_warmup: int = 5,
    num_trials: int = 20,
    discard_first: int = 2,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> PerfResult:
    """Measure wall-clock execution time of *kernel_fn* using CUDA events.

    Args:
        kernel_fn: The custom kernel forward pass ``(*args) -> Tensor``.
        args: Inputs to the kernel.
        ref_kernel_fn: Optional PyTorch reference (if given, speedup is computed).
        num_warmup: Warmup iterations before timing.
        num_trials: Number of timing trials.
        discard_first: Discard first N trials (GPU warmup effects).
        device: CUDA device.
        verbose: Print per-trial timings.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    device_idx = device.index if device.index is not None else 0
    hardware = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu"

    with torch.cuda.device(device):
        # Measure custom kernel -------------------------------------------------
        _warmup(kernel_fn, args, num_warmup, device)
        torch.cuda.empty_cache()
        custom_times = _time_trials(kernel_fn, args, num_trials, discard_first, device, verbose)
        custom_stats = _compute_stats(custom_times)

        result = PerfResult(
            runtime_ms=custom_stats["mean_ms"],
            runtime_std_ms=custom_stats["std_ms"],
            runtime_min_ms=custom_stats["min_ms"],
            runtime_max_ms=custom_stats["max_ms"],
            trials=len(custom_times),
            hardware=hardware,
            device_id=device_idx,
            raw_times_ms=custom_times,
        )

        # Measure reference if provided -----------------------------------------
        if ref_kernel_fn is not None:
            _warmup(ref_kernel_fn, args, num_warmup, device)
            torch.cuda.empty_cache()
            ref_times = _time_trials(ref_kernel_fn, args, num_trials, discard_first, device, verbose)
            ref_stats = _compute_stats(ref_times)
            result.ref_runtime_ms = ref_stats["mean_ms"]
            result.ref_runtime_std_ms = ref_stats["std_ms"]
            if result.runtime_ms > 0:
                result.speedup = ref_stats["mean_ms"] / result.runtime_ms
        return result


def clear_l2_cache(device: torch.device) -> None:
    """Thrash L2 cache with a large dummy tensor."""
    try:
        dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device=device)
        dummy.fill_(42)
        del dummy
    except Exception:
        pass  # never let cache-clearing crash the measurement


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _warmup(kernel_fn: Callable, args: List[Any], num_warmup: int, device: torch.device) -> None:
    for _ in range(num_warmup):
        kernel_fn(*args)
        torch.cuda.synchronize(device=device)


def _time_trials(
    kernel_fn: Callable,
    args: List[Any],
    num_trials: int,
    discard_first: int,
    device: torch.device,
    verbose: bool = False,
) -> List[float]:
    times: List[float] = []
    for trial in range(num_trials + discard_first):
        torch.cuda.synchronize(device=device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        clear_l2_cache(device=device)

        start_event.record()
        _ = kernel_fn(*args)
        end_event.record()
        torch.cuda.synchronize(device=device)

        elapsed_ms = start_event.elapsed_time(end_event)
        if trial >= discard_first:
            times.append(elapsed_ms)
            if verbose:
                logical = trial - discard_first + 1
                print(f"  Trial {logical}: {elapsed_ms:.4f} ms")
    return times


def _compute_stats(times: List[float]) -> Dict[str, float]:
    if not times:
        return {"mean_ms": 0.0, "std_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    t = torch.tensor(times, dtype=torch.float64)
    return {
        "mean_ms": float(t.mean().item()),
        "std_ms": float(t.std().item()) if len(times) > 1 else 0.0,
        "min_ms": float(t.min().item()),
        "max_ms": float(t.max().item()),
    }
