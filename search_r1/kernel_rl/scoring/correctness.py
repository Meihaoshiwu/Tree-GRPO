"""Numerical correctness checking against PyTorch reference.

Adapted from KernelBench ``run_and_check_correctness``.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch


@dataclass
class CorrectnessResult:
    passed: bool
    trials_total: int = 1
    trials_passed: int = 0
    max_diff: float = 0.0
    avg_diff: float = 0.0
    error_msg: str = ""
    error_type: str = ""  # "shape", "value", "runtime", ""
    outputs: List[Dict[str, Any]] = field(default_factory=list)


def check_correctness(
    ref_fn: Callable,
    new_fn: Callable,
    input_gen: Callable[[], List[torch.Tensor]],
    num_trials: int = 3,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
    verbose: bool = False,
) -> CorrectnessResult:
    """Check numerical correctness of *new_fn* against *ref_fn* over multiple trials.

    Args:
        ref_fn: Reference PyTorch function ``(*inputs) -> Tensor``.
        new_fn: Generated kernel function with the same signature.
        input_gen: Zero-arg callable returning ``list[Tensor]``.
        num_trials: How many different random inputs to test.
        rtol, atol: Tolerances passed to ``torch.allclose``.
        device: CUDA device.
        dtype: Compute dtype.
        seed: Base random seed (each trial derives its own seed).
        verbose: Print per-trial results.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    trial_seeds = [torch.randint(0, 2**31 - 1, (1,)).item() for _ in range(num_trials)]

    pass_count = 0
    max_diff = 0.0
    avg_diff = 0.0
    error_msg = ""
    error_type = ""
    outputs: List[Dict[str, Any]] = []

    with torch.no_grad():
        for trial in range(num_trials):
            trial_seed = trial_seeds[trial]
            torch.manual_seed(trial_seed)

            try:
                inputs = input_gen()
                inputs = [
                    x.to(device=device, dtype=dtype) if isinstance(x, torch.Tensor) else x
                    for x in inputs
                ]

                torch.manual_seed(trial_seed)
                ref_output = ref_fn(*inputs)

                torch.manual_seed(trial_seed)
                new_output = new_fn(*inputs)

                torch.cuda.synchronize(device=device)

                # Shape check
                if ref_output.shape != new_output.shape:
                    error_msg = (
                        f"Shape mismatch: ref {ref_output.shape} vs new {new_output.shape}"
                    )
                    error_type = "shape"
                    if verbose:
                        print(f"[FAIL] trial {trial}: {error_msg}")
                    return CorrectnessResult(
                        passed=False,
                        trials_total=num_trials,
                        trials_passed=pass_count,
                        error_msg=error_msg,
                        error_type=error_type,
                        outputs=outputs,
                    )

                # Value check
                if torch.allclose(ref_output, new_output, rtol=rtol, atol=atol):
                    pass_count += 1
                    if verbose:
                        print(f"[PASS] trial {trial}")
                else:
                    diff = (ref_output - new_output).abs()
                    max_diff = max(max_diff, diff.max().item())
                    avg_diff = max(avg_diff, diff.mean().item())
                    if verbose:
                        print(
                            f"[FAIL] trial {trial}: max_diff={max_diff:.6f}, avg_diff={avg_diff:.6f}"
                        )

                outputs.append({
                    "trial": trial,
                    "seed": trial_seed,
                    "passed": True,
                    "max_diff": max_diff,
                    "avg_diff": avg_diff,
                })

            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                error_type = "runtime"
                if verbose:
                    print(f"[ERROR] trial {trial}: {error_msg}")
                    traceback.print_exc()
                return CorrectnessResult(
                    passed=False,
                    trials_total=num_trials,
                    trials_passed=pass_count,
                    max_diff=max_diff,
                    avg_diff=avg_diff,
                    error_msg=error_msg,
                    error_type=error_type,
                    outputs=outputs,
                )

    return CorrectnessResult(
        passed=(pass_count == num_trials),
        trials_total=num_trials,
        trials_passed=pass_count,
        max_diff=max_diff,
        avg_diff=avg_diff,
        error_msg=error_msg,
        error_type=error_type if not (pass_count == num_trials) else "",
        outputs=outputs,
    )
