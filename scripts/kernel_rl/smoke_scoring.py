"""Smoke test for the kernel scoring (evaluator) module.

Validates compile → correctness → performance for a trivial Triton kernel.
Does NOT require Ray — runs the evaluator directly in-process.

Usage:
    python scripts/kernel_rl/smoke_scoring.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from search_r1.kernel_rl.scoring.evaluator import KernelEvaluator, EvalConfig

# A reference PyTorch function + input generator (simple format)
REF_CODE = """
import torch
import torch.nn.functional as F

def ref_fn(a, b):
    return a + b

def gen_inputs():
    return [torch.randn(4096, dtype=torch.float32), torch.randn(4096, dtype=torch.float32)]
"""

# A correct Triton kernel (vector add)
CORRECT_NEW_CODE = """
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

def new_fn(a, b):
    assert a.shape == b.shape
    n = a.numel()
    out = torch.empty_like(a)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    add_kernel[grid](a, b, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out
"""

# Truly broken: syntax error
SYNTAX_ERROR_CODE = """
def new_fn(a, b
    return a + b
"""


def test_compile_pass():
    """Correct kernel should compile."""
    evaluator = KernelEvaluator(EvalConfig(measure_perf=False))
    result = evaluator.evaluate(
        ref_code=REF_CODE,
        new_code=CORRECT_NEW_CODE,
        problem_format="simple",
    )
    assert result.compiled, f"Expected compiled=True, got {result.compiled}: {result.error_msg[:200]}"
    print("  [PASS] compile_pass")


def test_correctness():
    """Correct kernel should pass correctness check."""
    evaluator = KernelEvaluator(EvalConfig(measure_perf=False, num_correct_trials=3))
    result = evaluator.evaluate(
        ref_code=REF_CODE,
        new_code=CORRECT_NEW_CODE,
        problem_format="simple",
    )
    assert result.compiled, f"Compilation failed: {result.error_msg[:200]}"
    assert result.correctness, f"Correctness failed: {result.error_msg[:200]}"
    print(f"  [PASS] correctness (trials={result.num_passed_trials}/{result.num_correct_trials})")


def test_performance():
    """Should measure performance and compute speedup."""
    evaluator = KernelEvaluator(EvalConfig(
        measure_perf=True,
        num_warmup=5,
        num_perf_trials=10,
        num_correct_trials=1,
    ))
    result = evaluator.evaluate(
        ref_code=REF_CODE,
        new_code=CORRECT_NEW_CODE,
        problem_format="simple",
    )
    assert result.compiled, f"Compilation failed: {result.error_msg[:200]}"
    assert result.correctness, f"Correctness failed: {result.error_msg[:200]}"
    assert result.runtime_ms > 0, f"Expected positive runtime, got {result.runtime_ms}"
    assert result.speedup > 0, f"Expected positive speedup, got {result.speedup}"
    print(f"  [PASS] performance (runtime={result.runtime_ms:.4f}ms, ref={result.ref_runtime_ms:.4f}ms, speedup={result.speedup:.2f}x)")


def test_compile_failure():
    """Broken kernel should report compile error."""
    evaluator = KernelEvaluator(EvalConfig(measure_perf=False))
    result = evaluator.evaluate(
        ref_code=REF_CODE,
        new_code=SYNTAX_ERROR_CODE,
        problem_format="simple",
    )
    assert not result.compiled, f"Expected compile failure, got {result.status}"
    print(f"  [PASS] compile_failure (status={result.status}, error_type={result.error_type})")


def test_reward_computation():
    """Reward computation should produce valid scores."""
    evaluator = KernelEvaluator(EvalConfig(measure_perf=False))
    result = evaluator.evaluate(
        ref_code=REF_CODE,
        new_code=CORRECT_NEW_CODE,
        problem_format="simple",
    )
    rewards = evaluator.compute_rewards(result, target_speedup=2.0)
    assert "design" in rewards and "code" in rewards and "predict" in rewards
    assert rewards["code"] >= 0.4, f"Expected code reward >= 0.4 for correct kernel, got {rewards['code']}"
    print(f"  [PASS] rewards: {rewards}")


def test_feedback():
    """Feedback should be human-readable."""
    evaluator = KernelEvaluator(EvalConfig(measure_perf=False))
    result = evaluator.evaluate(
        ref_code=REF_CODE,
        new_code=CORRECT_NEW_CODE,
        problem_format="simple",
    )
    feedback = evaluator.build_feedback(result)
    assert "[COMPILED] OK" in feedback
    assert "[CORRECTNESS] PASS" in feedback
    print(f"  [PASS] feedback:\n{feedback}")


def main():
    print("=" * 60)
    print("Smoke test: Kernel Scoring Module")
    print("=" * 60)
    print(f"CUDA available: {__import__('torch').cuda.is_available()}")
    print()

    tests = [
        ("compile_pass", test_compile_pass),
        ("correctness", test_correctness),
        ("performance", test_performance),
        ("compile_failure", test_compile_failure),
        ("reward_computation", test_reward_computation),
        ("feedback", test_feedback),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1

    print()
    print(f"Results: {passed}/{passed+failed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
