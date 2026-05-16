"""
Generate SFT data from KernelBook + auto-generated design/predict.

Pipeline:
  1. Load KernelBook (18k PyTorch->Triton pairs)
  2. Analyze each kernel: op type, memory pattern, block strategy
  3. Auto-generate <design> (structured, truthful analysis)
  4. Auto-generate <predict> (performance estimates)
  5. Filter: skip trivially short code, keep diverse kernel types
  6. Output JSONL in three-section format

Usage:
    python scripts/kernel_rl/gen_sft_data_v2.py --num_samples 800 --output data/kernel_rl/sft_train_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ──────────────────────────────────────────────────────────────────────
# Kernel analyzer — extracts structural facts from Triton code
# ──────────────────────────────────────────────────────────────────────

@dataclass
class KernelProfile:
    """Structural facts extracted from a Triton kernel."""
    op_type: str = "unknown"           # elementwise, reduction, matmul, normalization, attention, other
    has_tiling: bool = False           # uses multiple program_id dimensions and blocks
    has_shared_memory: bool = False    # allocates shared memory
    has_atomic: bool = False           # uses atomic operations
    has_mask: bool = False             # uses boundary mask
    block_sizes: List[int] = field(default_factory=list)  # BLOCK_SIZE constants found
    num_program_dims: int = 1          # 1D, 2D, or 3D grid
    memory_pattern: str = "unknown"    # coalesced, strided, random
    uses_tensor_core: bool = False     # uses tl.dot
    uses_special_func: bool = False    # uses exp, erf, sigmoid, etc.
    approx_flops_per_element: float = 0.0  # rough FLOPs estimate
    approx_bytes_per_element: float = 0.0  # rough memory traffic estimate


def analyze_kernel(triton_code: str) -> KernelProfile:
    """Extract structural facts from a Triton kernel by pattern matching."""
    p = KernelProfile()

    # ── Operation type ──────────────────────────────────────────────
    if "tl.dot" in triton_code:
        p.op_type = "matmul"
        p.uses_tensor_core = True
    elif re.search(r'tl\.sum\(|tl\.max\(|tl\.min\(|tl\.argmax\(|tl\.argmin\(', triton_code):
        p.op_type = "reduction"
    elif re.search(r'mean\(|var\(|rms|norm|layernorm|batchnorm', triton_code, re.IGNORECASE):
        p.op_type = "normalization"
    elif re.search(r'softmax|attention|sdpa|flash', triton_code, re.IGNORECASE):
        p.op_type = "attention"
    else:
        p.op_type = "elementwise"

    # ── Tiling / grid ───────────────────────────────────────────────
    prog_ids = re.findall(r'tl\.program_id\((\d)\)', triton_code)
    p.num_program_dims = max([int(x) for x in prog_ids]) + 1 if prog_ids else 1
    p.has_tiling = p.num_program_dims >= 2 or "for" in triton_code and "BLOCK" in triton_code

    # ── Shared memory ───────────────────────────────────────────────
    p.has_shared_memory = bool(re.search(
        r'tl\.zeros\(|tl\.alloc\(|shared|smem|SRAM', triton_code, re.IGNORECASE
    ))

    # ── Atomic ops ──────────────────────────────────────────────────
    p.has_atomic = bool(re.search(r'tl\.atomic_', triton_code))

    # ── Boundary mask ───────────────────────────────────────────────
    p.has_mask = bool(re.search(r'mask\s*=', triton_code))

    # ── Block sizes ─────────────────────────────────────────────────
    p.block_sizes = [int(x) for x in re.findall(r'BLOCK_SIZE[:\s]*=\s*(\d+)', triton_code)]
    if not p.block_sizes:
        p.block_sizes = [int(x) for x in re.findall(r'BLOCK[:\s]*=\s*(\d+)', triton_code)]

    # ── Memory pattern ──────────────────────────────────────────────
    if "stride" in triton_code.lower() or re.search(r'\[\s*None\s*,\s*:\s*\]', triton_code):
        p.memory_pattern = "strided"
    elif re.search(r'offsets|pid\s*\*\s*BLOCK|\+ offset', triton_code):
        p.memory_pattern = "coalesced"
    else:
        p.memory_pattern = "coalesced"  # default for simple kernels

    # ── FLOPs & memory estimates ────────────────────────────────────
    if p.op_type == "elementwise":
        p.approx_flops_per_element = 2.0  # one arithmetic op
        p.approx_bytes_per_element = 8.0  # 1 read + 1 write = 8 bytes (fp32)
    elif p.op_type == "reduction":
        p.approx_flops_per_element = 3.0  # load + compute + partial sum
        p.approx_bytes_per_element = 4.0  # 1 read + tiny write
    elif p.op_type == "matmul":
        p.approx_flops_per_element = 2.0  # per output element: 2*K operations
        p.approx_bytes_per_element = 12.0  # A + B + C
    elif p.op_type == "normalization":
        p.approx_flops_per_element = 5.0  # mean + var + normalize
        p.approx_bytes_per_element = 8.0  # read + write
    else:
        p.approx_flops_per_element = 2.0
        p.approx_bytes_per_element = 8.0

    # ── Special functions ───────────────────────────────────────────
    p.uses_special_func = bool(re.search(
        r'tl\.exp|tl\.log|tl\.sqrt|tl\.sigmoid|tl\.tanh|tl\.math\.erf|tl\.rsqrt',
        triton_code
    ))

    return p


# ──────────────────────────────────────────────────────────────────────
# Design generator — produces structured, truthful design text
# ──────────────────────────────────────────────────────────────────────

def generate_design(profile: KernelProfile, python_code: str, triton_code: str) -> str:
    """Generate a structured <design> section based on kernel analysis."""
    lines = []

    # Op-type classification
    op_descriptions = {
        "elementwise": "This is an element-wise operation. Each output element depends only on the corresponding input element(s), making it embarrassingly parallel. All performance comes from memory bandwidth utilization.",
        "reduction": "This is a reduction operation. Elements are aggregated (sum/max/argmax) across one or more dimensions. The key challenge is efficiently combining partial results from multiple threads without excessive synchronization.",
        "matmul": "This is a matrix multiplication (GEMM) operation. It is compute-bound. Performance depends on tiling strategy, shared memory usage, and Tensor Core utilization via tl.dot().",
        "normalization": "This is a normalization operation (LayerNorm/RMSNorm/BatchNorm). It requires computing statistics (mean/variance) across a dimension, then normalizing. The reduction step for statistics is the main bottleneck.",
        "attention": "This is an attention-related operation. Attention is memory-bound for short sequences and compute-bound for long sequences. The QK^T matmul and softmax are the critical path.",
    }
    lines.append(op_descriptions.get(profile.op_type, op_descriptions["elementwise"]))

    # Memory access pattern
    if profile.memory_pattern == "coalesced":
        lines.append(
            "Memory access is coalesced: adjacent threads access adjacent memory addresses, "
            "maximizing L1 cache line utilization and effective memory bandwidth."
        )
    elif profile.memory_pattern == "strided":
        lines.append(
            "Memory access is strided. Non-unit stride reduces effective memory bandwidth "
            "due to cache line wastage. Consider transposition or padding to improve coalescing."
        )

    # Block size strategy
    if profile.block_sizes:
        bs = profile.block_sizes[0]
        if profile.op_type in ("elementwise",):
            lines.append(
                f"Block size is {bs}. For element-wise kernels, larger blocks (512-2048) "
                f"reduce launch overhead with minimal register pressure increase. "
                f"The optimal size is limited by GPU occupancy: too many threads per block "
                f"reduces the number of concurrent blocks per SM."
            )
        elif profile.op_type == "matmul":
            lines.append(
                f"Block size is {bs}. For matrix multiplication, block sizes of 64-256 "
                f"balance shared memory usage against occupancy. Smaller blocks allow more "
                f"concurrent warps per SM; larger blocks improve data reuse from shared memory."
            )
        elif profile.op_type == "reduction":
            lines.append(
                f"Block size is {bs}. For reductions, block size determines the number of "
                f"elements reduced per program. Larger blocks reduce the number of programs "
                f"but increase the reduction tree depth within each program."
            )

    # Tiling strategy
    if profile.has_tiling and profile.num_program_dims >= 2:
        lines.append(
            f"This kernel uses a {profile.num_program_dims}D launch grid with tiling. "
            "Multi-dimensional tiling improves data locality: each program instance "
            "operates on a small tile that fits in registers and L1 cache, reducing "
            "global memory traffic through reuse."
        )

    # Shared memory
    if profile.has_shared_memory:
        lines.append(
            "Shared memory (SRAM) is used to cache tiles of input data. "
            "This is essential for matmul and convolution patterns where each "
            "input element is reused multiple times. Shared memory bandwidth "
            "is ~10-20x higher than global memory (HBM)."
        )

    # Tensor cores
    if profile.uses_tensor_core:
        lines.append(
            "The kernel uses tl.dot() which maps to Tensor Cores on NVIDIA GPUs. "
            "Tensor Cores provide ~4-8x throughput over FP32 CUDA cores for matrix "
            "multiply-accumulate. On RTX 4090 (Ada Lovelace), Tensor Cores deliver "
            "up to 330 TFLOPS (FP16) vs 82.6 TFLOPS (FP32 CUDA)."
        )

    # Atomic ops
    if profile.has_atomic:
        lines.append(
            "Atomic operations are used for cross-program reduction. "
            "Atomic contention becomes a bottleneck when many programs write "
            "to the same memory location. For large reductions, consider "
            "a two-level reduction (local → global) to reduce contention."
        )

    # Computational intensity (compute vs memory bound)
    if profile.approx_bytes_per_element > 0:
        ci = profile.approx_flops_per_element / profile.approx_bytes_per_element
        if ci > 20:
            lines.append(
                f"Computational intensity is high ({ci:.1f} FLOPs/byte). "
                "This kernel is compute-bound on modern GPUs. Focus on maximizing "
                "FPU/Tensor Core utilization rather than memory bandwidth."
            )
        elif ci < 5:
            lines.append(
                f"Computational intensity is low ({ci:.1f} FLOPs/byte). "
                "This kernel is memory-bound. The optimization priority is: "
                "(1) coalesce memory accesses, (2) fuse with adjacent operations "
                "to avoid intermediate writes, (3) use vectorized loads/stores."
            )
        else:
            lines.append(
                f"Computational intensity is moderate ({ci:.1f} FLOPs/byte). "
                "Both compute and memory throughput matter. Profile to identify "
                "the actual bottleneck on target hardware."
            )

    # Special functions note
    if profile.uses_special_func:
        lines.append(
            "The kernel uses transcendental functions (exp/log/erf/sigmoid). "
            "These execute on the GPU's Special Function Unit (SFU), which has "
            "lower throughput than FP32 ALUs. For compute-bound kernels, "
            "consider polynomial approximations for erf/sigmoid if accuracy allows."
        )

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Predict generator — estimates performance characteristics
# ──────────────────────────────────────────────────────────────────────

def generate_predict(profile: KernelProfile, python_code: str, triton_code: str) -> str:
    """Generate a <predict> section with performance estimates."""
    lines = []

    # Memory vs compute bound classification
    lines.append("Performance classification:")

    if profile.op_type == "matmul":
        lines.append("- Likely compute-bound for typical matrix sizes (>256x256).")
        lines.append("- Memory-bound for very small matrices where launch overhead dominates.")
    elif profile.op_type in ("elementwise",):
        lines.append("- Memory-bound. The arithmetic is trivial; all time is spent moving data.")
        lines.append("- Speedup vs. PyTorch eager: ~1.0x standalone. Speedup when fused with adjacent op: ~1.5-2.0x.")
    elif profile.op_type == "reduction":
        lines.append("- Reduction-bound. Synchronization between threads limits throughput.")
        lines.append("- Speedup vs. PyTorch: ~0.8-1.2x for simple reductions. Main value is in fusion.")
    elif profile.op_type == "normalization":
        lines.append("- Memory-bound for typical feature dimensions. Reduction step adds ~10-20% overhead.")
        lines.append("- Speedup vs. PyTorch: ~1.2-1.5x with fusion (the norm is always fused in practice).")
    elif profile.op_type == "attention":
        lines.append("- Memory-bound for short sequences (<2048), compute-bound for long sequences (>4096).")
        lines.append("- Naive attention uses O(N^2) memory. Flash Attention reduces this to O(N).")

    # Block size impact
    if profile.block_sizes:
        bs = profile.block_sizes[0]
        lines.append(f"\nExpected runtime with BLOCK_SIZE={bs}:")
        if profile.op_type == "matmul":
            lines.append(f"- Shared memory used: ~{bs * bs * 4 // 1024} KB per tile.")
            lines.append(f"- Occupancy: depends on register pressure. Typical SM occupancy ~25-50% for tiled matmul.")
        else:
            lines.append(f"- Launch grid: ceil(N / {bs}) program instances.")
            lines.append(f"- GPU occupancy: determined by block count vs. SM count. {bs} threads/block is typical.")

    # Hardware-specific estimate
    lines.append("\nOn RTX 4090 (Ada Lovelace, ~1000 GB/s HBM bandwidth, 82.6 TFLOPS FP32):")
    lines.append(f"- Estimated memory time: dominated by {profile.approx_bytes_per_element:.0f} bytes/element at ~1 TB/s.")
    lines.append("- Kernel launch overhead: ~5-10 microseconds per kernel launch.")
    lines.append("- Primary bottleneck: memory bandwidth (HBM) for element-wise/norm; compute for matmul.")

    lines.append("\nConfidence: medium (estimates based on static code analysis, not profiling).")
    lines.append("Actual performance depends on input sizes, GPU cache state, and concurrent kernel execution.")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Prompt builder
# ──────────────────────────────────────────────────────────────────────

FORMAT_INSTRUCTIONS = """Respond with exactly three sections in this order:
<design>
Explain the optimization idea, correctness constraints, and memory / launch tradeoffs.
</design>
<code>
Provide the Triton kernel code or the edited code block.
</code>
<predict>
Estimate performance, expected bottlenecks, and confidence in the estimate.
</predict>"""


def build_prompt(python_code: str, entry_point: str) -> str:
    """Build a task-style prompt from the reference PyTorch code."""
    # Truncate python_code to reasonable length
    py_preview = python_code[:2000] if len(python_code) > 2000 else python_code
    if len(python_code) > 2000:
        py_preview += "\n# ... (truncated)"

    return (
        "You are improving a Triton operator implementation.\n"
        "Read the task carefully, preserve correctness, and optimize for real GPU performance.\n\n"
        "Task specification:\n"
        f'{{"name": "{entry_point}", "task": "Convert the following PyTorch module to an optimized Triton kernel"}}\n\n'
        "Reference Python implementation:\n"
        "```python\n"
        f"{py_preview}\n"
        "```\n\n"
        + FORMAT_INSTRUCTIONS
    )


# ──────────────────────────────────────────────────────────────────────
# Quality filters
# ──────────────────────────────────────────────────────────────────────

def is_quality_example(python_code: str, triton_code: str, profile: KernelProfile) -> bool:
    """Filter out low-quality or unsuitable examples."""
    # Skip trivially short code
    if len(triton_code) < 100:
        return False
    if len(python_code) < 50:
        return False

    # Skip examples with only triton imports but no kernel
    if "triton.jit" not in triton_code and "@triton" not in triton_code:
        # Some KernelBook examples use torch inductor's triton-rewritten format
        # without @triton.jit — still valid triton code
        if "import triton" not in triton_code:
            return False

    # Skip kernel code that's just a stub
    code_lines = [l for l in triton_code.split("\n") if l.strip() and not l.strip().startswith("#")]
    if len(code_lines) < 8:
        return False

    # Skip if python_code is just a wrapper (too simple)
    py_lines = [l for l in python_code.split("\n") if l.strip() and not l.strip().startswith("#")]
    if len(py_lines) < 3:
        return False

    return True


def filter_by_diversity(examples: List[Dict], target_count: int) -> List[Dict]:
    """Ensure diverse operation types in the final selection."""
    op_buckets: Dict[str, List[Dict]] = {}
    for ex in examples:
        op = ex.get("_op_type", "unknown")
        op_buckets.setdefault(op, []).append(ex)

    # Target proportions
    proportions = {
        "elementwise": 0.25,
        "reduction": 0.15,
        "matmul": 0.25,
        "normalization": 0.15,
        "attention": 0.10,
        "unknown": 0.10,
    }

    selected = []
    for op_type, prop in proportions.items():
        bucket = op_buckets.get(op_type, [])
        n = int(target_count * prop)
        selected.extend(random.sample(bucket, min(n, len(bucket))))

    # If we're short, fill with remaining
    if len(selected) < target_count:
        remaining = [ex for ex in examples if ex not in selected]
        needed = target_count - len(selected)
        selected.extend(random.sample(remaining, min(needed, len(remaining))))

    random.shuffle(selected)
    return selected[:target_count]


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=800)
    parser.add_argument("--output", type=str,
                        default="data/kernel_rl/sft_train_v2.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_filter", action="store_true",
                        help="Skip quality filtering")
    args = parser.parse_args()

    random.seed(args.seed)
    output_path = os.path.join(
        os.path.dirname(__file__), "..", "..", args.output
    )
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Loading KernelBook dataset...")
    from datasets import load_dataset
    ds = load_dataset("GPUMODE/KernelBook", split="train")
    print(f"Loaded {len(ds)} examples")

    examples = []
    op_counter = Counter()
    skipped = 0

    print(f"Processing {len(ds)} examples (target: {args.num_samples})...")
    for i, row in enumerate(ds):
        python_code = row["python_code"]
        triton_code = row["triton_code"]
        entry_point = row.get("entry_point", row.get("module_name", f"kernel_{i}"))

        if not isinstance(python_code, str) or not isinstance(triton_code, str):
            skipped += 1
            continue

        profile = analyze_kernel(triton_code)

        if not args.no_filter and not is_quality_example(python_code, triton_code, profile):
            skipped += 1
            continue

        design = generate_design(profile, python_code, triton_code)
        predict = generate_predict(profile, python_code, triton_code)
        prompt = build_prompt(python_code, entry_point)

        # Wrap triton_code in <code> tags
        completion = f"<design>\n{design}\n</design>\n<code>\n{triton_code}\n</code>\n<predict>\n{predict}\n</predict>"

        examples.append({
            "prompt": prompt,
            "completion": completion,
            "_op_type": profile.op_type,
            "_entry_point": entry_point,
        })
        op_counter[profile.op_type] += 1

        if (i + 1) % 2000 == 0:
            print(f"  Processed {i+1} / {len(ds)}... "
                  f"kept={len(examples)}, skipped={skipped}")

    print(f"\nKept {len(examples)} examples (skipped {skipped})")
    print(f"Op types: {dict(op_counter)}")

    # Filter for diversity
    if len(examples) > args.num_samples:
        examples = filter_by_diversity(examples, args.num_samples)
        print(f"Selected {len(examples)} diverse examples")

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            # Remove internal metadata fields
            out = {k: v for k, v in ex.items() if not k.startswith("_")}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} SFT examples to {output_path}")


if __name__ == "__main__":
    main()
