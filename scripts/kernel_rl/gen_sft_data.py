"""Generate SFT training data for Triton kernel development.

Each example teaches the model to output three explicit sections:
    <design>  — optimization strategy, correctness constraints, memory tradeoffs
    <code>    — the actual Triton kernel or edited code block
    <predict> — expected performance, bottlenecks, and confidence

Output: data/kernel_rl/sft_train.jsonl
"""

from __future__ import annotations

import json
import os

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "kernel_rl", "sft_train.jsonl"
)

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

# ============================================================================
# SFT examples
# ============================================================================

EXAMPLES = [
    # ------------------------------------------------------------------
    # 1. Vector Add (element-wise, simplest Triton pattern)
    # ------------------------------------------------------------------
    {
        "prompt": (
            "You are improving a Triton operator implementation.\n"
            "Read the task carefully, preserve correctness, and optimize for real GPU performance.\n\n"
            "Task specification:\n"
            "{\n"
            '  "name": "vector_add",\n'
            '  "description": "Element-wise vector addition C = A + B. All three tensors have the same shape.",\n'
            '  "input_shape": [1048576],\n'
            '  "dtype": "float32",\n'
            '  "constraints": "preserve numerical precision, handle arbitrary alignment, avoid bank conflicts"\n'
            "}\n\n"
            "Reference Python implementation:\n"
            "```python\n"
            "def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:\n"
            "    return a + b\n"
            "```\n\n"
            + FORMAT_INSTRUCTIONS
        ),
        "completion": (
            "<design>\n"
            "Element-wise vector addition is a memory-bound operation. The key optimization is to maximize memory bandwidth utilization by using coalesced global memory access patterns.\n\n"
            "Strategy:\n"
            "- Each program instance processes a contiguous block of elements to ensure cache-line-aligned loads/stores.\n"
            "- Use a power-of-2 BLOCK_SIZE (1024) to maximize L1 cache line usage and minimize bank conflicts.\n"
            "- The grid is computed as ceil(N / BLOCK_SIZE), giving enough program instances to cover all elements.\n"
            "- Apply a boundary mask to handle the last partial block correctly — this is the only correctness constraint beyond the trivial pointwise addition.\n\n"
            "Launch tradeoffs:\n"
            "- BLOCK_SIZE=1024 gives good occupancy (1 warp per program = 32 threads, 32 warps per SM @ 4 SMs).\n"
            "- The kernel is fully bandwidth-bound, so increasing BLOCK_SIZE beyond 1024 does not improve throughput on most GPUs.\n"
            "- No shared memory is needed since each element is accessed only once.\n"
            "</design>\n"
            "<code>\n"
            "import torch\n"
            "import triton\n"
            "import triton.language as tl\n\n"
            "@triton.jit\n"
            "def add_kernel(\n"
            "    a_ptr, b_ptr, out_ptr,\n"
            "    n_elements: int,\n"
            "    BLOCK_SIZE: tl.constexpr,\n"
            "):\n"
            "    pid = tl.program_id(0)\n"
            "    block_start = pid * BLOCK_SIZE\n"
            "    offsets = block_start + tl.arange(0, BLOCK_SIZE)\n"
            "    mask = offsets < n_elements\n"
            "    a = tl.load(a_ptr + offsets, mask=mask)\n"
            "    b = tl.load(b_ptr + offsets, mask=mask)\n"
            "    tl.store(out_ptr + offsets, a + b, mask=mask)\n\n\n"
            "def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:\n"
            "    assert a.shape == b.shape, f'Shape mismatch: {a.shape} vs {b.shape}'\n"
            "    n = a.numel()\n"
            "    out = torch.empty_like(a)\n"
            "    BLOCK_SIZE = 1024\n"
            "    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)\n"
            "    add_kernel[grid](a, b, out, n, BLOCK_SIZE=BLOCK_SIZE)\n"
            "    return out\n"
            "</code>\n"
            "<predict>\n"
            "Expected performance:\n"
            "- This kernel is memory-bandwidth-bound. On an RTX 4090 with ~1000 GB/s memory bandwidth,\n"
            "  the theoretical max throughput for float32 (4 bytes per element, 3 tensors = 12 bytes per FLOP)\n"
            "  is ~83 GB/s → ~83e9 / 12 ≈ 7e9 elements/s.\n"
            "- For 1M elements, expected runtime ~0.15 ms (cold cache).\n"
            "- Speedup over eager PyTorch: ~0.8-1.2x. PyTorch's built-in add is already heavily optimized;\n"
            "  Triton can match it but not significantly beat it on pure element-wise ops.\n"
            "- Where Triton wins: when this add is fused into a larger kernel (e.g., add + activation),\n"
            "  avoiding the extra memory round-trip saves ~30-40% bandwidth.\n"
            "Confidence: high (element-wise add is the most basic Triton kernel, well-studied).\n"
            "</predict>"
        ),
    },

    # ------------------------------------------------------------------
    # 2. GELU Activation
    # ------------------------------------------------------------------
    {
        "prompt": (
            "You are improving a Triton operator implementation.\n"
            "Read the task carefully, preserve correctness, and optimize for real GPU performance.\n\n"
            "Task specification:\n"
            "{\n"
            '  "name": "gelu_activation",\n'
            '  "description": "GELU activation function: GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))",\n'
            '  "input_shape": [65536, 256],\n'
            '  "dtype": "float32",\n'
            '  "constraints": "bitwise identical to torch.nn.functional.gelu(approximate=none)"\n'
            "}\n\n"
            "Reference Python implementation:\n"
            "```python\n"
            "def gelu(x: torch.Tensor) -> torch.Tensor:\n"
            "    import torch.nn.functional as F\n"
            "    return F.gelu(x, approximate='none')\n"
            "```\n\n"
            + FORMAT_INSTRUCTIONS
        ),
        "completion": (
            "<design>\n"
            "GELU is element-wise and memory-bound. The exact formula x * 0.5 * (1 + erf(x / sqrt(2))) involves the error function, which is expensive to compute from scratch.\n\n"
            "Strategy:\n"
            "- The key optimization is using an efficient polynomial approximation for erf(x) instead of the exact formula. The tanh approximation (used in PyTorch's 'approximate=tanh' mode) is faster:\n"
            "  GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))\n"
            "- This replaces a transcendental erf() with a few multiply-adds and one tanh().\n"
            "- Load coalesced blocks, compute the polynomial, and write results.\n"
            "- When accuracy requires the exact erf, we compute it directly — the memory wall still dominates.\n\n"
            "Block size choice: 1024 elements per block gives good GPU occupancy.\n"
            "</design>\n"
            "<code>\n"
            "import torch\n"
            "import triton\n"
            "import triton.language as tl\n\n"
            "@triton.jit\n"
            "def gelu_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):\n"
            "    pid = tl.program_id(0)\n"
            "    block_start = pid * BLOCK_SIZE\n"
            "    offsets = block_start + tl.arange(0, BLOCK_SIZE)\n"
            "    mask = offsets < n_elements\n\n"
            "    x = tl.load(in_ptr + offsets, mask=mask)\n"
            "    # Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))\n"
            '    sqrt2 = 1.4142135623730951\n'
            "    y = 0.5 * x * (1.0 + tl.math.erf(x / sqrt2))\n"
            "    tl.store(out_ptr + offsets, y, mask=mask)\n\n\n"
            "def gelu(x: torch.Tensor) -> torch.Tensor:\n"
            "    n = x.numel()\n"
            "    out = torch.empty_like(x)\n"
            "    BLOCK_SIZE = 1024\n"
            "    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)\n"
            "    gelu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)\n"
            "    return out\n"
            "</code>\n"
            "<predict>\n"
            "Performance estimate:\n"
            "- Memory-bound: 1 read + 1 write = 8 bytes/element (fp32).\n"
            "- On RTX 4090 (~1000 GB/s), theoretical ~125e9 elements/s.\n"
            "- For 16M elements (65536*256), expected ~0.13 ms.\n"
            "- Speedup over PyTorch eager: ~1.0-1.1x. The main gain comes when this kernel is fused\n"
            "  with the preceding or following operation (e.g., matmul → GELU fusion).\n"
            "- Using the tanh approximation would be ~15% faster at the cost of ~1e-6 absolute error.\n"
            "Confidence: high.\n"
            "</predict>"
        ),
    },

    # ------------------------------------------------------------------
    # 3. RMSNorm (forward)
    # ------------------------------------------------------------------
    {
        "prompt": (
            "You are improving a Triton operator implementation.\n"
            "Read the task carefully, preserve correctness, and optimize for real GPU performance.\n\n"
            "Task specification:\n"
            "{\n"
            '  "name": "rms_norm",\n'
            '  "description": "Root Mean Square Normalization: y = x / sqrt(mean(x^2) + eps). Normalize along the last dimension.",\n'
            '  "input_shape": [256, 1024],\n'
            '  "dtype": "float32",\n'
            '  "constraints": "numerically stable, match PyTorch reference to 1e-5 rtol, avoid nan/inf for extreme inputs"\n'
            "}\n\n"
            "Reference Python implementation:\n"
            "```python\n"
            "def rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:\n"
            "    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)\n"
            "    return x / rms\n"
            "```\n\n"
            + FORMAT_INSTRUCTIONS
        ),
        "completion": (
            "<design>\n"
            "RMSNorm requires a reduction (mean of squares along the feature dimension) followed by an element-wise normalization. The reduction step is the performance bottleneck because it requires synchronization across threads within a program.\n\n"
            "Strategy:\n"
            "- Use a 2D grid: one dimension for rows (batch*sequence), one for the reduction within each row.\n"
            "- Each program loads a block of the feature vector, computes partial sums of x^2 in registers.\n"
            "- Use tl.reduce or manual warp-level reduction to accumulate across the feature dimension.\n"
            "- Store the partial sum to shared memory, synchronize, then reduce across the warp.\n"
            "- The normalization step is a simple element-wise division — fast and trivially parallel.\n"
            "- Choose NUM_WARPS such that each warp handles a good chunk of the feature dim.\n\n"
            "Correctness: eps guards against division by zero. Use float32 accumulation for the sum-of-squares to avoid overflow for large feature dimensions.\n"
            "</design>\n"
            "<code>\n"
            "import torch\n"
            "import triton\n"
            "import triton.language as tl\n\n"
            "@triton.jit\n"
            "def rmsnorm_kernel(\n"
            "    x_ptr, out_ptr,\n"
            "    n_rows: int,\n"
            "    n_cols: int,\n"
            "    eps: float,\n"
            "    BLOCK_SIZE: tl.constexpr,\n"
            "):\n"
            "    row_idx = tl.program_id(0)\n"
            "    col_start = tl.program_id(1) * BLOCK_SIZE\n"
            "    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)\n"
            "    col_mask = col_offsets < n_cols\n\n"
            "    x_offsets = row_idx * n_cols + col_offsets\n"
            "    x = tl.load(x_ptr + x_offsets, mask=col_mask, other=0.0)\n\n"
            "    # Compute partial sum of squares\n"
            "    x_sq = x * x\n"
            "    partial_sum = tl.sum(x_sq, axis=0)\n\n"
            "    # Atomic add to accumulate across programs in the same row\n"
            "    sum_ptr = out_ptr + n_rows * n_cols + row_idx  # use output tail as scratch\n"
            "    tl.atomic_add(sum_ptr, partial_sum)\n"
            "    tl.debug_barrier()\n\n"
            "    # Read the total sum and compute rms\n"
            "    total_sum = tl.load(sum_ptr)\n"
            "    rms = tl.sqrt(total_sum / n_cols + eps)\n\n"
            "    # Normalize and store\n"
            "    y = x / rms\n"
            "    tl.store(out_ptr + x_offsets, y, mask=col_mask)\n\n\n"
            "def rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:\n"
            "    assert x.ndim >= 2, f'Expected >= 2D tensor, got {x.ndim}D'\n"
            "    n_rows = x.numel() // x.shape[-1]\n"
            "    n_cols = x.shape[-1]\n"
            "    out = torch.empty(n_rows * n_cols + n_rows, dtype=x.dtype, device=x.device)\n"
            "    BLOCK_SIZE = 128\n"
            "    grid = (n_rows, triton.cdiv(n_cols, BLOCK_SIZE))\n"
            "    rmsnorm_kernel[grid](x, out, n_rows, n_cols, eps, BLOCK_SIZE=BLOCK_SIZE)\n"
            "    return out[:n_rows * n_cols].view_as(x)\n"
            "</code>\n"
            "<predict>\n"
            "Performance estimate:\n"
            "- RMSNorm is reduction-bound: the mean-of-squares step requires cross-thread synchronization.\n"
            "- On RTX 4090, for (256, 1024) input:\n"
            "  - Memory: 256*1024*4 = 1MB read + 1MB write = 2MB total → ~0.002 ms at full bandwidth.\n"
            "  - Reduction overhead: atomic add across ~8 programs per row adds ~0.01 ms.\n"
            "  - Expected total ~0.02-0.05 ms per call.\n"
            "- Speedup over PyTorch eager: ~1.2-1.5x. PyTorch's native RMSNorm (via CUDA) is well-optimized,\n"
            "  but our Triton kernel avoids the framework overhead of small kernel launches.\n"
            "- The atomic approach works for moderate feature dims (<4096). For larger dims, a two-pass\n"
            "  kernel (compute global mean first, then normalize) would be more efficient.\n"
            "Confidence: medium (the atomic-based approach is simple but not optimal for all dims).\n"
            "</predict>"
        ),
    },

    # ------------------------------------------------------------------
    # 4. Softmax (row-wise, numerically stable)
    # ------------------------------------------------------------------
    {
        "prompt": (
            "You are improving a Triton operator implementation.\n"
            "Read the task carefully, preserve correctness, and optimize for real GPU performance.\n\n"
            "Task specification:\n"
            "{\n"
            '  "name": "softmax",\n'
            '  "description": "Row-wise softmax along the last dimension: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_i - max(x)))",\n'
            '  "input_shape": [4096, 4096],\n'
            '  "dtype": "float32",\n'
            '  "constraints": "numerically stable (subtract max before exp), avoid nan/inf, match PyTorch to 1e-5 rtol"\n'
            "}\n\n"
            "Reference Python implementation:\n"
            "```python\n"
            "def softmax(x: torch.Tensor) -> torch.Tensor:\n"
            "    import torch.nn.functional as F\n"
            "    return F.softmax(x, dim=-1)\n"
            "```\n\n"
            + FORMAT_INSTRUCTIONS
        ),
        "completion": (
            "<design>\n"
            "Row-wise softmax requires a two-pass algorithm for numerical stability: first compute the row-wise max (for stable exp), then compute exp(x - max) and normalize. Each row is independent, so rows are parallelized across program instances.\n\n"
            "Strategy:\n"
            "- Use a 1D grid where each program handles one complete row.\n"
            "- Each program iterates over the row in blocks, computing:\n"
            "  Pass 1: row-wise maximum (to subtract for stability).\n"
            "  Pass 2: sum of exp(x_i - max) for normalization.\n"
            "  Pass 3: compute exp(x_i - max) / sum and write output.\n"
            "- This online algorithm avoids materializing the full exp(x) intermediate tensor,\n"
            "  saving memory bandwidth.\n"
            "- Use tl.math.exp for fast hardware-accelerated exp on GPU.\n\n"
            "Block size: 128-256 elements per iteration. Larger blocks reduce loop iterations but increase register pressure.\n"
            "</design>\n"
            "<code>\n"
            "import torch\n"
            "import triton\n"
            "import triton.language as tl\n\n"
            "@triton.jit\n"
            "def softmax_kernel(in_ptr, out_ptr, n_cols: int, BLOCK_SIZE: tl.constexpr):\n"
            "    row_idx = tl.program_id(0)\n"
            "    row_start = row_idx * n_cols\n\n"
            "    # Pass 1: find max for numerical stability\n"
            "    row_max = float('-inf')\n"
            "    for start in range(0, n_cols, BLOCK_SIZE):\n"
            "        offsets = start + tl.arange(0, BLOCK_SIZE)\n"
            "        mask = offsets < n_cols\n"
            "        x = tl.load(in_ptr + row_start + offsets, mask=mask, other=float('-inf'))\n"
            "        row_max = tl.maximum(row_max, tl.max(x, axis=0))\n\n"
            "    # Pass 2: compute exp sum\n"
            "    exp_sum = 0.0\n"
            "    for start in range(0, n_cols, BLOCK_SIZE):\n"
            "        offsets = start + tl.arange(0, BLOCK_SIZE)\n"
            "        mask = offsets < n_cols\n"
            "        x = tl.load(in_ptr + row_start + offsets, mask=mask, other=float('-inf'))\n"
            "        exp_x = tl.exp(x - row_max)\n"
            "        exp_sum += tl.sum(exp_x, axis=0)\n\n"
            "    # Pass 3: normalize and write\n"
            "    for start in range(0, n_cols, BLOCK_SIZE):\n"
            "        offsets = start + tl.arange(0, BLOCK_SIZE)\n"
            "        mask = offsets < n_cols\n"
            "        x = tl.load(in_ptr + row_start + offsets, mask=mask, other=float('-inf'))\n"
            "        y = tl.exp(x - row_max) / exp_sum\n"
            "        tl.store(out_ptr + row_start + offsets, y, mask=mask)\n\n\n"
            "def softmax(x: torch.Tensor) -> torch.Tensor:\n"
            "    assert x.ndim >= 2, f'Expected >= 2D, got {x.ndim}D'\n"
            "    n_rows = x.numel() // x.shape[-1]\n"
            "    n_cols = x.shape[-1]\n"
            "    out = torch.empty_like(x)\n"
            "    BLOCK_SIZE = 128\n"
            "    grid = (n_rows,)\n"
            "    softmax_kernel[grid](x, out, n_cols, BLOCK_SIZE=BLOCK_SIZE)\n"
            "    return out\n"
            "</code>\n"
            "<predict>\n"
            "Performance:\n"
            "- For (4096, 4096) input: 4096 rows × 4096 cols = 16M elements.\n"
            "- Each row requires 3 passes over 4096 elements → ~12K loads + ~4K stores per row.\n"
            "- Total memory: ~16M * 4 * 4 = 256 MB (fp32).\n"
            "- On RTX 4090 (~1000 GB/s): ~0.25 ms for data movement.\n"
            "- Additional compute: 3 * 16M exp operations → ~0.1 ms.\n"
            "- Total expected: ~0.3-0.5 ms.\n"
            "- Speedup over PyTorch eager: ~1.5-2x (PyTorch's multi-kernel launch overhead is significant for softmax).\n"
            "- The three-pass algorithm is simple but reads each element 3 times.\n"
            "  A more advanced online algorithm (using shared memory for partial reductions)\n"
            "  could reduce to 1.5 passes and gain another ~30% throughput.\n"
            "Confidence: medium-high.\n"
            "</predict>"
        ),
    },

    # ------------------------------------------------------------------
    # 5. Matrix Multiply (tiled, with shared memory)
    # ------------------------------------------------------------------
    {
        "prompt": (
            "You are improving a Triton operator implementation.\n"
            "Read the task carefully, preserve correctness, and optimize for real GPU performance.\n\n"
            "Task specification:\n"
            "{\n"
            '  "name": "matmul_tiled",\n'
            '  "description": "Standard matrix multiplication C = A @ B where A is (M, K) and B is (K, N). Use tiled shared-memory algorithm with block-level accumulation.",\n'
            '  "input_shape": {"A": [1024, 1024], "B": [1024, 1024]},\n'
            '  "dtype": "float32",\n'
            '  "constraints": "match PyTorch matmul to 1e-3 rtol, use tiling with BLOCK_M x BLOCK_N x BLOCK_K, accumulate in registers"\n'
            "}\n\n"
            "Reference Python implementation:\n"
            "```python\n"
            "def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:\n"
            "    return a @ b\n"
            "```\n\n"
            + FORMAT_INSTRUCTIONS
        ),
        "completion": (
            "<design>\n"
            "Matrix multiplication is compute-bound. The key optimization is tiling: decomposing the M×K×N problem into small blocks that fit in shared memory and registers, then accumulating partial results.\n\n"
            "Strategy:\n"
            "- 2D grid over the output (M, N), each program computes one BLOCK_M × BLOCK_N tile of C.\n"
            "- Inner loop over K dimension: each iteration loads a BLOCK_M × BLOCK_K tile of A and BLOCK_K × BLOCK_N tile of B into shared memory.\n"
            "- Use tl.dot() for warp-level matrix multiply-accumulate on tensor cores (if available).\n"
            "- Block sizes: BLOCK_M=128, BLOCK_N=128, BLOCK_K=32. This gives 128×128 output per program with good SM utilization.\n"
            "- Shared memory usage: (128*32 + 32*128) * 4 bytes = 32 KB per program, well within 48 KB limit.\n"
            "- Register accumulation: each thread accumulates a small sub-tile and writes once at the end.\n\n"
            "Correctness: ensure K dimension loop covers all elements. Use boundary masks for non-multiple dimensions.\n"
            "</design>\n"
            "<code>\n"
            "import torch\n"
            "import triton\n"
            "import triton.language as tl\n\n"
            "@triton.jit\n"
            "def matmul_kernel(\n"
            "    a_ptr, b_ptr, c_ptr,\n"
            "    M: int, N: int, K: int,\n"
            "    stride_am: int, stride_ak: int,\n"
            "    stride_bk: int, stride_bn: int,\n"
            "    stride_cm: int, stride_cn: int,\n"
            "    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,\n"
            "):\n"
            "    pid_m = tl.program_id(0)\n"
            "    pid_n = tl.program_id(1)\n\n"
            "    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)\n"
            "    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)\n"
            "    offs_k = tl.arange(0, BLOCK_K)\n\n"
            "    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak\n"
            "    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn\n\n"
            "    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)\n"
            "    for k in range(0, K, BLOCK_K):\n"
            "        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k)\n"
            "        b_mask = (offs_k[:, None] < K - k) & (offs_n[None, :] < N)\n"
            "        a = tl.load(a_ptrs, mask=a_mask, other=0.0)\n"
            "        b = tl.load(b_ptrs, mask=b_mask, other=0.0)\n"
            "        accumulator += tl.dot(a, b)\n"
            "        a_ptrs += BLOCK_K * stride_ak\n"
            "        b_ptrs += BLOCK_K * stride_bk\n\n"
            "    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)\n"
            "    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn\n"
            "    tl.store(c_ptrs, accumulator, mask=c_mask)\n\n\n"
            "def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:\n"
            "    assert a.ndim == 2 and b.ndim == 2, 'Only 2D matmul supported'\n"
            "    M, K_a = a.shape\n"
            "    K_b, N = b.shape\n"
            "    assert K_a == K_b, f'Inner dim mismatch: {K_a} vs {K_b}'\n"
            "    c = torch.empty((M, N), dtype=a.dtype, device=a.device)\n"
            "    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32\n"
            "    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))\n"
            "    matmul_kernel[grid](\n"
            "        a, b, c,\n"
            "        M, N, K_a,\n"
            "        a.stride(0), a.stride(1),\n"
            "        b.stride(0), b.stride(1),\n"
            "        c.stride(0), c.stride(1),\n"
            "        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,\n"
            "    )\n"
            "    return c\n"
            "</code>\n"
            "<predict>\n"
            "Performance (1024x1024 fp32 on RTX 4090):\n"
            "- Compute: 2*M*N*K = 2*1024^3 = 2.1G FLOPs.\n"
            "- RTX 4090 theoretical: 82.6 TFLOPS (fp32). Expected matmul throughput ~30-40 TFLOPS for tiled kernel.\n"
            "- Expected runtime: 2.1G / 35 TFLOPS ≈ 0.06 ms.\n"
            "- Memory: A(4MB) + B(4MB) + C(4MB) = 12MB. At 1000 GB/s: 0.012 ms. Compute clearly dominates.\n"
            "- Speedup vs PyTorch eager: ~0.3-0.5x initially (PyTorch calls cuBLAS, heavily optimized).\n"
            "  For a 7B model's SFT-generated kernel, matching cuBLAS is unrealistic.\n"
            "  The educational value is in learning tiling, shared memory, and tl.dot.\n"
            "- Speedup vs naive triple-loop: >100x.\n"
            "Confidence: medium (tiled matmul is the classic Triton tutorial; real-world performance depends on tuning block sizes for the specific GPU).\n"
            "</predict>"
        ),
    },

    # ------------------------------------------------------------------
    # 6. LayerNorm (forward)
    # ------------------------------------------------------------------
    {
        "prompt": (
            "You are improving a Triton operator implementation.\n"
            "Read the task carefully, preserve correctness, and optimize for real GPU performance.\n\n"
            "Task specification:\n"
            "{\n"
            '  "name": "layer_norm",\n'
            '  "description": "Layer Normalization: y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta. Normalize along the last dimension.",\n'
            '  "input_shape": [256, 1024],\n'
            '  "dtype": "float32",\n'
            '  "constraints": "numerically stable, handle trainable gamma/beta parameters, match PyTorch to 1e-5 rtol"\n'
            "}\n\n"
            "Reference Python implementation:\n"
            "```python\n"
            "def layer_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:\n"
            "    import torch.nn.functional as F\n"
            "    return F.layer_norm(x, gamma.shape, gamma, beta, eps)\n"
            "```\n\n"
            + FORMAT_INSTRUCTIONS
        ),
        "completion": (
            "<design>\n"
            "LayerNorm requires computing both mean and variance along the feature dimension, then applying affine transform with gamma/beta. This is a two-pass reduction similar to RMSNorm but with an additional mean computation.\n\n"
            "Strategy:\n"
            "- Online Welford algorithm for numerically stable mean+variance in one pass:\n"
            "  mean = sum(x_i) / N\n"
            "  var  = sum((x_i - mean)^2) / N\n"
            "- Each program handles one row. Iterate over feature dim in blocks.\n"
            "- Pass 1: accumulate partial sums for mean (sum of x) and variance (sum of x^2).\n"
            "- After the loop: compute global mean and var for the row.\n"
            "- Pass 2: normalize and apply gamma/beta.\n"
            "- Use shared memory for inter-block reduction within a row (requires multiple programs per row for large feature dims).\n"
            "- For small-to-medium feature dims (<2048), a single program per row is sufficient.\n"
            "- BLOCK_SIZE=256 balances register usage and loop iterations.\n"
            "</design>\n"
            "<code>\n"
            "import torch\n"
            "import triton\n"
            "import triton.language as tl\n\n"
            "@triton.jit\n"
            "def layernorm_kernel(\n"
            "    x_ptr, gamma_ptr, beta_ptr, out_ptr,\n"
            "    n_cols: int,\n"
            "    eps: float,\n"
            "    BLOCK_SIZE: tl.constexpr,\n"
            "):\n"
            "    row_idx = tl.program_id(0)\n"
            "    row_start = row_idx * n_cols\n\n"
            "    # Pass 1: compute mean and variance\n"
            "    mean_sum = 0.0\n"
            "    var_sum = 0.0\n"
            "    for start in range(0, n_cols, BLOCK_SIZE):\n"
            "        offsets = start + tl.arange(0, BLOCK_SIZE)\n"
            "        mask = offsets < n_cols\n"
            "        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)\n"
            "        mean_sum += tl.sum(x, axis=0)\n"
            "        var_sum += tl.sum(x * x, axis=0)\n\n"
            "    mean = mean_sum / n_cols\n"
            "    var = var_sum / n_cols - mean * mean\n"
            "    inv_std = 1.0 / tl.sqrt(var + eps)\n\n"
            "    # Pass 2: normalize and apply gamma/beta\n"
            "    for start in range(0, n_cols, BLOCK_SIZE):\n"
            "        offsets = start + tl.arange(0, BLOCK_SIZE)\n"
            "        mask = offsets < n_cols\n"
            "        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)\n"
            "        gamma = tl.load(gamma_ptr + offsets, mask=mask, other=1.0)\n"
            "        beta = tl.load(beta_ptr + offsets, mask=mask, other=0.0)\n"
            "        y = (x - mean) * inv_std * gamma + beta\n"
            "        tl.store(out_ptr + row_start + offsets, y, mask=mask)\n\n\n"
            "def layer_norm(\n"
            "    x: torch.Tensor,\n"
            "    gamma: torch.Tensor,\n"
            "    beta: torch.Tensor,\n"
            "    eps: float = 1e-5,\n"
            ") -> torch.Tensor:\n"
            "    n_rows = x.numel() // x.shape[-1]\n"
            "    n_cols = x.shape[-1]\n"
            "    out = torch.empty_like(x)\n"
            "    BLOCK_SIZE = 256\n"
            "    grid = (n_rows,)\n"
            "    layernorm_kernel[grid](x, gamma, beta, out, n_cols, eps, BLOCK_SIZE=BLOCK_SIZE)\n"
            "    return out\n"
            "</code>\n"
            "<predict>\n"
            "Performance:\n"
            "- LayerNorm does 2 passes: one for mean+var, one for normalization.\n"
            "- For (256, 1024) input (256KB): ~0.5MB total I/O. Memory time negligible (~0.001 ms).\n"
            "- Compute: 256*1024*~10 ops ≈ 2.6M ops. Trivial on GPU.\n"
            "- Runtime dominated by kernel launch overhead (~0.005-0.01 ms).\n"
            "- Speedup vs PyTorch: ~1.2-1.5x for small tensors, converging to 1.0x for large tensors.\n"
            "- Main advantage: when fused with preceding operation (e.g., Linear → LayerNorm), avoiding the intermediate write+read saves ~50% bandwidth.\n"
            "Confidence: high.\n"
            "</predict>"
        ),
    },

    # ------------------------------------------------------------------
    # 7. ReLU (simplest activation)
    # ------------------------------------------------------------------
    {
        "prompt": (
            "You are improving a Triton operator implementation.\n"
            "Read the task carefully, preserve correctness, and optimize for real GPU performance.\n\n"
            "Task specification:\n"
            "{\n"
            '  "name": "relu",\n'
            '  "description": "ReLU activation: max(0, x). Purely element-wise with no reduction.",\n'
            '  "input_shape": [131072, 64],\n'
            '  "dtype": "float32",\n'
            '  "constraints": "exact match to torch.relu, handle negative zero correctly"\n'
            "}\n\n"
            "Reference Python implementation:\n"
            "```python\n"
            "def relu(x: torch.Tensor) -> torch.Tensor:\n"
            "    import torch.nn.functional as F\n"
            "    return F.relu(x)\n"
            "```\n\n"
            + FORMAT_INSTRUCTIONS
        ),
        "completion": (
            "<design>\n"
            "ReLU is the simplest activation: max(0, x). It is fully memory-bound with trivial compute.\n\n"
            "Strategy:\n"
            "- Each program loads a contiguous block of elements, applies max(0, x) element-wise, and stores the result.\n"
            "- Use BLOCK_SIZE=1024 for good occupancy. No shared memory needed.\n"
            "- The grid covers ceil(N / BLOCK_SIZE) programs. Boundary mask handles the last partial block.\n"
            "- No reduction, no synchronization — embarrassingly parallel.\n"
            "- The key value of a Triton ReLU kernel is fusion: when composed with the preceding kernel\n"
            "  (e.g., matmul+relu, conv+relu), the fusion saves one full round-trip to global memory (~30-50% bandwidth savings).\n"
            "</design>\n"
            "<code>\n"
            "import torch\n"
            "import triton\n"
            "import triton.language as tl\n\n"
            "@triton.jit\n"
            "def relu_kernel(in_ptr, out_ptr, n_elements: int, BLOCK_SIZE: tl.constexpr):\n"
            "    pid = tl.program_id(0)\n"
            "    block_start = pid * BLOCK_SIZE\n"
            "    offsets = block_start + tl.arange(0, BLOCK_SIZE)\n"
            "    mask = offsets < n_elements\n"
            "    x = tl.load(in_ptr + offsets, mask=mask)\n"
            "    y = tl.maximum(x, 0.0)\n"
            "    tl.store(out_ptr + offsets, y, mask=mask)\n\n\n"
            "def relu(x: torch.Tensor) -> torch.Tensor:\n"
            "    n = x.numel()\n"
            "    out = torch.empty_like(x)\n"
            "    BLOCK_SIZE = 1024\n"
            "    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)\n"
            "    relu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)\n"
            "    return out\n"
            "</code>\n"
            "<predict>\n"
            "Performance:\n"
            "- Pure memory-bound: 1 read + 1 write = 8 bytes/element (fp32).\n"
            "- 131072*64 = 8.4M elements ≈ 67 MB total I/O. At 1000 GB/s: ~0.067 ms.\n"
            "- Expected runtime: ~0.07-0.10 ms (kernel launch overhead adds ~0.005 ms).\n"
            "- Speedup vs PyTorch: ~1.0x. PyTorch's ReLU is already a simple CUDA kernel.\n"
            "- Real value: fusion with preceding matmul/conv saves one memory round-trip.\n"
            "Confidence: very high.\n"
            "</predict>"
        ),
    },

    # ------------------------------------------------------------------
    # 8. Fused GELU + Multiply (fusion pattern)
    # ------------------------------------------------------------------
    {
        "prompt": (
            "You are improving a Triton operator implementation.\n"
            "Read the task carefully, preserve correctness, and optimize for real GPU performance.\n\n"
            "Task specification:\n"
            "{\n"
            '  "name": "fused_gelu_mul",\n'
            '  "description": "Fused operation: y = GELU(x) * scale, where scale is a scalar. Demonstrate kernel fusion by computing GELU and multiply in a single kernel to avoid intermediate tensor allocation.",\n'
            '  "input_shape": [65536, 128],\n'
            '  "dtype": "float32",\n'
            '  "constraints": "same GELU accuracy as unfused version, demonstrate memory bandwidth savings from fusion"\n'
            "}\n\n"
            "Reference Python implementation:\n"
            "```python\n"
            "def fused_gelu_mul(x: torch.Tensor, scale: float) -> torch.Tensor:\n"
            "    import torch.nn.functional as F\n"
            "    return F.gelu(x) * scale\n"
            "```\n\n"
            + FORMAT_INSTRUCTIONS
        ),
        "completion": (
            "<design>\n"
            "This kernel demonstrates fusion: combining GELU and scalar multiply into a single GPU kernel. Without fusion, two separate kernels execute: (1) GELU: read X → compute → write intermediate, (2) multiply: read intermediate → compute → write output. With fusion: read X → compute GELU → compute multiply → write output, saving one full memory round-trip.\n\n"
            "Strategy:\n"
            "- Single kernel loads x, computes y = GELU(x) * scale, stores y.\n"
            "- Memory savings: 1 read + 1 write instead of 1 read + 2 writes (unfused), saving 33% bandwidth.\n"
            "- The compute is unchanged; the win is purely from reduced memory traffic.\n"
            "- This pattern generalizes: any element-wise sequence can be fused into one kernel.\n"
            "- BLOCK_SIZE=1024 as standard for element-wise ops.\n"
            "</design>\n"
            "<code>\n"
            "import torch\n"
            "import triton\n"
            "import triton.language as tl\n\n"
            "@triton.jit\n"
            "def fused_gelu_mul_kernel(in_ptr, out_ptr, n_elements: int, scale: float, BLOCK_SIZE: tl.constexpr):\n"
            "    pid = tl.program_id(0)\n"
            "    block_start = pid * BLOCK_SIZE\n"
            "    offsets = block_start + tl.arange(0, BLOCK_SIZE)\n"
            "    mask = offsets < n_elements\n\n"
            "    x = tl.load(in_ptr + offsets, mask=mask)\n"
            "    # GELU: x * 0.5 * (1 + erf(x / sqrt(2)))\n"
            "    sqrt2 = 1.4142135623730951\n"
            "    gelu_x = 0.5 * x * (1.0 + tl.math.erf(x / sqrt2))\n"
            "    y = gelu_x * scale\n"
            "    tl.store(out_ptr + offsets, y, mask=mask)\n\n\n"
            "def fused_gelu_mul(x: torch.Tensor, scale: float) -> torch.Tensor:\n"
            "    n = x.numel()\n"
            "    out = torch.empty_like(x)\n"
            "    BLOCK_SIZE = 1024\n"
            "    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)\n"
            "    fused_gelu_mul_kernel[grid](x, out, n, scale, BLOCK_SIZE=BLOCK_SIZE)\n"
            "    return out\n"
            "</code>\n"
            "<predict>\n"
            "Performance (vs unfused):\n"
            "- Unfused: 4 memory operations (2 reads + 2 writes) = 16 bytes/element.\n"
            "- Fused:   2 memory operations (1 read + 1 write)   = 8 bytes/element.\n"
            "- Expected speedup vs unfused: ~1.8-2.0x (memory bandwidth is the bottleneck).\n"
            "- For 8.4M elements: fused ~0.07 ms, unfused ~0.14 ms.\n"
            "- This demonstrates why kernel fusion is the primary use case for custom Triton kernels\n"
            "  in production: individual element-wise ops are rarely worth rewriting, but fusing 2-3 ops\n"
            "  together consistently yields 1.5-3x speedups.\n"
            "Confidence: high (fusion benefit is well-understood and predictable from bandwidth arithmetic).\n"
            "</predict>"
        ),
    },

    # ------------------------------------------------------------------
    # 9. Swish / SiLU Activation
    # ------------------------------------------------------------------
    {
        "prompt": (
            "You are improving a Triton operator implementation.\n"
            "Read the task carefully, preserve correctness, and optimize for real GPU performance.\n\n"
            "Task specification:\n"
            "{\n"
            '  "name": "swish_activation",\n'
            '  "description": "Swish (SiLU) activation: Swish(x) = x * sigmoid(x) = x / (1 + exp(-x)). Widely used in modern LLMs.",\n'
            '  "input_shape": [32768, 256],\n'
            '  "dtype": "float32",\n'
            '  "constraints": "match torch.nn.functional.silu() exactly"\n'
            "}\n\n"
            "Reference Python implementation:\n"
            "```python\n"
            "def swish(x: torch.Tensor) -> torch.Tensor:\n"
            "    import torch.nn.functional as F\n"
            "    return F.silu(x)\n"
            "```\n\n"
            + FORMAT_INSTRUCTIONS
        ),
        "completion": (
            "<design>\n"
            "Swish is x * sigmoid(x). The sigmoid involves exp(), which is moderately expensive. The key optimization is computing sigmoid efficiently and handling extreme values to avoid NaN.\n\n"
            "Strategy:\n"
            "- Use the numerically stable formulation: for large negative x, sigmoid(x) → 0, so Swish → 0.\n"
            "  For large positive x, sigmoid(x) → 1, so Swish → x.\n"
            "- Compute sigmoid as 1 / (1 + exp(-x)). Clamp -x to avoid overflow in exp.\n"
            "- Alternatively use tl.sigmoid() which is hardware-optimized.\n"
            "- Element-wise, memory-bound. Same BLOCK_SIZE=1024 as other element-wise kernels.\n"
            "- For LLM inference, this kernel is typically fused with the preceding Linear layer.\n"
            "</design>\n"
            "<code>\n"
            "import torch\n"
            "import triton\n"
            "import triton.language as tl\n\n"
            "@triton.jit\n"
            "def swish_kernel(in_ptr, out_ptr, n_elements: int, BLOCK_SIZE: tl.constexpr):\n"
            "    pid = tl.program_id(0)\n"
            "    block_start = pid * BLOCK_SIZE\n"
            "    offsets = block_start + tl.arange(0, BLOCK_SIZE)\n"
            "    mask = offsets < n_elements\n\n"
            "    x = tl.load(in_ptr + offsets, mask=mask)\n"
            "    # Swish: x * sigmoid(x)\n"
            "    y = x * tl.sigmoid(x)\n"
            "    tl.store(out_ptr + offsets, y, mask=mask)\n\n\n"
            "def swish(x: torch.Tensor) -> torch.Tensor:\n"
            "    n = x.numel()\n"
            "    out = torch.empty_like(x)\n"
            "    BLOCK_SIZE = 1024\n"
            "    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)\n"
            "    swish_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)\n"
            "    return out\n"
            "</code>\n"
            "<predict>\n"
            "Performance:\n"
            "- Memory-bound: 8 bytes/element. 8.4M elements → ~0.07 ms.\n"
            "- tl.sigmoid is hardware-accelerated via GPU's special function unit (SFU).\n"
            "- Speedup vs PyTorch eager: ~1.0x (standalone). Speedup vs fused pattern: 1.5-2x.\n"
            "- In production LLM inference, Swish is always fused — this standalone kernel is a building block.\n"
            "Confidence: very high.\n"
            "</predict>"
        ),
    },

    # ------------------------------------------------------------------
    # 10. Row-wise Sum Reduction
    # ------------------------------------------------------------------
    {
        "prompt": (
            "You are improving a Triton operator implementation.\n"
            "Read the task carefully, preserve correctness, and optimize for real GPU performance.\n\n"
            "Task specification:\n"
            "{\n"
            '  "name": "row_sum",\n'
            '  "description": "Compute the sum of each row in a 2D tensor: out[i] = sum(input[i, :]). Reduction along the last dimension.",\n'
            '  "input_shape": [8192, 512],\n'
            '  "dtype": "float32",\n'
            '  "constraints": "match PyTorch sum(dim=-1) to 1e-4 rtol"\n'
            "}\n\n"
            "Reference Python implementation:\n"
            "```python\n"
            "def row_sum(x: torch.Tensor) -> torch.Tensor:\n"
            "    return x.sum(dim=-1)\n"
            "```\n\n"
            + FORMAT_INSTRUCTIONS
        ),
        "completion": (
            "<design>\n"
            "Row-wise sum is a reduction operation. The challenge is accumulating partial sums across many elements efficiently.\n\n"
            "Strategy:\n"
            "- Each program handles one row. Iterate over the row in blocks of BLOCK_SIZE.\n"
            "- Accumulate partial sums in a register. After the loop, write the single scalar result.\n"
            "- Use tl.sum() within each block for efficient warp-level reduction.\n"
            "- For rows shorter than a few thousand elements, this is efficient.\n"
            "- For very long rows (>10K), a tree reduction with multiple programs per row would be better.\n"
            "- BLOCK_SIZE=256 balances register usage and iteration count.\n"
            "- Output is a 1D tensor of shape (n_rows,).\n"
            "</design>\n"
            "<code>\n"
            "import torch\n"
            "import triton\n"
            "import triton.language as tl\n\n"
            "@triton.jit\n"
            "def row_sum_kernel(in_ptr, out_ptr, n_cols: int, BLOCK_SIZE: tl.constexpr):\n"
            "    row_idx = tl.program_id(0)\n"
            "    row_start = row_idx * n_cols\n\n"
            "    total = 0.0\n"
            "    for start in range(0, n_cols, BLOCK_SIZE):\n"
            "        offsets = start + tl.arange(0, BLOCK_SIZE)\n"
            "        mask = offsets < n_cols\n"
            "        x = tl.load(in_ptr + row_start + offsets, mask=mask, other=0.0)\n"
            "        total += tl.sum(x, axis=0)\n\n"
            "    tl.store(out_ptr + row_idx, total)\n\n\n"
            "def row_sum(x: torch.Tensor) -> torch.Tensor:\n"
            "    n_rows = x.numel() // x.shape[-1]\n"
            "    n_cols = x.shape[-1]\n"
            "    out = torch.empty(n_rows, dtype=x.dtype, device=x.device)\n"
            "    BLOCK_SIZE = 256\n"
            "    grid = (n_rows,)\n"
            "    row_sum_kernel[grid](x, out, n_cols, BLOCK_SIZE=BLOCK_SIZE)\n"
            "    return out\n"
            "</code>\n"
            "<predict>\n"
            "Performance:\n"
            "- 8192 rows × 512 cols = 4.2M elements. 1 read + 1/512 write (amortized) ≈ 4 bytes/element.\n"
            "- Memory: ~17 MB read + 32 KB write. At 1000 GB/s: ~0.017 ms.\n"
            "- Compute: 4.2M additions → negligible.\n"
            "- Expected runtime: ~0.02 ms.\n"
            "- Speedup vs PyTorch: ~1.0-1.2x. PyTorch's sum is already efficient; Triton's advantage is again in fusion.\n"
            "Confidence: high.\n"
            "</predict>"
        ),
    },

    # ------------------------------------------------------------------
    # 11. Scaled Dot-Product Attention (simplified, no softmax scaling bug)
    # ------------------------------------------------------------------
    {
        "prompt": (
            "You are improving a Triton operator implementation.\n"
            "Read the task carefully, preserve correctness, and optimize for real GPU performance.\n\n"
            "Task specification:\n"
            "{\n"
            '  "name": "scaled_dot_product_attention",\n'
            '  "description": "Simplified scaled dot-product attention: S = Q @ K^T / sqrt(d_k), then P = softmax(S), then O = P @ V. Q,K,V are (batch, heads, seq, d_k). Implement the core S = QK^T + softmax part (not full flash attention).",\n'
            '  "input_shape": {"Q": [1, 8, 1024, 64], "K": [1, 8, 1024, 64], "V": [1, 8, 1024, 64]},\n'
            '  "dtype": "float32",\n'
            '  "constraints": "match PyTorch scaled_dot_product_attention, handle causal mask if needed"\n'
            "}\n\n"
            "Reference Python implementation:\n"
            "```python\n"
            "def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:\n"
            "    import torch.nn.functional as F\n"
            "    return F.scaled_dot_product_attention(q, k, v)\n"
            "```\n\n"
            + FORMAT_INSTRUCTIONS
        ),
        "completion": (
            "<design>\n"
            "Scaled dot-product attention is the core of transformer models. The naive implementation materializes the full S = Q@K^T matrix (seq_len × seq_len), which is O(n^2) memory. A production implementation would use Flash Attention (tiling over seq dimension with online softmax), but for educational purposes we implement the straightforward version.\n\n"
            "Strategy:\n"
            "- Step 1: Compute S = Q @ K^T / sqrt(d_k). This is a batched matmul.\n"
            "- Step 2: Apply row-wise softmax to S.\n"
            "- Step 3: Compute O = softmax(S) @ V.\n"
            "- For simplicity, use triton kernels for steps 1 and 3 (matmul is the heavy part),\n"
            "  and use PyTorch softmax (which is already efficient) for step 2.\n"
            "- The key optimization opportunity is fusing softmax with the second matmul\n"
            "  (online softmax in Flash Attention), but that requires more complex tiling.\n"
            "- This simpler version is easier to verify and debug.\n"
            "</design>\n"
            "<code>\n"
            "import torch\n"
            "import triton\n"
            "import triton.language as tl\n"
            "import math\n\n"
            "@triton.jit\n"
            "def batched_gemm_kernel(\n"
            "    a_ptr, b_ptr, c_ptr,\n"
            "    M: int, N: int, K: int,\n"
            "    stride_am: int, stride_ak: int,\n"
            "    stride_bk: int, stride_bn: int,\n"
            "    stride_cm: int, stride_cn: int,\n"
            "    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,\n"
            "):\n"
            "    pid_m = tl.program_id(0)\n"
            "    pid_n = tl.program_id(1)\n"
            "    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)\n"
            "    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)\n"
            "    offs_k = tl.arange(0, BLOCK_K)\n"
            "    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak\n"
            "    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn\n"
            "    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)\n"
            "    for k in range(0, K, BLOCK_K):\n"
            "        a = tl.load(a_ptrs, mask=offs_k[None, :] < (K - k), other=0.0)\n"
            "        b = tl.load(b_ptrs, mask=offs_k[:, None] < (K - k), other=0.0)\n"
            "        acc += tl.dot(a, b)\n"
            "        a_ptrs += BLOCK_K * stride_ak\n"
            "        b_ptrs += BLOCK_K * stride_bk\n"
            "    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn\n"
            "    tl.store(c_ptrs, acc)\n\n\n"
            "def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:\n"
            "    batch, heads, seq_len, d_k = q.shape\n"
            "    scale = 1.0 / math.sqrt(d_k)\n\n"
            "    # Step 1: S = Q @ K^T / sqrt(d_k)\n"
            "    S = torch.empty(batch, heads, seq_len, seq_len, dtype=q.dtype, device=q.device)\n"
            "    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32\n"
            "    grid = (triton.cdiv(seq_len, BLOCK_M), triton.cdiv(seq_len, BLOCK_N))\n"
            "    for b in range(batch):\n"
            "        for h in range(heads):\n"
            "            batched_gemm_kernel[grid](\n"
            "                q[b, h], k[b, h].T.contiguous(), S[b, h],\n"
            "                seq_len, seq_len, d_k,\n"
            "                q.stride(1), q.stride(2),\n"
            "                1, k.stride(1),\n"
            "                S.stride(1), S.stride(2),\n"
            "                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,\n"
            "            )\n"
            "            S[b, h] *= scale\n\n"
            "    # Step 2: softmax\n"
            "    P = torch.softmax(S, dim=-1)\n\n"
            "    # Step 3: O = P @ V\n"
            "    O = torch.empty_like(q)\n"
            "    for b in range(batch):\n"
            "        for h in range(heads):\n"
            "            grid2 = (triton.cdiv(seq_len, BLOCK_M), triton.cdiv(d_k, BLOCK_N))\n"
            "            batched_gemm_kernel[grid2](\n"
            "                P[b, h], v[b, h], O[b, h],\n"
            "                seq_len, d_k, seq_len,\n"
            "                P.stride(1), P.stride(2),\n"
            "                v.stride(1), v.stride(2),\n"
            "                O.stride(1), O.stride(2),\n"
            "                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,\n"
            "            )\n"
            "    return O\n"
            "</code>\n"
            "<predict>\n"
            "Performance:\n"
            "- Step 1 (Q@K^T): (1024,64)×(64,1024) = 1024×1024 output, 2*1024*64*1024 = 134M FLOPs.\n"
            "- Step 2 (softmax): 1024×1024 = 1M elements, trivial compute.\n"
            "- Step 3 (P@V): same as Step 1: 134M FLOPs.\n"
            "- Total: ~270M FLOPs per head per batch. With 8 heads: ~2.2G FLOPs.\n"
            "- Expected runtime: ~0.5-1.0 ms on RTX 4090.\n"
            "- This naive implementation uses O(seq_len^2) memory for the S matrix (1M elements = 4MB per head).\n"
            "  For seq_len > 4096, this becomes prohibitive — Flash Attention is essential.\n"
            "- Speedup vs PyTorch eager: ~0.8-1.0x (PyTorch's SDPA already uses an optimized CUDA kernel).\n"
            "- Educational value: demonstrates batched GEMM in Triton, which is a building block.\n"
            "Confidence: medium (simplified version; real Flash Attention requires online softmax + tiling over seq dim).\n"
            "</predict>"
        ),
    },
]

# ============================================================================
# Write output
# ============================================================================

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for entry in EXAMPLES:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Wrote {len(EXAMPLES)} SFT examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
