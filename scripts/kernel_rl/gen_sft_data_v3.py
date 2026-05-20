"""
Generate clean SFT data v3 — hand-written Triton kernels, NO inductor patterns.

Each kernel is a canonical Triton implementation suitable for SFT training.
Covers element-wise, normalization, reduction, matmul, and fusion patterns.
Output: data/kernel_rl/sft_train_v3.jsonl
"""

from __future__ import annotations

import json
import os
import sys

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "kernel_rl", "sft_train_v3.jsonl"
)

# ── Shared format instructions ──────────────────────────────────────

FORMAT_INSTRUCTIONS = """Respond with exactly three sections in this order:
<design>
Optimization strategy. Be specific: which block size and why, what memory layout, what tiling strategy. If this is a revision round, start by briefly critiquing the PREVIOUS attempt.
</design>
<code>
Complete standalone Triton kernel: @triton.jit decorated kernel function
PLUS a wrapper function with explicit grid=(...) launch. NO torch._inductor
patterns, NO libdevice calls, NO empty_strided_cuda. Use standard Triton:
tl.load, tl.store, tl.arange, tl.program_id, tl.sum, tl.max, tl.math.erf, etc.
</code>
<predict>
Expected speedup vs PyTorch, bottleneck analysis, confidence level.
</predict>"""

HARD_RULES = """CRITICAL RULES — our automated benchmark will enforce these:
1. Write a STANDALONE Triton kernel with explicit grid launch.
   DO NOT call torch.nn.functional or any PyTorch op inside your kernel.
   DO NOT use torch._inductor patterns. DO NOT wrap a PyTorch call.
2. COMPILATION FAILURE → code reward = 0.
3. CORRECTNESS FAILURE → code reward = 0.
4. Slower than torch.compile → PENALTY. Faster → BONUS. 2× faster → MAX BONUS.
5. Your <predict> section MUST include an estimated speedup number vs PyTorch."""


def build_prompt(name, description, ref_python, input_shape, dtype="float32"):
    task_spec = json.dumps({"name": name, "description": description,
                            "input_shape": input_shape, "dtype": dtype})
    return (
        f"You are writing a high-performance Triton GPU kernel.\n\n"
        f"{HARD_RULES}\n\n"
        f"Task specification:\n{task_spec}\n\n"
        f"Reference PyTorch implementation (what you must beat):\n"
        f"```python\n{ref_python}\n```\n\n"
        f"{FORMAT_INSTRUCTIONS}"
    )


# ══════════════════════════════════════════════════════════════════════
# Canonical hand-written Triton kernels (NO inductor patterns)
# ══════════════════════════════════════════════════════════════════════

EXAMPLES = []

def add(name, desc, ref_py, shape, dtype, design, triton_code, predict):
    prompt = build_prompt(name, desc, ref_py, shape, dtype)
    completion = f"<design>\n{design}\n</design>\n<code>\n{triton_code}\n</code>\n<predict>\n{predict}\n</predict>"
    EXAMPLES.append({"prompt": prompt, "completion": completion})


# ── 1. ReLU ─────────────────────────────────────────────────────────

add(
    "relu", "ReLU activation: max(0, x)", "def relu(x):\n    return torch.relu(x)",
    [65536, 128], "float32",
    """Element-wise activation. Trivially memory-bound — every element is read once, written once.
Strategy: BLOCK_SIZE=1024 for good occupancy. Each program loads a contiguous block, applies max(0, x), stores result. No reduction or synchronization needed.
The real value of a custom ReLU kernel is fusion with the preceding layer (matmul/conv), saving one full memory round-trip.""",
    """import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(in_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(in_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.maximum(x, 0.0), mask=mask)

def relu(x):
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out""",
    "Speedup vs PyTorch eager: ~1.0x standalone (both are memory-bound). Speedup when fused: ~1.5-2.0x. Bottleneck: memory bandwidth. Confidence: very high."
)

# ── 2. GELU ─────────────────────────────────────────────────────────

add(
    "gelu", "GELU activation: x * 0.5 * (1 + erf(x/sqrt(2)))",
    "def gelu(x):\n    import torch.nn.functional as F\n    return F.gelu(x, approximate='none')",
    [65536, 256], "float32",
    """Element-wise activation with transcendental math (erf). Memory-bound like all element-wise ops.
Strategy: Use tl.math.erf() for the exact GELU formula. BLOCK_SIZE=1024 balances occupancy and launch overhead.
For applications that tolerate approximation, the tanh formulation is ~15% faster, but we use exact erf for correctness.
The computation is simple but the memory wall dominates — each element requires 1 read + 1 write = 8 bytes (fp32).""",
    """import torch
import triton
import triton.language as tl

@triton.jit
def gelu_kernel(in_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(in_ptr + offs, mask=mask)
    sqrt2 = 1.4142135623730951
    y = 0.5 * x * (1.0 + tl.math.erf(x / sqrt2))
    tl.store(out_ptr + offs, y, mask=mask)

def gelu(x):
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out""",
    "Speedup vs PyTorch: ~1.0x standalone. ~1.3-1.5x when fused with preceding matmul. Bottleneck: memory bandwidth. Confidence: high."
)

# ── 3. Swish / SiLU ─────────────────────────────────────────────────

add(
    "swish", "Swish (SiLU): x * sigmoid(x)",
    "def swish(x):\n    import torch.nn.functional as F\n    return F.silu(x)",
    [32768, 256], "float32",
    """Swish is x * sigmoid(x), widely used in LLMs (LLaMA, etc.). Element-wise and memory-bound.
Use tl.sigmoid() which maps to GPU SFU (Special Function Unit) for efficient transcendental computation.
BLOCK_SIZE=1024 standard for element-wise. The memory pattern is contiguous read + write.""",
    """import torch
import triton
import triton.language as tl

@triton.jit
def swish_kernel(in_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(in_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * tl.sigmoid(x), mask=mask)

def swish(x):
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    swish_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out""",
    "Speedup vs PyTorch: ~1.0x standalone. ~1.5x fused. Bottleneck: memory bandwidth. Confidence: very high."
)

# ── 4. LeakyReLU ────────────────────────────────────────────────────

add(
    "leaky_relu", "LeakyReLU: max(0, x) + neg_slope * min(0, x)",
    "def leaky_relu(x, neg_slope=0.01):\n    import torch.nn.functional as F\n    return F.leaky_relu(x, negative_slope=neg_slope)",
    [65536, 128], "float32",
    """LeakyReLU is element-wise. Use tl.where() for predicated execution (GPU handles divergence efficiently via predication, not branching).
BLOCK_SIZE=1024. Pass negative_slope as runtime argument (not constexpr) to allow tuning without recompilation.""",
    """import torch
import triton
import triton.language as tl

@triton.jit
def leaky_relu_kernel(in_ptr, out_ptr, n, neg_slope: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(in_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.where(x > 0, x, x * neg_slope), mask=mask)

def leaky_relu(x, neg_slope=0.01):
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    leaky_relu_kernel[grid](x, out, n, neg_slope, BLOCK_SIZE=BLOCK_SIZE)
    return out""",
    "Speedup vs PyTorch: ~1.0x standalone. ~1.5x fused. Bottleneck: memory bandwidth. Confidence: very high."
)

# ── 5. Sigmoid ──────────────────────────────────────────────────────

add(
    "sigmoid", "Sigmoid: 1 / (1 + exp(-x))",
    "def sigmoid(x):\n    return torch.sigmoid(x)",
    [32768, 128], "float32",
    """Sigmoid is element-wise. tl.sigmoid() is hardware-optimized via GPU SFU.
BLOCK_SIZE=1024. Extreme values are handled gracefully by the GPU's built-in sigmoid implementation.""",
    """import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_kernel(in_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(in_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.sigmoid(x), mask=mask)

def sigmoid(x):
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    sigmoid_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out""",
    "Speedup vs PyTorch: ~1.0x standalone. ~1.5x fused. Bottleneck: memory bandwidth. Confidence: very high."
)

# ── 6. Tanh ─────────────────────────────────────────────────────────

add(
    "tanh", "Hyperbolic tangent: tanh(x)",
    "def tanh(x):\n    return torch.tanh(x)",
    [32768, 256], "float32",
    """Tanh is element-wise. Use tl.math.tanh() for hardware-accelerated tanh on GPU SFU.
BLOCK_SIZE=1024. No numerical issues — the GPU implementation handles extreme values internally.""",
    """import torch
import triton
import triton.language as tl

@triton.jit
def tanh_kernel(in_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(in_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.math.tanh(x), mask=mask)

def tanh(x):
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    tanh_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out""",
    "Speedup vs PyTorch: ~1.0x standalone. ~1.5x fused. Bottleneck: memory bandwidth. Confidence: very high."
)

# ── 7. ELU ──────────────────────────────────────────────────────────

add(
    "elu", "ELU: x if x>0 else alpha*(exp(x)-1)",
    "def elu(x, alpha=1.0):\n    import torch.nn.functional as F\n    return F.elu(x, alpha=alpha)",
    [65536, 128], "float32",
    """ELU is element-wise with a conditional: exp(x)-1 for negative values. Use tl.where() for predication.
BLOCK_SIZE=1024. The exp() call uses GPU SFU. Alpha is a runtime parameter for flexibility.""",
    """import torch
import triton
import triton.language as tl

@triton.jit
def elu_kernel(in_ptr, out_ptr, n, alpha: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(in_ptr + offs, mask=mask)
    y = tl.where(x > 0, x, alpha * (tl.exp(x) - 1.0))
    tl.store(out_ptr + offs, y, mask=mask)

def elu(x, alpha=1.0):
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    elu_kernel[grid](x, out, n, alpha, BLOCK_SIZE=BLOCK_SIZE)
    return out""",
    "Speedup vs PyTorch: ~1.0x standalone. ~1.5x fused. Bottleneck: memory bandwidth. Confidence: high."
)

# ── 8. RMSNorm ──────────────────────────────────────────────────────

add(
    "rms_norm", "RMS Normalization: y = x / sqrt(mean(x^2) + eps), normalize along last dim",
    "def rms_norm(x, eps=1e-5):\n    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)\n    return x / rms",
    [256, 1024], "float32",
    """RMSNorm requires a reduction (sum of squares) followed by element-wise normalize.
Strategy: Each program handles one row. Loop over the feature dimension in blocks, accumulating sum of x^2 in a register.
After the loop: rms = sqrt(sum/N + eps). Then normalize: output = x / rms.
BLOCK_SIZE=128 balances register usage vs loop iterations. For larger feature dims (>4096), consider a two-pass approach with shared memory reduction.
This is the MOST COMMON fusion target in LLMs — fusing RMSNorm with residual add saves 40% memory bandwidth.""",
    """import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(in_ptr, out_ptr, n_rows, n_cols, eps: float, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    sq_sum = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        x = tl.load(in_ptr + row_start + offs, mask=mask, other=0.0)
        sq_sum += tl.sum(x * x, axis=0)
    rms = tl.sqrt(sq_sum / n_cols + eps)
    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        x = tl.load(in_ptr + row_start + offs, mask=mask, other=0.0)
        tl.store(out_ptr + row_start + offs, x / rms, mask=mask)

def rms_norm(x, eps=1e-5):
    n_rows = x.numel() // x.shape[-1]
    n_cols = x.shape[-1]
    out = torch.empty_like(x)
    BLOCK_SIZE = 128
    grid = (n_rows,)
    rmsnorm_kernel[grid](x, out, n_rows, n_cols, eps, BLOCK_SIZE=BLOCK_SIZE)
    return out""",
    "Speedup vs PyTorch: ~1.2-1.5x standalone. ~1.8-2.0x when fused with residual add. Bottleneck: reduction synchronisation dominates for small n_cols, memory bandwidth for large. Confidence: high."
)

# ── 9. LayerNorm ────────────────────────────────────────────────────

add(
    "layer_norm", "Layer Normalization: (x-mean)/sqrt(var+eps) * gamma + beta",
    "def layer_norm(x, gamma, beta, eps=1e-5):\n    import torch.nn.functional as F\n    return F.layer_norm(x, gamma.shape, gamma, beta, eps)",
    [256, 1024], "float32",
    """LayerNorm is a two-pass reduction + affine transform. Welford's online algorithm for numerical stability.
Strategy: Each program handles one row. Pass 1 accumulates sum(x) and sum(x^2). Compute mean and var.
Pass 2 normalizes and applies gamma/beta. BLOCK_SIZE=256.
For small feature dims, this single-program-per-row approach is efficient. For large dims, multi-program reduction with shared memory is better.
Main optimization: fuse with preceding Linear layer to avoid intermediate tensor allocation.""",
    """import torch
import triton
import triton.language as tl

@triton.jit
def layernorm_kernel(in_ptr, gamma_ptr, beta_ptr, out_ptr, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    mean_sum = 0.0
    var_sum = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        x = tl.load(in_ptr + row_start + offs, mask=mask, other=0.0)
        mean_sum += tl.sum(x, axis=0)
        var_sum += tl.sum(x * x, axis=0)
    mean = mean_sum / n_cols
    var = var_sum / n_cols - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)
    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        x = tl.load(in_ptr + row_start + offs, mask=mask, other=0.0)
        gamma = tl.load(gamma_ptr + offs, mask=mask, other=1.0)
        beta = tl.load(beta_ptr + offs, mask=mask, other=0.0)
        y = (x - mean) * inv_std * gamma + beta
        tl.store(out_ptr + row_start + offs, y, mask=mask)

def layer_norm(x, gamma, beta, eps=1e-5):
    n_rows = x.numel() // x.shape[-1]
    n_cols = x.shape[-1]
    out = torch.empty_like(x)
    BLOCK_SIZE = 256
    grid = (n_rows,)
    layernorm_kernel[grid](x, gamma, beta, out, n_cols, eps, BLOCK_SIZE=BLOCK_SIZE)
    return out""",
    "Speedup vs PyTorch: ~1.2-1.5x standalone. ~1.8-2.0x fused with Linear. Bottleneck: reduction for small dims, memory bandwidth for large. Confidence: high."
)

# ── 10. Softmax ─────────────────────────────────────────────────────

add(
    "softmax", "Row-wise softmax along last dim",
    "def softmax(x):\n    import torch.nn.functional as F\n    return F.softmax(x, dim=-1)",
    [4096, 4096], "float32",
    """Row-wise softmax with numerical stability (subtract max before exp). Three-pass algorithm:
Pass 1: find row-wise max. Pass 2: compute sum of exp(x-max). Pass 3: normalize and write.
Each program handles one complete row, iterating in BLOCK_SIZE chunks.
Avoids materializing the full exp(x) intermediate — each element is read 3 times but only 1 write.
For very large rows (>16384), consider online softmax (Flash Attention style) for better locality.""",
    """import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(in_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    row_max = float('-inf')
    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        x = tl.load(in_ptr + row_start + offs, mask=mask, other=float('-inf'))
        row_max = tl.maximum(row_max, tl.max(x, axis=0))
    exp_sum = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        x = tl.load(in_ptr + row_start + offs, mask=mask, other=float('-inf'))
        exp_sum += tl.sum(tl.exp(x - row_max), axis=0)
    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        x = tl.load(in_ptr + row_start + offs, mask=mask, other=float('-inf'))
        tl.store(out_ptr + row_start + offs, tl.exp(x - row_max) / exp_sum, mask=mask)

def softmax(x):
    n_rows = x.numel() // x.shape[-1]
    n_cols = x.shape[-1]
    out = torch.empty_like(x)
    BLOCK_SIZE = 128
    grid = (n_rows,)
    softmax_kernel[grid](x, out, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out""",
    "Speedup vs PyTorch: ~1.5-2.0x for (4096,4096) matrices. The three-pass algorithm re-reads each element 3x but avoids intermediate tensor materialization which saves significant memory bandwidth. Bottleneck: memory bandwidth for large rows. Confidence: medium-high."
)

# ── 11. Row-wise Sum ────────────────────────────────────────────────

add(
    "row_sum", "Sum along last dim: out[i] = sum(input[i,:])",
    "def row_sum(x):\n    return x.sum(dim=-1)",
    [8192, 512], "float32",
    """Row-wise sum reduction. Each program handles one row, accumulating in a register across blocks.
BLOCK_SIZE=256. At the end, write a single scalar per row. For very long rows (>10K elements), a tree reduction with shared memory would be more efficient, but for moderate sizes the simple accumulate is sufficient.
The key optimization opportunity: fusion with preceding element-wise op saves writing the intermediate tensor.""",
    """import torch
import triton
import triton.language as tl

@triton.jit
def row_sum_kernel(in_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    total = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        x = tl.load(in_ptr + row_start + offs, mask=mask, other=0.0)
        total += tl.sum(x, axis=0)
    tl.store(out_ptr + row_idx, total)

def row_sum(x):
    n_rows = x.numel() // x.shape[-1]
    n_cols = x.shape[-1]
    out = torch.empty(n_rows, dtype=x.dtype, device=x.device)
    BLOCK_SIZE = 256
    grid = (n_rows,)
    row_sum_kernel[grid](x, out, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out""",
    "Speedup vs PyTorch: ~1.0x standalone. ~1.5x fused with preceding op. Bottleneck: memory bandwidth. Confidence: high."
)

# ── 12. Row-wise Mean ───────────────────────────────────────────────

add(
    "row_mean", "Mean along last dim: out[i] = mean(input[i,:])",
    "def row_mean(x):\n    return x.mean(dim=-1)",
    [8192, 512], "float32",
    """Row-wise mean is sum divided by n_cols. Same pattern as row_sum, with a final division.
After accumulating sum across blocks, divide by n_cols and write. BLOCK_SIZE=256.""",
    """import torch
import triton
import triton.language as tl

@triton.jit
def row_mean_kernel(in_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    total = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        x = tl.load(in_ptr + row_start + offs, mask=mask, other=0.0)
        total += tl.sum(x, axis=0)
    tl.store(out_ptr + row_idx, total / n_cols)

def row_mean(x):
    n_rows = x.numel() // x.shape[-1]
    n_cols = x.shape[-1]
    out = torch.empty(n_rows, dtype=x.dtype, device=x.device)
    BLOCK_SIZE = 256
    grid = (n_rows,)
    row_mean_kernel[grid](x, out, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out""",
    "Speedup vs PyTorch: ~1.0x standalone. ~1.5x fused. Bottleneck: memory bandwidth. Confidence: high."
)

# ── 13. Argmax ─────────────────────────────────────────────────────

add(
    "argmax", "Argmax along last dim: returns index of max per row",
    "def argmax(x):\n    return x.argmax(dim=-1)",
    [4096, 1024], "float32",
    """Argmax tracks both the maximum value and its index. Each program handles one row.
Iterate over blocks: find local max + local argmax. Update global max if local is larger.
On ties, the first index wins (matching torch behavior). BLOCK_SIZE=128.""",
    """import torch
import triton
import triton.language as tl

@triton.jit
def argmax_kernel(in_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    max_val = float('-inf')
    max_idx = -1
    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        x = tl.load(in_ptr + row_start + offs, mask=mask, other=float('-inf'))
        local_max = tl.max(x, axis=0)
        local_argmax = tl.argmax(x, axis=0)
        if local_max > max_val:
            max_val = local_max
            max_idx = start + local_argmax
    tl.store(out_ptr + row_idx, max_idx)

def argmax(x):
    n_rows = x.numel() // x.shape[-1]
    n_cols = x.shape[-1]
    out = torch.empty(n_rows, dtype=torch.int64, device=x.device)
    BLOCK_SIZE = 128
    grid = (n_rows,)
    argmax_kernel[grid](x, out, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out""",
    "Speedup vs PyTorch: ~1.0-1.2x. Bottleneck: reduction with branching. Confidence: medium."
)

# ── 14. Tiled Matmul ────────────────────────────────────────────────

add(
    "matmul_tiled", "Matrix multiply C = A @ B with tiling. A(M,K), B(K,N)",
    "def matmul(a, b):\n    return a @ b",
    {"A": [1024, 1024], "B": [1024, 1024]}, "float32",
    """Tiled matrix multiplication is compute-bound. The key optimization hierarchy:
1. Tiling: decompose MxKxN into small blocks (BLOCK_M x BLOCK_K of A, BLOCK_K x BLOCK_N of B) that fit in registers and shared memory.
2. Shared memory: cache tiles of A and B in SRAM (much higher bandwidth than HBM).
3. tl.dot(): maps to Tensor Cores on NVIDIA GPUs, providing ~4-8x throughput over FP32 CUDA cores.
4. Register accumulation: partial dot products accumulate in registers, writes once at end.
Block sizes: BLOCK_M=128, BLOCK_N=128, BLOCK_K=32. These balance shared memory usage (128*32*2*4 = 32KB per tile) against SM occupancy.
Grid is 2D: (M/BLOCK_M, N/BLOCK_N) programs. Each program computes one 128x128 tile of C.""",
    """import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def matmul(a, b):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](a, b, c, M, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return c""",
    "Speedup vs PyTorch eager: ~0.5-0.8x (cuBLAS is heavily optimized for matmul). However, vs naive triple-loop: >100x. The educational value of this kernel is in demonstrating tiling, shared memory, and tl.dot for Tensor Core usage. For production: use torch.matmul or cuBLAS. For fusion: the tiled structure enables fusing the matmul with a subsequent activation (e.g., matmul+ReLU) in a single kernel, which can give 1.3-1.5x speedup over separate launches. Bottleneck: compute. Confidence: medium."
)

# ── 15. MSE Loss ────────────────────────────────────────────────────

add(
    "mse_loss", "Mean Squared Error: mean((pred - target)^2)",
    "def mse_loss(pred, target):\n    import torch.nn.functional as F\n    return F.mse_loss(pred, target)",
    [65536, 128], "float32",
    """MSE loss is element-wise subtraction + square + global mean. The global reduction requires synchronization.
Strategy: each program computes partial sum of squared diffs, then uses atomic add to a global accumulator.
After all programs finish, divide by N on host (single scalar read).
BLOCK_SIZE=1024. The atomic contention is acceptable for moderate grid sizes.""",
    """import torch
import triton
import triton.language as tl

@triton.jit
def mse_kernel(pred_ptr, target_ptr, sum_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    p = tl.load(pred_ptr + offs, mask=mask, other=0.0)
    t = tl.load(target_ptr + offs, mask=mask, other=0.0)
    diff = p - t
    tl.atomic_add(sum_ptr, tl.sum(diff * diff, axis=0))

def mse_loss(pred, target):
    n = pred.numel()
    total = torch.zeros(1, dtype=pred.dtype, device=pred.device)
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    mse_kernel[grid](pred, target, total, n, BLOCK_SIZE=BLOCK_SIZE)
    return total / n""",
    "Speedup vs PyTorch: ~0.9-1.0x standalone (PyTorch's reduction is already optimized). ~1.3x when fused with preceding prediction layer. Bottleneck: atomic contention for large grids. Confidence: medium."
)

# ── 16. Fused GELU + Mul ────────────────────────────────────────────

add(
    "fused_gelu_mul", "Fused GELU(x) * scale. Demonstrate kernel fusion.",
    "def fused_gelu_mul(x, scale):\n    import torch.nn.functional as F\n    return F.gelu(x) * scale",
    [65536, 128], "float32",
    """This demonstrates kernel fusion: combining GELU and scalar multiply into one kernel.
Without fusion: kernel1 reads X → writes GELU(X) intermediate, kernel2 reads intermediate → writes GELU(X)*scale.
With fusion: read X → compute GELU → compute multiply → write. Saves one full memory round-trip (33% bandwidth).
This pattern generalizes to any element-wise sequence. BLOCK_SIZE=1024 standard for element-wise.
The compute is unchanged; the win is purely from reduced memory traffic (8 bytes/el vs 16 bytes/el).""",
    """import torch
import triton
import triton.language as tl

@triton.jit
def fused_gelu_mul_kernel(in_ptr, out_ptr, n, scale, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(in_ptr + offs, mask=mask)
    sqrt2 = 1.4142135623730951
    gelu_x = 0.5 * x * (1.0 + tl.math.erf(x / sqrt2))
    tl.store(out_ptr + offs, gelu_x * scale, mask=mask)

def fused_gelu_mul(x, scale):
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    fused_gelu_mul_kernel[grid](x, out, n, scale, BLOCK_SIZE=BLOCK_SIZE)
    return out""",
    "Speedup vs unfused: ~1.8-2.0x (memory bandwidth savings: 8 bytes/el vs 16 bytes/el). This is the single most impactful pattern for element-wise ops. Bottleneck: memory bandwidth. Confidence: very high."
)

# ── 17. Fused RMSNorm + Residual ────────────────────────────────────

add(
    "fused_rmsnorm_residual", "Fused: y = RMSNorm(x + residual). Most common LLM fusion.",
    "def fused_rmsnorm_residual(x, residual, eps=1e-5):\n    y = x + residual\n    rms = torch.sqrt(torch.mean(y**2, dim=-1, keepdim=True) + eps)\n    return y / rms",
    [256, 1024], "float32",
    """RMSNorm(x + residual) is the single most common fusion in transformer blocks (used in LLaMA, Gemma, etc.).
Without fusion: 3 kernels (add → square+reduce → normalize). With fusion: 1 kernel.
Fuse the residual add into the reduction pass: load x and residual, compute y = x + residual, accumulate y^2 sum.
After loop: rms = sqrt(sum/N + eps). Then normalize: output = y / rms.
Bandwidth savings: ~40% (1 read x + 1 read residual + 1 write vs 2 reads + 2 writes unfused).
BLOCK_SIZE=128. For large feature dims (>4096), use two-pass: compute rms in first pass, normalize in second.""",
    """import torch
import triton
import triton.language as tl

@triton.jit
def fused_rmsnorm_residual_kernel(x_ptr, res_ptr, out_ptr, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    sq_sum = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        x = tl.load(x_ptr + row_start + offs, mask=mask, other=0.0)
        r = tl.load(res_ptr + row_start + offs, mask=mask, other=0.0)
        y = x + r
        sq_sum += tl.sum(y * y, axis=0)
        tl.store(out_ptr + row_start + offs, y, mask=mask)
    rms = tl.sqrt(sq_sum / n_cols + eps)
    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        y = tl.load(out_ptr + row_start + offs, mask=mask, other=0.0)
        tl.store(out_ptr + row_start + offs, y / rms, mask=mask)

def fused_rmsnorm_residual(x, residual, eps=1e-5):
    n_rows = x.numel() // x.shape[-1]
    n_cols = x.shape[-1]
    out = torch.empty_like(x)
    BLOCK_SIZE = 128
    grid = (n_rows,)
    fused_rmsnorm_residual_kernel[grid](x, residual, out, n_cols, eps, BLOCK_SIZE=BLOCK_SIZE)
    return out""",
    "Speedup vs unfused: ~1.5-1.8x. This is the most impactful single fusion for LLM inference. Bottleneck: memory bandwidth. Confidence: high."
)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for entry in EXAMPLES:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Wrote {len(EXAMPLES)} clean hand-written Triton SFT examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
