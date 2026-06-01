"""
Build seed tasks for RL training — Levels 1, 2, 3 from KernelBench patterns.

Level 1 (single kernel, element-wise + simple reduction):  10 tasks
Level 2 (reduction + normalization + fusion):              8  tasks
Level 3 (matmul + attention + complex fusion):             6  tasks
Total: 24 tasks

Each task: task_spec (name, desc, shape, dtype, constraints) +
           bench_spec (input_gen matching ref_fn signature, target_speedup, correctness) +
           reference_python (clean PyTorch) +
           extra_info (index, difficulty, category)

CRITICAL: input_gen param count MUST match ref_fn signature.
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "kernel_rl", "train.parquet"
)

# ══════════════════════════════════════════════════════════════════════
# Task definitions
# Each: (name, description, reference_python, input_shape, dtype,
#         input_gen_str, target_speedup, correctness, rtol, constraints,
#         difficulty, category)
# ══════════════════════════════════════════════════════════════════════

TASKS = []


def _add(name, desc, ref_py, shape, dtype, input_gen, target_sp, correctness,
         rtol, constraints, difficulty, category):
    TASKS.append({
        "name": name, "desc": desc, "ref_py": ref_py, "shape": shape,
        "dtype": dtype, "input_gen": input_gen, "target_speedup": target_sp,
        "correctness": correctness, "rtol": rtol, "constraints": constraints,
        "difficulty": difficulty, "category": category,
    })


# ══════════════════════════════════════════════════════════════════════
# Level 1 — Single kernel: element-wise ops + simple reductions (10 tasks)
# ══════════════════════════════════════════════════════════════════════

# 1. vector_add (FIXED: input_gen now has 2 tensors matching f(a,b))
_add(
    "vector_add", "Element-wise vector addition C = A + B",
    "def vector_add(a, b):\n    return a + b",
    [65536], "float32",
    "torch.randn(65536, dtype=torch.float32), torch.randn(65536, dtype=torch.float32)",
    2.0, "exact_match", None,
    "preserve numerical precision, handle arbitrary alignment",
    "easy", "elementwise",
)

# 2. gelu_activation (OK: 1 param)
_add(
    "gelu_activation", "GELU activation: x * 0.5 * (1 + erf(x/sqrt(2)))",
    "def gelu(x):\n    import torch.nn.functional as F\n    return F.gelu(x, approximate='none')",
    [65536], "float32",
    "torch.randn(65536, dtype=torch.float32)",
    1.5, "exact_match", None,
    "bitwise identical to F.gelu(approximate='none')",
    "easy", "activation",
)

# 3. relu
_add(
    "relu", "ReLU activation: max(0, x)",
    "def relu(x):\n    return torch.relu(x)",
    [65536], "float32",
    "torch.randn(65536, dtype=torch.float32)",
    1.5, "exact_match", None,
    "handle both positive and negative values correctly",
    "easy", "activation",
)

# 4. sigmoid
_add(
    "sigmoid", "Sigmoid activation: 1 / (1 + exp(-x))",
    "def sigmoid(x):\n    return torch.sigmoid(x)",
    [32768], "float32",
    "torch.randn(32768, dtype=torch.float32)",
    1.5, "approx_match", 1e-4,
    "numerically stable for large |x|, no NaN for extreme values",
    "easy", "activation",
)

# 5. tanh
_add(
    "tanh", "Hyperbolic tangent: tanh(x)",
    "def tanh(x):\n    return torch.tanh(x)",
    [32768], "float32",
    "torch.randn(32768, dtype=torch.float32)",
    1.5, "approx_match", 1e-4,
    "numerically stable, saturates to ±1 for extreme values",
    "easy", "activation",
)

# 6. swish
_add(
    "swish", "Swish (SiLU): x * sigmoid(x)",
    "def swish(x):\n    import torch.nn.functional as F\n    return F.silu(x)",
    [32768], "float32",
    "torch.randn(32768, dtype=torch.float32)",
    1.5, "approx_match", 1e-4,
    "smooth non-monotonic activation, widely used in LLMs",
    "easy", "activation",
)

# 7. leaky_relu
_add(
    "leaky_relu", "LeakyReLU: max(0, x) + neg_slope * min(0, x)",
    "def leaky_relu(x, neg_slope=0.01):\n    import torch.nn.functional as F\n    return F.leaky_relu(x, negative_slope=neg_slope)",
    [65536], "float32",
    "torch.randn(65536, dtype=torch.float32)",
    1.5, "exact_match", None,
    "default neg_slope=0.01, pass-through test with single input",
    "easy", "activation",
)

# 8. elu
_add(
    "elu", "ELU: x if x>0 else alpha*(exp(x)-1)",
    "def elu(x, alpha=1.0):\n    import torch.nn.functional as F\n    return F.elu(x, alpha=alpha)",
    [65536], "float32",
    "torch.randn(65536, dtype=torch.float32)",
    1.5, "approx_match", 1e-4,
    "default alpha=1.0, smooth exponential for negative values",
    "easy", "activation",
)

# 9. row_sum
_add(
    "row_sum", "Sum along last dim: out[i] = sum(input[i,:])",
    "def row_sum(x):\n    return x.sum(dim=-1)",
    [8192, 512], "float32",
    "torch.randn(8192, 512, dtype=torch.float32)",
    1.8, "approx_match", 1e-4,
    "reduction over last dimension, output shape (N,) for input (N,D)",
    "easy", "reduction",
)

# 10. row_mean
_add(
    "row_mean", "Mean along last dim: out[i] = mean(input[i,:])",
    "def row_mean(x):\n    return x.mean(dim=-1)",
    [8192, 512], "float32",
    "torch.randn(8192, 512, dtype=torch.float32)",
    1.8, "approx_match", 1e-4,
    "reduction over last dimension, output shape (N,) for input (N,D)",
    "easy", "reduction",
)

# ══════════════════════════════════════════════════════════════════════
# Level 2 — Reduction + normalization + fusion patterns (8 tasks)
# ══════════════════════════════════════════════════════════════════════

# 11. softmax (OK: 1 param)
_add(
    "softmax", "Row-wise softmax along last dimension",
    "def softmax(x):\n    import torch.nn.functional as F\n    return F.softmax(x, dim=-1)",
    [4096, 4096], "float32",
    "torch.randn(4096, 4096, dtype=torch.float32) * 10",
    2.0, "approx_match", 1e-4,
    "numerically stable (subtract max before exp), rows sum to 1.0",
    "medium", "reduction",
)

# 12. layer_norm (FIXED: input_gen now has 3 tensors matching f(x, gamma, beta))
_add(
    "layer_norm", "Layer normalization over last dim with affine params",
    "def layer_norm(x, gamma, beta, eps=1e-5):\n    import torch.nn.functional as F\n    return F.layer_norm(x, gamma.shape, gamma, beta, eps)",
    [256, 1024], "float32",
    "torch.randn(256, 1024, dtype=torch.float32), torch.ones(1024, dtype=torch.float32), torch.zeros(1024, dtype=torch.float32)",
    2.0, "approx_match", 1e-4,
    "eps=1e-5, normalized output shape matches input, gamma=ones, beta=zeros",
    "medium", "normalization",
)

# 13. rms_norm
_add(
    "rms_norm", "RMS Normalization: y = x / sqrt(mean(x^2) + eps)",
    "def rms_norm(x, eps=1e-5):\n    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)\n    return x / rms",
    [256, 1024], "float32",
    "torch.randn(256, 1024, dtype=torch.float32)",
    2.0, "approx_match", 1e-4,
    "normalize along last dim, output has same shape as input",
    "medium", "normalization",
)

# 14. argmax
_add(
    "argmax", "Argmax along last dim: index of max value per row",
    "def argmax(x):\n    return x.argmax(dim=-1)",
    [4096, 1024], "float32",
    "torch.randn(4096, 1024, dtype=torch.float32)",
    1.8, "approx_match", None,
    "output dtype int64, shape (N,), first index on ties",
    "medium", "reduction",
)

# 15. mse_loss
_add(
    "mse_loss", "Mean Squared Error: mean((pred - target)^2)",
    "def mse_loss(pred, target):\n    import torch.nn.functional as F\n    return F.mse_loss(pred, target)",
    [65536], "float32",
    "torch.randn(65536, dtype=torch.float32), torch.randn(65536, dtype=torch.float32)",
    1.8, "approx_match", 1e-4,
    "output is a scalar (0-d tensor), element-wise squared diff + global mean",
    "medium", "reduction",
)

# 16. fused_gelu_mul
_add(
    "fused_gelu_mul", "Fused GELU(x) * scale — demonstrate kernel fusion",
    "def fused_gelu_mul(x, scale):\n    import torch.nn.functional as F\n    return F.gelu(x) * scale",
    [65536], "float32",
    "torch.randn(65536, dtype=torch.float32), torch.tensor(2.0)",
    2.0, "approx_match", 1e-4,
    "fuse gelu + multiply into single kernel, same shape as input",
    "medium", "fusion",
)

# 17. fused_rmsnorm_residual
_add(
    "fused_rmsnorm_residual", "Fused: y = RMSNorm(x + residual) — most common LLM fusion",
    "def fused_rmsnorm_residual(x, residual, eps=1e-5):\n    y = x + residual\n    rms = torch.sqrt(torch.mean(y ** 2, dim=-1, keepdim=True) + eps)\n    return y / rms",
    [256, 1024], "float32",
    "torch.randn(256, 1024, dtype=torch.float32), torch.randn(256, 1024, dtype=torch.float32)",
    2.0, "approx_match", 1e-4,
    "add residual then normalize, output same shape as input",
    "medium", "fusion",
)

# 18. batch_norm_forward
_add(
    "batch_norm_forward", "BatchNorm forward: (x-running_mean)/sqrt(running_var+eps)*gamma+beta",
    "def batch_norm(x, gamma, beta, running_mean, running_var, eps=1e-5):\n    import torch.nn.functional as F\n    return F.batch_norm(x, running_mean, running_var, weight=gamma, bias=beta, training=False, eps=eps)",
    [32, 128, 64], "float32",
    "torch.randn(32, 128, 64, dtype=torch.float32), torch.ones(128, dtype=torch.float32), torch.zeros(128, dtype=torch.float32), torch.zeros(128, dtype=torch.float32), torch.ones(128, dtype=torch.float32)",
    2.0, "approx_match", 1e-4,
    "2D input (N,C,H), normalize per-channel, running_mean=0 running_var=1 gamma=1 beta=0",
    "medium", "normalization",
)

# ══════════════════════════════════════════════════════════════════════
# Level 3 — Matmul + attention + complex fusion (6 tasks)
# ══════════════════════════════════════════════════════════════════════

# 19. matmul_naive (OK: 2 params)
_add(
    "matmul_naive", "Matrix multiplication C = A @ B, (M,K) x (K,N)",
    "def matmul(a, b):\n    return a @ b",
    {"A": [1024, 1024], "B": [1024, 1024]}, "float32",
    "torch.randn(1024, 1024, dtype=torch.float32), torch.randn(1024, 1024, dtype=torch.float32)",
    3.0, "approx_match", 1e-3,
    "no padding assumptions, all dimensions may vary, C.shape = (M,N)",
    "hard", "gemm",
)

# 20. attention_score (FIXED: input_gen now has 3 valid tensors)
_add(
    "attention_score", "Scaled dot-product attention: softmax(QK^T/sqrt(d)) @ V",
    "def attention_score(q, k, v, mask=None):\n    import torch.nn.functional as F\n    scale = q.shape[-1] ** 0.5\n    attn = F.softmax((q @ k.transpose(-2, -1)) / scale, dim=-1)\n    if mask is not None:\n        attn = attn + mask\n    return attn @ v",
    {"batch": 4, "heads": 8, "seq_len": 128, "head_dim": 64}, "float32",
    "torch.randn(4, 8, 128, 64, dtype=torch.float32), torch.randn(4, 8, 128, 64, dtype=torch.float32), torch.randn(4, 8, 128, 64, dtype=torch.float32)",
    2.5, "approx_match", 1e-3,
    "support optional causal mask (mask=None for this task), output shape (B,H,S,d)",
    "hard", "attention",
)

# 21. matmul_tiled
_add(
    "matmul_tiled", "Tiled matrix multiplication with configurable tile sizes",
    "def matmul_tiled(a, b):\n    return a @ b",
    {"A": [2048, 1024], "B": [1024, 2048]}, "float32",
    "torch.randn(2048, 1024, dtype=torch.float32), torch.randn(1024, 2048, dtype=torch.float32)",
    3.0, "approx_match", 1e-3,
    "A(M,K) x B(K,N) = C(M,N), must use tiling (TL) with shared memory, no cuBLAS fallback",
    "hard", "gemm",
)

# 22. scaled_dot_product_attention (full sdpa)
_add(
    "sdpa_full", "Full scaled dot-product attention: softmax(QK^T/sqrt(d) + mask) @ V",
    "def sdpa_full(q, k, v, mask=None):\n    import torch.nn.functional as F\n    scale = q.shape[-1] ** 0.5\n    attn = F.softmax((q @ k.transpose(-2, -1)) / scale, dim=-1)\n    if mask is not None:\n        attn = attn + mask\n    return attn @ v",
    {"batch": 2, "heads": 16, "seq_len": 256, "head_dim": 64}, "float32",
    "torch.randn(2, 16, 256, 64, dtype=torch.float32), torch.randn(2, 16, 256, 64, dtype=torch.float32), torch.randn(2, 16, 256, 64, dtype=torch.float32)",
    3.0, "approx_match", 1e-3,
    "multi-head attention without mask, QK^T produces (B,H,S,S) intermediate",
    "hard", "attention",
)

# 23. fused_linear_gelu
_add(
    "fused_linear_gelu", "Fused Linear + GELU: GELU(x @ W^T + b)",
    "def fused_linear_gelu(x, weight, bias):\n    import torch.nn.functional as F\n    return F.gelu(F.linear(x, weight, bias))",
    {"N": 512, "in_features": 1024, "out_features": 2048}, "float32",
    "torch.randn(512, 1024, dtype=torch.float32), torch.randn(2048, 1024, dtype=torch.float32), torch.randn(2048, dtype=torch.float32)",
    3.0, "approx_match", 1e-3,
    "fuse matmul + bias + GELU in single kernel, save 2 intermediate tensors",
    "hard", "fusion",
)

# 24. fused_matmul_bias_relu
_add(
    "fused_matmul_bias_relu", "Fused Matmul + Bias + ReLU: ReLU(A @ B + bias)",
    "def fused_matmul_bias_relu(a, b, bias):\n    return torch.relu(a @ b + bias)",
    {"M": 512, "K": 1024, "N": 2048}, "float32",
    "torch.randn(512, 1024, dtype=torch.float32), torch.randn(1024, 2048, dtype=torch.float32), torch.randn(2048, dtype=torch.float32)",
    3.0, "approx_match", 1e-3,
    "fuse matmul + bias add + ReLU in single kernel, (M,K)x(K,N)→(M,N)",
    "hard", "fusion",
)


# ══════════════════════════════════════════════════════════════════════
# Build parquet
# ══════════════════════════════════════════════════════════════════════

def main():
    rows = []
    for i, t in enumerate(TASKS):
        index_str = f"kernel_{i:02d}"

        task_spec = json.dumps({
            "name": t["name"],
            "description": t["desc"],
            "input_shape": t["shape"],
            "dtype": t["dtype"],
            "constraints": t["constraints"],
        })

        bench = {
            "metric": "runtime",
            "target_speedup": t["target_speedup"],
            "correctness": t["correctness"],
            "input_gen": t["input_gen"],
        }
        if t["rtol"] is not None:
            bench["rtol"] = t["rtol"]
        bench_spec = json.dumps(bench)

        extra_info = json.dumps({
            "index": index_str,
            "difficulty": t["difficulty"],
            "category": t["category"],
        })

        rows.append({
            "task_spec": task_spec,
            "bench_spec": bench_spec,
            "reference_python": t["ref_py"],
            "extra_info": extra_info,
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Wrote {len(rows)} seed tasks to {OUTPUT_PATH}")
    print()
    for t in TASKS:
        print(f"  {t['name']:30s}  L{'123'[['easy','medium','hard'].index(t['difficulty'])]}  {t['category']:15s}  {t['difficulty']}")


if __name__ == "__main__":
    main()
