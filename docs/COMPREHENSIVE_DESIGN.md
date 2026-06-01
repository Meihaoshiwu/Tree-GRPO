# Tree-GRPO Kernel RL — 完整设计方案

## 1. 项目概述

### 1.1 目标

训练 **Qwen2.5-Coder-7B** 模型写出**击败 `torch.compile` 性能**的 Triton GPU kernel。

核心思路：用树搜索（BFS, depth=2）组织多轮迭代改进，每轮由 scorer 提供客观反馈（编译、正确性、性能），用 GRPO + Generational Advantage 驱动 RL 训练。

### 1.2 技术栈

| 组件 | 选型 | 原因 |
|------|------|------|
| 基座模型 | Qwen2.5-Coder-7B-Instruct | 代码能力最强的 7B 模型 |
| RL 框架 | veRL (hybrid engine) | vLLM 推理 + FSDP 训练共享 GPU，无需手动切换 |
| 推理引擎 | vLLM v0.8.5 (V1) | PagedAttention KV cache，前缀缓存 |
| 训练策略 | FSDP full_shard (4 GPU) | 全参数微调 |
| 树搜索 | BFS level-wise | 同深度节点组 batch，vLLM 原生 n>1 分支 |
| 评测基线 | KernelBench (ICML'25) | 标准化 Triton kernel 评测流程 |

### 1.3 核心设计决策

- **三段式输出**（design/code/predict）：互不耦合，各自独立 reward
- **树搜索 + 迭代改进**：子节点基于父节点代码 + scorer 反馈进行改进
- **GRPO + Generational 双源优势值**：同深度组内对比（GRPO）叠加跨代绝对改进（generational）
- **AST 防作弊**：单次 `ast.parse()` 检测所有作弊模式
- **SFT 冷启动**：17 种手写 Triton kernel，零 inductor 模式

---

## 2. 四进程架构

```
┌─ Ray 集群 ────────────────────────────────────────────────┐
│                                                            │
│  ② vLLM Rollout 进程 (GPU)                                 │
│  ├─ 接收 batch prompt + n 分支数                            │
│  ├─ 原生 n>1 多分支采样（不重复 prompt）                      │
│  └─ 返回 response_ids → ①                                   │
│                                                            │
│  ③ Scorer 评分进程 (GPU, Ray actor)                          │
│  ├─ 接收 KernelScoreRequest (node_uid, code_text, ...)     │
│  ├─ Step1: subprocess 编译检测 (crash 隔离)                  │
│  ├─ Step2: N 次随机输入正确性验证 (torch.allclose)           │
│  ├─ Step3: CUDA event 性能测量 (speedup vs torch.compile)   │
│  └─ 返回 KernelScoreResult (feedback, scalar_rewards) → ①   │
│                                                            │
│  ① 主训练循环进程 (CPU)                                      │
│  ├─ KernelTreeSearchManager: BFS 逐层树展开                 │
│  ├─ exporter: 每个非根节点 → 一个 PPO sample                 │
│  ├─ compute_log_prob (当前策略)                             │
│  ├─ compute_multi_section_advantages (双源 adaptive)       │
│  └─ 组装 DataProto → 送 ④                                   │
│                                                            │
│  ④ 训练进程 (GPU, FSDP workers)                             │
│  ├─ PPO 梯度更新: L = L_design + L_code + L_predict          │
│  └─ 权重同步，下一轮 rollout 自动使用新权重                    │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

**通信链路**：

```
① tree_manager ──(prompt batch)──→ ② vLLM ──(response_ids)──→ ①
① tree_manager ──(KernelScoreRequest)──→ ③ scorer ──(KernelScoreResult)──→ ①
① fit() ──(DataProto)──→ ④ FSDP workers ──(gradients)──→ 权重更新
```

---

## 3. SFT 策略

### 3.1 设计原则

- **冷启动质量决定 RL 上限**（DeepSeek-R1 核心经验）。如果 SFT 给的代码是低质的，RL 收敛到低质局部最优
- **只用干净的手写 Triton 代码**，不用 `torch._inductor` 生成的代码（含 `triton_poi_fused_*`、`libdevice.*`、`empty_strided_cuda` 等不可独立复用的模式）
- **同时训练三段式格式**（design/code/predict），让模型在 RL 前就掌握输出格式

### 3.2 尝试过的方法

#### 方案一：KernelBook 数据集（v2，已废弃）

从 KernelBook 提取 800 条 PyTorch module → Triton kernel 转换示例，auto-generate design/predict 文本。

**实际效果**：
- 编译通过率 ~75%
- **问题**：KernelBook 数据包含大量 `torch._inductor` 生成的 triton 代码。模型学会输出 `triton_poi_fused_*`、`libdevice.`、`empty_strided_cuda` 等 inductor 内部 API，不可独立运行
- SFT 模型在未见任务上输出 inductor 模式，评分全 0

#### 方案二：手写规范 kernel（v3，当前使用）

17 种手写 Triton kernel，每种覆盖一种核心模式，带手写 design/predict 分析。

**实际效果**：
- 编译通过率 87.5%（从 75% 提升）
- **零 inductor 模式**：模型输出干净 `@triton.jit` + `tl.load/store` + 显式 grid launch
- 未见任务上 100% 输出三段式格式

### 3.3 最终方案：17 种手写规范 kernel

```
元素级 (7):  relu, gelu, swish, leaky_relu, sigmoid, tanh, elu
规约 (3):    softmax, row_sum, row_mean
归一化 (2):  rms_norm, layer_norm
Matmul (1):  matmul_tiled (tl.dot + Tensor Cores)
融合 (2):    fused_gelu_mul, fused_rmsnorm_residual (LLM 中最常见)
损失/特殊 (2): mse_loss (atomic_add), argmax
```

SFT 数据路径：`data/kernel_rl/sft_train_v3.jsonl`（86 条，含 size variants + revision 样例）

构建脚本：`scripts/kernel_rl/gen_sft_data_v3.py`

---

## 4. Reward 设计

### 4.1 设计原则

1. **三段独立**：code/design/predict 各算各的，互不污染
2. **可验证 reward only**：不给主观分，所有奖励基于客观指标
3. **清晰的梯度层次**：编译失败 < 编译但不正确 < 正确但慢 < 正确且快
4. **作弊负惩罚**：明确检测并惩罚投机取巧的行为
5. **不给所有失败都打 0**：给小的正向信号驱动探索（TritonForge 教训）

### 4.2 Code Reward — 基于加速比的 S-Curve

**实现**：`search_r1/kernel_rl/scoring/reward.py:107-156`

```
r_code = -0.3    (作弊: @triton.jit 定义但用 PyTorch 操作)
         -0.2    (编译失败 + 无 @triton.jit + kernel 体过短)
          0.05   (编译失败，但努力尝试了)
          0.15   (编译通过，但不正确)
          0.2-0.70 (编译通过+正确，S-curve 映射 speedup)
```

**S-Curve 公式**：

```
if speedup <= 0.5:  r = 0.2
elif speedup < 0.8: r = 0.25
else:               r = 0.3 + sigmoid(4.0 * (speedup - 1.2)) * 0.4
r_final = r + 0.05 * correctness_ratio
```

**设计意图**：
- speedup 1.0 (等于 torch.compile): reward = 0.3 + sigmoid(-0.8) * 0.4 ≈ 0.32
- speedup 1.2 (略快于 baseline): reward ≈ 0.30 + 0.5 * 0.4 = 0.50
- speedup 2.0 (目标): reward ≈ 0.30 + 0.96 * 0.4 = 0.68
- correctness_ratio 加成最多 +0.05，平衡正确性样本数量差异

### 4.3 Design Reward — 策略-实现一致性

**实现**：`search_r1/kernel_rl/scoring/reward.py:163-206`

```
score = 0.0
+0.1   design 声称的 BLOCK_SIZE 与 code 中实际一致
+0.1   design 提到 coalesced memory 且 code 中确实使用连续访问
+0.1   design 有具体分析内容（非模板化关键词堆砌）
+0.1   speedup >= 1.0（性能验证 design 策略有效）
上限: 0.4
```

**为什么不和 code reward 挂钩**：早期方案把 design 分数上限定为 `0.6 * r_code`，但这导致 code 好坏直接锁死 design 分数，失去了独立信号。当前方案用实际的"策略-实现一致性"给分，让 design 能独立贡献训练信号。

### 4.4 Predict Reward — 预测准确度

**实现**：`search_r1/kernel_rl/scoring/reward.py:221-257`

```
score = 0.0
编译预测: +0.1  预测与实测一致      -0.03  预测"会编译"但实测失败
加速比预测: +0.15  误差 <20%         +0.08  误差 <50%         -0.05  误差 >200%
瓶颈分析: +0.05  正确识别 compute-bound / memory-bound / balanced
范围: [-0.1, 0.3]
```

**为什么 predict 始终生效**：早期方案讨论过"只有 code 编译通过时才给 predict reward"，但最终决定三段完全独立。原因：即使 code 编译失败，模型正确预测"编译失败"也是有价值的能力信号。只要三段独立计算 advantage，不存在"靠 predict 刷分"的问题。

### 4.5 Penalty 设计：两种 cheat 模式

| 模式 | 检测方式 | Reward | 含义 |
|------|---------|--------|------|
| **Cheating** | AST: `@triton.jit` 存在但从未 launch，wrapper 调 PyTorch | -0.3 | 假装写 Triton 实际用 PyTorch |
| **Parasitic** | AST: kernel body < 3 行（空壳） | -0.2 (等同"没尝试") | 只有装饰器和 return 语句 |
| **No effort** | AST: 无 `@triton.jit` + 编译失败 | -0.2 | 根本没尝试写 Triton |

---

## 5. Advantage 设计

### 5.1 双源优势值

**实现**：`search_r1/kernel_rl/advantage.py`

两个独立的优势值来源，互补：

| 来源 | 计算方式 | 语义 | 归一化 |
|------|---------|------|--------|
| **GRPO** (同深度) | `(r_i - group_mean) / group_std` | "我在和同级节点比，表现如何" | z-score 天然归一化 |
| **Generational** (代际) | `(r_child - r_parent) / 0.3` | "我比父节点改进了多少" | 除以 baseline 0.3 |

### 5.2 尝试过的方法

**方案一（v1，已废弃）**：固定权重 0.5/0.5 合并，gen 用比值 `(r_child - r_parent) / r_parent`

- **问题 1**：GRPO 输出 z-score (std≈1)，gen 输出比值 (std 可能 0.1 也可能 10)，尺度不一致，一方完全主导
- **问题 2**：`r_parent → 0` 时 gen 比值爆炸（r_parent=0.01, r_child=0.02 → gen=1.0, 而 r_parent=0.5, r_child=0.6 → gen=0.2，绝对改进更大的反而信号更小）
- **问题 3**：双重 masking bug，gen_adv 被 section_mask 乘了两次

**方案二（v1.1，已废弃）**：自适应权重 `w_grpo = gen_std / (grpo_std + gen_std)`，gen 用对数比值

- **问题**：对数比值虽然对称有界，但在零点附近不连续（r_parent→0 时 log(ε/0.01) 行为不确定）

### 5.3 最终方案：自适应归一化等权合并

```python
# 1. GRPO base advantage (组内 z-score, 天然均值0方差1)
adv_grpo = compute_grpo_outcome_advantage(scores, mask, group_index)

# 2. Generational advantage (绝对差值 / 0.3)
# 0.3 = speedup=1.0 时的 code reward, 是"及格线"
gen_adv = (r_child - r_parent) / 0.3

# 3. 自适应归一化: 双方除标准差到 unit variance → 等权合并
grpo_std = adv_grpo.std().clamp(min=1e-6)
gen_std  = gen_adv.std().clamp(min=1e-6)

if grpo_valid and gen_valid:
    gen_expanded = gen_adv.unsqueeze(-1)  # [batch] → [batch, 1]
    adv_combined = adv_grpo / grpo_std + gen_expanded / gen_std
elif grpo_valid:
    adv_combined = adv_grpo
elif gen_valid:
    adv_combined = gen_adv.unsqueeze(-1).expand_as(section_mask)
else:
    # 4. 双方都无效 → 绝对 baseline 保底: (r - 0.15) / 0.3
    sample_rewards = (section_scores * section_mask).sum(dim=1) / mask_sum
    adv_combined = ((sample_rewards - 0.15) / 0.3).unsqueeze(-1)
```

**std < 0.05 视为无效**：当组内所有节点 reward 完全相同（比如全部编译失败），GRPO 无法提供区分信号

---

## 6. 树搜索结构

### 6.1 BFS 逐层展开

**实现**：`search_r1/kernel_rl/tree_manager.py`

```
配置: max_depth=2, branch_factors=[2,2], keep_per_depth=[4,4]

Step 1: batch_size=8 → 8 棵独立的树
        每棵树: root → 2 children (depth 1) → 4 leaves (depth 2)
        总计: 8 × 7 = 56 个节点,  56 - 8 = 48 个训练样本
```

### 6.2 逐层流程

```
for depth in [0, 1]:
    1. _build_prompt_batch(active_nodes)
       - 根节点: task_spec prompt (约1500 tokens)
       - 子节点: 父 prompt + 父 response + scorer feedback + 改进指令 (约2500 tokens)
    2. vLLM generate_sequences(n=branch_factor)
       - 每个 prompt 生成 n 个独立响应
    3. _materialize_children()
       - 解析 response (三段式)
       - 创建 KernelScoreRequest
       - 调用 scorer.score_many() (并行评分)
       - 将 scorer 结果 attach 到节点
    4. _select_active_children() (按 total_reward 剪枝到 keep_count)
```

### 6.3 KV Cache 与前缀共享

**当前状态**：
- 同 batch 内同父节点的兄弟子节点天然共享前缀（vLLM automatic prefix caching 生效）
- **跨层不共享**：每层的 child prompt 是完整的独立文本（含祖先 prompt + response + feedback），不是前缀拼接
- 根因：逐层评分需要 scorer 反馈 → 必须等当前层全部评分完才能构建下一层 prompt → 中间有 30s+ 间隔 → 前缀 KV cache 被 LRU 逐出

**已知问题**：prompt 随深度线性增长（1500 → 2500 → 3500 tokens），vLLM 无前缀复用时为 O(depth × batch × branch²) 的计算量

### 6.4 树日志结构

```
outputs/kernel_tree_logs/
└── tree_{task_name}/
    ├── d0_n0.txt    ← 根: 完整 prompt
    ├── d1_n0.txt    ← 子: 父 prompt + 父 response + feedback + 模型输出 + 评分
    ├── d1_n1.txt
    ├── d2_n0.txt    ← 叶: 同上
    ├── d2_n1.txt
    ├── d2_n2.txt
    └── d2_n3.txt
```

每个节点文件包含：完整 prompt、模型 response、scorer 反馈、所有 metrics。

---

## 7. 评测流水线 (Scoring Pipeline)

### 7.1 三步评测

**实现**：`search_r1/kernel_rl/scoring/evaluator.py`

```
Step 0: 编译检测 (subprocess 隔离)
  ├─ compiler.compile_in_subprocess(code, timeout=120s)
  ├─ crash 不会杀死 actor
  └─ 失败 → EvalResult(compiled=False, error_msg=stderr)

Step 1: in-process 加载
  ├─ @triton.jit 不支持 exec() → tempfile + importlib 方案
  ├─ 提取 ref_fn / new_fn / input_gen
  └─ 失败 → EvalResult(compiled=False)

Step 2: 正确性验证
  ├─ check_correctness(ref_fn, new_fn, input_gen, num_trials=3)
  ├─ 每次 trial: input_gen() 生成随机输入 → ref_fn(*inputs) vs new_fn(*inputs)
  ├─ torch.allclose(atol=1e-4, rtol=1e-4)
  └─ 失败 → EvalResult(correctness=False, num_passed_trials, max_diff)

Step 3: 性能测量
  ├─ torch.compile(ref_fn, mode="reduce-overhead") 作为 baseline
  ├─ CUDA event 计时: 5 warmup + 20 trials (discard first 2)
  ├─ speedup = ref_runtime / kernel_runtime
  └─ → EvalResult(speedup, runtime_ms, ref_runtime_ms)
```

### 7.2 Scorer 架构

```
KernelScoringPool (轮询调度)
├─ KernelScoringWorker 0 (GPU 0, num_gpus=0 → CPU scheduling)
├─ KernelScoringWorker 1 (GPU 0)
└─ score_many(requests[]) → ray.get() → List[KernelScoreResult]
```

**num_gpus=0 的原因**：设为 0.2 会碎片化 GPU0，veRL 的 4×STRICT_PACK 调度策略无法找到 4 个完整 GPU → 死锁。

### 7.3 环境反馈

**实现**：`evaluator.py:build_feedback()`

分层反馈，帮助模型自我改进：

```
编译失败: [COMPILE ERROR] + 错误类型 + stderr(前800字符含行号) + HINT
不正确:   [COMPILED] + [CORRECTNESS] FAILED + 通过/总 trial 数 + max_diff + HINT
正确:     [COMPILED] + [CORRECTNESS] PASS + [PERFORMANCE] speedup + 分级提示
          - < 0.8x: "SLOWER than torch.compile"
          - 0.8-1.0x: "Close, try larger BLOCK_SIZE"
          - 1.0-2.0x: "GOOD, can you push further?"
          - > 2.0x: "EXCELLENT"
```

---

## 8. 防作弊机制 (Anti-Hacking)

### 8.1 统一 AST 分析

**实现**：`search_r1/kernel_rl/scoring/reward.py:25-95`

单次 `ast.parse()` 完成所有检测，替代之前的多次独立检查：

```python
def analyze_code_structure(code_text: str) -> dict:
    tree = ast.parse(code_text)

    # 1. 找到所有 @triton.jit 装饰的函数 + body 行数
    for node in ast.walk(tree):
        if is_function_with_triton_jit_decorator(node):
            triton_functions[name] = body_line_count

    # 2. 检测 wrapper 中的调用模式
    for node in ast.walk(tree):
        if is_call(node):
            # Triton launch: fn_name[grid](args)
            # PyTorch ops: torch.nn.functional.*, torch.relu, F.gelu, etc.

    # 3. 综合判断
    is_cheating  = has_triton_jit AND pytorch_called AND NOT triton_launched
    is_parasitic = has_triton_jit AND min_kernel_body < 3 lines
```

### 8.2 三种作弊模式

| 模式 | AST 特征 | Reward 后果 |
|------|---------|------------|
| **Cheating** | `@triton.jit` 定义了但从未 `kernel[grid](...)` 调用，wrapper 调 PyTorch | code reward = -0.3 |
| **Parasitic** | `@triton.jit` 存在但 kernel body < 3 行（空壳） | 等同于"没尝试", max 0.05 |
| **No Triton** | 代码中无 `@triton.jit` + 编译失败 | code reward = -0.2 |

### 8.3 @triton.jit + exec() 不兼容

Triton 的 `@triton.jit` 装饰器依赖源码文件路径和 `__file__` 属性，`exec()` 没有这些元数据。

**解决方案**：tempfile + importlib（参考 KernelBench `load_custom_model_with_tempfile`）

```python
# compiler.py
import tempfile, importlib, sys
with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
    f.write(code.encode())
# 动态添加到 sys.modules
spec = importlib.util.spec_from_file_location(module_name, f.name)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
```

---

## 9. 训练流水线

### 9.1 种子任务

**24 任务，3 个难度级别**：`data/kernel_rl/train.parquet`

| Level | 数量 | 类别 | 任务 |
|-------|------|------|------|
| 1 (easy) | 10 | elementwise, activation, reduction | vector_add, gelu, relu, sigmoid, tanh, swish, leaky_relu, elu, row_sum, row_mean |
| 2 (medium) | 8 | normalization, reduction, fusion | softmax, layer_norm, rms_norm, argmax, mse_loss, fused_gelu_mul, fused_rmsnorm_residual, batch_norm |
| 3 (hard) | 6 | gemm, attention, fusion | matmul_naive, attention_score, matmul_tiled, sdpa_full, fused_linear_gelu, fused_matmul_bias_relu |

### 9.2 配置

```yaml
# RTX 4090 / A100 / H200 通用参数
algorithm:
  adv_estimator: grpo
  kernel_adv_estimator: grpo

kernel:
  max_depth: 2
  branch_factors: [2, 2]
  keep_per_depth: [4, 4]
  scorer:
    mode: eval
    num_workers: 2
    timeout_s: 300

trainer:
  total_training_steps: 50
  total_epochs: 10

actor:
  lr: 1e-6
  ppo_epochs: 1
  entropy_coeff: 0.001
  policy_loss: dual_clip
  multi_advantage: enabled: True
```

### 9.3 RTX 4090 内存问题

4× RTX 4090 (49GB) 上运行 hybrid engine 7B 模型的实验结果：

| 配置 | 结果 | 原因 |
|------|------|------|
| `gpu_memory_utilization: 0.6` | Step 1 成功，Step 2 OOM | FSDP 训练后 vLLM cumem 池被 PyTorch 占用，wake_up 失败 |
| `gpu_memory_utilization: 0.55` | 初始化失败 | 低于 vLLM 初始化阈值（~0.59） |
| `gpu_memory_utilization: 0.4` | 初始化失败 | KV cache 空间不足 |
| `expandable_segments: True` | 初始化失败 | vLLM CuMemAllocator assert 拒绝（已知不兼容） |

**Step 1 训练信号**（证明 RL 链路有效）：

| 指标 | Run 1 | Run 2 |
|------|-------|-------|
| `pg_loss` | -0.78 | -2.85 |
| `pg_loss_design` | +1.35 | -2.58 |
| `pg_loss_code` | -2.13 | -0.27 |
| `entropy_loss` | 0.036 | 0.034 |

**结论**：RL 训练链路完全正常，RTX 4090 49GB 内存不足以稳定运行 multi-step。已切换到 H200。

### 9.4 PPO Sample 组成

```
每个非根节点导出为一个 PPO sample:

input_ids         = cat([prompt_ids, response_ids])
attention_mask    = cat([prompt_attn, response_attn])
loss_mask         = OR(design_mask, code_mask, predict_mask)
design_token_scores  = design_mask × scalar_design_reward
code_token_scores    = code_mask × scalar_code_reward
predict_token_scores = predict_mask × scalar_predict_reward

训练时:
  L = L_design + L_code + L_predict
  pg_loss_{section} = -advantage × log_prob_ratio (clipped)
```

**Fallback**: parser 解析不出任何 section 时，将所有有效 response token 视为 code section，避免 loss_mask 全零

---

## 10. H200 部署指南

### 10.1 一键启动

```bash
# 设置路径
export MODEL_PATH=/path/to/Qwen2.5-Coder-7B-Instruct-SFT-kernel-v3
export RUN_DIR=/path/to/output_dir

# 一键启动（含进度跟踪）
bash scripts/kernel_rl/deploy_h200.sh
```

**前置条件**：H200 机器上已安装 PyTorch, Triton, vLLM, Ray, veRL。

**自动执行**：检查依赖 → 显示 GPU 信息 → 启动训练 → 后台进度跟踪 → 训练结束自动清理 Ray。

### 10.2 输出文件地图

```
$RUN_DIR/
├── train.log              ← 完整训练日志，实时写入
├── progress.json          ← 每步更新，JSON 格式进度文件
├── deploy.log             ← 部署脚本输出
│
├── tree_logs/             ← 每个节点的完整输出
│   └── tree_{task_name}/
│       ├── d0_n0.txt      ← 含 prompt + response + scorer 反馈
│       └── ...
│
├── scorer_logs/           ← 评分原始数据 (JSONL)
│   └── scorer_pid*.jsonl  ← 每行: {node_uid, status, compiled,
│                              correctness, speedup, rewards}
│
├── samples/               ← 每 10 步提取模型生成的代码
│   └── step_0010/
│       ├── tree_kernel_00_d1_n0.py  ← 可独立运行的 Triton kernel
│       └── ...
│
└── checkpoints/           ← FSDP 模型权重（每 10 步保存）
```

### 10.3 progress.json 格式

```json
{
  "ts": "2026-06-01T10:30:00",
  "step": 12,
  "elapsed": "0:25:30",
  "eta": "1:20:00",
  "training": {
    "pg_loss": -1.234,
    "pg_loss_design": -0.567,
    "pg_loss_code": -2.100,
    "pg_loss_predict": -0.030,
    "entropy_loss": 0.028
  },
  "scorer": {
    "compile_rate": 0.875,
    "correctness_rate": 0.42,
    "avg_speedup": 0.85,
    "max_speedup": 2.1,
    "avg_code_reward": 0.23,
    "avg_design_reward": 0.12,
    "avg_predict_reward": 0.08,
    "total_scored": 672
  },
  "by_task": {
    "kernel_00": {"compile_rate": 1.0, "correctness_rate": 0.8},
    "kernel_10": {"compile_rate": 0.5, "correctness_rate": 0.0}
  }
}
```

### 10.4 如何监控训练进展

**1. 实时看每一步指标**（终端输出）：
```
[step   5] pg_loss=-2.848  code=-0.268  design=-2.581  compile=87%  correct=42%  speedup_avg=0.73x  ETA=1:35:00
[step  10] pg_loss=-1.920  code=-1.530  design=-0.390  compile=90%  correct=50%  speedup_avg=0.81x  ETA=1:20:00
  [samples] extracted to .../samples/step_0010
```

**2. 最新进度快照**：
```bash
cat $RUN_DIR/progress.json | python3 -m json.tool
```

**3. 查看模型生成的代码**：
```bash
ls $RUN_DIR/samples/step_0010/
cat $RUN_DIR/samples/step_0010/tree_kernel_00_d1_n0.py
```

**4. 查看具体某个节点的完整信息**（prompt → response → scorer 反馈）：
```bash
cat $RUN_DIR/tree_logs/tree_kernel_03/d2_n1.txt
```

**5. 分析 scorer 趋势**：
```bash
# 编译率趋势
python3 -c "
import json
with open('$RUN_DIR/progress.json') as f:
    p = json.load(f)
    print(f'Step {p[\"step\"]}: compile={p[\"scorer\"][\"compile_rate\"]:.1%} '
          f'correct={p[\"scorer\"][\"correctness_rate\"]:.1%} '
          f'speedup_avg={p[\"scorer\"][\"avg_speedup\"]:.2f}x')
"
```

### 10.5 关注的关键趋势

| 指标 | 期望趋势 | 含义 |
|------|---------|------|
| `compile_rate` | 上升至 >90% | 模型学会了写可编译的 Triton 代码 |
| `correctness_rate` | 从 0% 开始上升 | 模型开始写出数值正确的 kernel |
| `avg_speedup` | 突破 1.0 | 模型代码开始比 torch.compile 快 |
| `pg_loss_code` | 保持负值 | RL 持续改进 code section |
| `pg_loss_design` | 保持负值 | RL 持续改进 design 分析 |
| `entropy_loss` | 0.02-0.05 | 策略有适当探索性 |

---

## 11. 设计决策记录

### 11.1 为什么用 GRPO 而不是标准 PPO+Critic

- Critic 网络需要额外 ~7GB 显存，在 RTX 4090 上不可行
- GRPO 用 group-wise z-score 替代 value function，不依赖 critic 网络
- 当 group 内所有样本相同时 GRPO 失效 → Generational advantage 兜底
- 当双方都失效时 → 绝对 baseline 保底

### 11.2 为什么 Design/Predict 不按 code reward 缩放

早期方案讨论过"design 分数上界不超过 `0.6 * code_reward`"。最终否决：

- 这会锁死 design 和 predict 的信号，使它们变成 code 的派生值
- RL 需要三个独立信号各自驱动不同能力的提升（代码质量、分析能力、预测能力）
- 三段独立计算 advantage（各自用各自的 mask + scores），数学上不存在"靠 predict 刷分"的问题

### 11.3 为什么 SFT 不用 KernelBench/KernelBook 数据

- KernelBook 含大量 `torch._inductor` 生成的代码（`triton_poi_fused_*`、`libdevice.*`）
- KernelBench 数据格式与 scorer 期望不完全匹配（需要额外的 task_spec/bench_spec 适配）
- 手写 17 种模式覆盖了所有核心 Triton 编程范式，且经过 scorer 验证

### 11.4 为什么用 depth=2 的二叉树

- Depth 更大 → prompt 更长 → 前缀 KV cache 失效 → 计算量指数增长
- Depth=2 给了模型一轮"分析-修复"的机会，足够验证迭代改进能力
- 实验后在更大显存机器上可扩展到 depth=3

### 11.5 vLLM sleep/wake 机制与内存竞争

veRL hybrid engine 的核心问题：

```
__enter__ (切换到 vLLM):
  wake_up() → 从 CUDA 重新分配 KV cache 池
  sync_model_weights(FSDP → vLLM)
  del params, torch.cuda.empty_cache()

__exit__ (切换回 FSDP):
  sleep(level=1) → 释放 KV cache 池归还 CUDA
  module.train()
  torch.cuda.empty_cache()
```

**49GB 上失败的原因**：FSDP 训练后 PyTorch CUDA allocator 占用了之前 vLLM cumem 池的空间，即使 `torch.cuda.empty_cache()` 也无法保证 cumem 重新分配到同样大小的连续块。`expandable_segments: True` 可以缓解但 vLLM CuMemAllocator 显式拒绝。

**H200 上无此问题**：141GB 远超 7B 模型 + FSDP + KV cache 的总需求（~60GB），内存池竞争不再发生。
