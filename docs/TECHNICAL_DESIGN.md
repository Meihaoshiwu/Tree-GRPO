# Tree-GRPO Kernel RL — 完整技术设计方案

> 版本: Phase 2 | 日期: 2026-05-25

---

## 1. 项目概述

基于 Tree-GRPO + veRL 构建的 **Triton 算子开发 RL 框架**。核心思路：用树搜索结构组织多轮迭代改进，用可验证的客观 reward（编译、正确性、性能）驱动 RL 训练，最终让模型学会写出比 `torch.compile` 更快的 Triton kernel。

### 1.1 四进程架构

```
┌─ Ray 集群 ────────────────────────────────────────────────┐
│                                                            │
│  ② vLLM Rollout (GPU) — 数据并行，每卡独立 KV cache        │
│  ├─ 接收 batch prompt + 分支数 n                            │
│  └─ 返回 response_ids → ①                                   │
│                                                            │
│  ③ Scorer 评分 (CPU/GPU, Ray actor, num_gpus=0)            │
│  ├─ Step 1: subprocess 编译检测 (crash 隔离)               │
│  ├─ Step 2: N 次随机输入 torch.allclose 正确性验证          │
│  ├─ Step 3: CUDA event 性能测量 vs torch.compile baseline  │
│  └─ 返回 feedback + scalar_rewards → ①                     │
│                                                            │
│  ① 主训练循环 (tree_manager)                               │
│  ├─ BFS 逐层收集 prompt → tokenize → 组 batch               │
│  ├─ 调 vLLM → parse → 创建子节点                            │
│  ├─ 调 scorer → 给节点赋 section rewards                   │
│  ├─ exporter: 每个非根节点 → 一个 PPO sample                │
│  └─ 计算 advantage + 组装 DataProto → 送 ④                  │
│                                                            │
│  ④ FSDP Training (GPU)                                       │
│  ├─ 分段 PPO loss: L = L_design + L_code + L_predict        │
│  ├─ 自适应优势值: GRPO (同深度) + Generational (父子)       │
│  └─ 梯度更新，权重同步                                      │
└────────────────────────────────────────────────────────────┘
```

### 1.2 树搜索结构

```
depth=0 (ROOT):   d0_n0               ← 原始任务 prompt
                    │
depth=1:       d1_n0  d1_n1           ← 父 response + scorer 反馈后生成
                 │       │
depth=2:   d2_n0 d2_n1 d2_n2 d2_n3   ← 再次迭代改进
```

- BFS 逐层推理：同深度节点组 batch 送入 vLLM
- vLLM 原生 `n>1` 多分支采样（不重复 prompt）
- 子节点 prompt = 父节点 prompt + 父节点 response + scorer 反馈
- 每个非根节点导出为一个独立 PPO 训练样本

---

## 2. 模型与数据流

### 2.1 模型

| 项目 | 值 |
|------|------|
| 基座模型 | Qwen2.5-Coder-7B-Instruct |
| SFT 模型 | Qwen2.5-Coder-7B-Instruct-SFT-kernel-v3 |
| 参数量 | 7.61B |
| 上下文 | 128K (YaRN) |
| 推理引擎 | vLLM 0.8.5 (V1 engine, Flash Attention) |
| 训练策略 | FSDP full_shard, BF16, 4×GPU |

### 2.2 数据格式

模型输出固定三段式，由 parser 提取 span：

```
<design>
优化策略、block size 选择依据、内存布局决策。
如果是修正轮次，先 critique 上一轮的尝试。
</design>
<code>
@triton.jit 装饰的 kernel + wrapper 函数（含显式 grid launch）
</code>
<predict>
预计加速比 vs PyTorch、瓶颈分析、置信度
</predict>
```

### 2.3 PPO Sample 组成

每个 `KernelTrainSample` 包含：

```
input_ids = cat([prompt_ids, response_ids])
loss_mask = OR(design_mask, code_mask, predict_mask)
token_level_scores = design_mask × scalar_design_reward
                   + code_mask × scalar_code_reward
                   + predict_mask × scalar_predict_reward
advantages = GRPO(同深度) + Generational(父子) [自适应归一化]
```

---

## 3. SFT 策略

### 3.1 设计原则

1. **代码必须手写 Triton**：不用 torch._inductor 生成的代码（`triton_poi_fused_*`、`libdevice.*`、`empty_strided_cuda` 模式）
2. **覆盖核心 kernel 模式**：element-wise、normalization、reduction、matmul、fusion
3. **包含 revision 样例**：模型需要见过 "错误尝试 → scorer 反馈 → critique → 改进" 的模式
4. **Design 有信息量**：基于 kernel 实际特征的自动分析（操作类型、内存模式、block size 策略），不是模板填充

### 3.2 数据来源

| 来源 | 描述 | 数量 |
|------|------|------|
| 手写 Triton kernels | 17 种 kernel 模式（ReLU, GELU, Swish, RMSNorm, LayerNorm, Softmax, Matmul 等） | 17 base |
| 变体扩展 | 不同输入尺寸的 prompt 变体 | ×5-8 |
| Revision 样例 | 故意错误的 kernel + scorer 反馈 + 修正版本 | 3 |
| **SFT v3 总计** | | **86** |

### 3.3 SFT 训练配置

```yaml
模型: Qwen2.5-Coder-7B-Instruct
数据: sft_train_v3.jsonl (86 examples)
训练方式: FSDP full_shard, BF16
Epochs: 10
Batch: per_device=1, grad_accum=2, effective=8
Learning rate: 1e-5, cosine decay
Max sequence length: 2048
Gradient checkpointing: enabled
```

### 3.4 SFT 数据中禁止的 inductor 模式

```
❌ triton_poi_fused_*        → @triton.jit + 描述性函数名
❌ libdevice.*               → tl.math.*
❌ empty_strided_cuda        → torch.empty
❌ assert_size_stride        → 显式 shape 检查
❌ @triton.jit + grid() 宏    → 显式 grid=(n,) launch
❌ def call(args)            → 独立 wrapper 函数
```

---

## 4. 奖励函数设计

### 4.1 三段独立原则

每段用自己的客观指标给分，互不污染：

| 段 | 信号来源 | 客观性 |
|------|------|------|
| code | 编译结果 + 正确性比例 + speedup vs torch.compile | ✅ 完全客观 |
| design | 策略-实现一致性 + 性能指标支撑 | ✅ 基于可测事实 |
| predict | 预测 vs 实测偏差 | ✅ 可计算偏差 |

### 4.2 Code Reward — S-Curve

```
编译失败 (无 @triton.jit):   -0.20   ← 根本没尝试
编译失败 (寄生 kernel):       -0.20   ← AST 检测 body < 3 行
编译失败 (努力过):             0.05   ← 有 @triton.jit 但编译不过
编译通过 + 不正确:             0.15   ← 能编译但算错了
编译通过 + 正确:
  speedup ≤ 0.5:             0.20   ← 慢但至少正确
  speedup = 0.8:             0.25
  speedup = 1.0:             0.30   ← 持平 torch.compile
  speedup = 1.5:             0.52   ← 明显更快
  speedup = 2.0:             0.65   ← 2× 加速
  speedup ≥ 3.0:             0.70   ← 封顶

公式: r = 0.3 + sigmoid(4.0 × (speedup - 1.2)) × 0.4
            + 0.05 × correctness_ratio
```

**设计特点**：
- 低加速比区域平滑——鼓励尝试
- 中等加速比陡峭——区分好坏
- 高加速比封顶——防 reward hacking
- 平庸代码（慢但正确）不给惩罚——只有两种例外给负分

### 4.3 Design Reward

基于性能指标，不依赖关键词匹配：

```
score = 0.0
+ 0.10  如果 design 中声称的 block_size 与 code 中实际一致
+ 0.10  如果 design 中的内存模式描述与 code 实现匹配
+ 0.10  如果 design 包含具体分析（非模板化泛泛而谈）
+ 0.10  如果 speedup ≥ 1.0（性能达标 → design 有效）
上限: 0.40
```

### 4.4 Predict Reward

预测准确性，按指标逐个给分：

```
Metric 1: 编译预测正确         → +0.10
Metric 2: 加速比预测误差 < 20% → +0.15
         加速比预测误差 < 50% → +0.08
         加速比预测误差 > 200%→ -0.05 (wildly wrong)
Metric 3: 瓶颈分析正确         → +0.05
范围: [-0.10, 0.30]
```

**Predict 在三段中独立计算**——不会因为 code 编译失败而影响 predict 的评分。

### 4.5 防作弊 (AST 分析)

一次 `ast.parse()` 返回完整结构信息：

```
analyze_code_structure(code) → {
    has_triton_jit:  bool    # 有 @triton.jit 函数？
    triton_launched: bool    # Triton kernel 被 [...] launch 了？
    pytorch_called:  bool    # wrapper 调了 PyTorch？
    is_cheating:     bool    # 有 @triton.jit 但不 launch，用了 PyTorch
    is_parasitic:    bool    # kernel body < 3 行（空壳）
}
```

三层防御：
1. **作弊** (`is_cheating`): -0.3 — 写了 @triton.jit 但从不用，wrapper 偷偷调 PyTorch
2. **寄生** (`is_parasitic`): -0.2 — 塞了个空壳 `@triton.jit def f(): pass`
3. **未尝试** (no @triton.jit): -0.2 — 根本没写 Triton

---

## 5. 优势值设计

### 5.1 两个信号源

| 信号源 | 含义 | 计算方式 |
|------|------|------|
| GRPO (同深度) | `(r_i - group_mean) / (group_std + ε)` | z-score，组内对比 |
| Generational (代际) | `(r_child - r_parent) / 0.3` | 绝对差值 ÷ baseline |

- **GRPO**：在同一深度的节点之间比较。好的相对于差的得到正优势值。
- **Generational**：子节点与父节点比较。改进得到正优势值，退步得到负优势值。
- `0.3` 是 speedup=1.0 时的 code reward，作为归一化基准——比比值`(r_child-r_parent)/r_parent`更稳定，不会在 r_parent→0 时爆炸。

### 5.2 自适应归一化等权合并

```python
# 双方归一化到单位方差
adv_grpo_norm = adv_grpo / adv_grpo.std()   # 方差=1
adv_gen_norm  = adv_gen  / adv_gen.std()    # 方差=1

# 等权合并
adv_combined = 0.5 * adv_grpo_norm + 0.5 * adv_gen_norm
```

**设计理由**：GRPO 方差可能远大于 Generational（一个节点偶然编译成功时），固定权重会导致 Generational 信号被淹没。归一化后双方均分影响力，确保模型同时学习"谁在组内更好"和"谁比父节点有进步"。

### 5.3 绝对 Reward 保底

当 GRPO 和 Generational 同时失效（全组 reward 相同，无父节点差异）：

```python
if grpo_std < 0.05 and gen_std < 1e-8:
    # 回退到绝对 baseline 对比
    adv = (sample_reward - 0.15) / 0.3
    # 0.15 = "编译通过但不正确" 的 reward floor
```

### 5.4 三段独立优势值

```
design_advantages  ← GRPO + Generational on design rewards
code_advantages    ← GRPO + Generational on code rewards
predict_advantages ← GRPO + Generational on predict rewards

PPO loss = L_design + L_code + L_predict
          ↑ 分别用各自的 advantages 计算
```

三段互不干扰：code 差不会拖累 design 的优势值，predict 好不会掩盖 code 的问题。

---

## 6. 环境反馈设计

### 6.1 Scorer → 模型反馈

子节点的 prompt 中注入上一轮的 scorer 反馈。反馈格式：

```
[COMPILE ERROR] — Your code failed to compile.
Error type: import
Compiler output (use this to fix your code):
  ... compiler stderr with line numbers ...

HINT: Check @triton.jit syntax, function signatures, ...

[COMPILED] OK — Your Triton kernel compiled successfully.
[CORRECTNESS] FAILED — 0/3 trials passed.
Max numerical difference from reference: 0.5432

HINT: Check input/output shapes, dtype consistency, ...

[PERFORMANCE] Your kernel: 0.1234ms
[PERFORMANCE] torch.compile baseline: 0.0987ms
[PERFORMANCE] Speedup: 0.80x
Your kernel is SLOWER than torch.compile. Try larger BLOCK_SIZE...
```

### 6.2 反馈的层级递进

| 层级 | 反馈内容 | 何时出现 |
|------|------|------|
| 编译错误 | stderr + 行号 + 编译提示 | 编译失败 |
| 运行时错误 | error type + traceback + shape hint | 编译通过但 crash |
| 数值偏差 | max_diff + passed/total | 结果不对 |
| 性能比较 | runtime + speedup + 优化建议 | 正确但需要优化 |

### 6.3 SFT 中的反馈训练

在 SFT 数据中包含了 3 个 revision 样例，展示了模型如何从 scorer 反馈中学习改进：

```
prompt:
  [原始任务 + 上一轮的有问题的代码 + scorer 的具体错误反馈]
  "Your task: Analyze what went wrong. Critique the previous version,
   then propose your improved strategy."

completion:
  <design>
  Critique: [指出上轮的具体问题]
  Strategy: [提出改进方案]
  </design>
  <code>
  [改正后能编译通过的代码]
  </code>
```

---

## 7. 训练流水线

### 7.1 E2E 训练步骤

```bash
# 1. 启动训练
bash scripts/kernel_rl/run_e2e.sh

# 2. 监控指标
tail -f scripts/kernel_rl/e2e_run_*.log | grep "actor/pg_loss\|rewards/mean"

# 3. 查看生成代码
cat outputs/kernel_tree_logs/tree_kernel_00/d1_n0.txt

# 4. 查看 scorer 结果
cat outputs/kernel_scorer_logs/scorer_pid*.jsonl
```

### 7.2 关键监控指标

| Metric | 含义 | 健康趋势 |
|------|------|------|
| `response_length/mean` | 平均生成长度 | 500-1024 |
| `critic/rewards/mean` | 平均 reward | 逐步上升 |
| `actor/pg_loss_code` | code 段 PPO loss | 负值且下降 |
| `actor/grad_norm` | 梯度范数 | 0.01-1.0 |
| `actor/pg_loss_design` | design 段 loss | 与 code 同趋势 |
| `actor/pg_loss_predict` | predict 段 loss | 与 code 同趋势 |

### 7.3 关键文件

| 文件 | 作用 |
|------|------|
| `verl/trainer/config/ppo_trainer_kernel.yaml` | 训练配置 |
| `search_r1/kernel_rl/prompt_builder.py` | Prompt 构造 (含 CRITICAL RULES) |
| `search_r1/kernel_rl/scoring/evaluator.py` | 三步评测流水线 |
| `search_r1/kernel_rl/scoring/reward.py` | 奖励函数 + AST 防作弊 |
| `search_r1/kernel_rl/scoring/compiler.py` | Subprocess 编译隔离 |
| `search_r1/kernel_rl/scoring/correctness.py` | 数值正确性验证 |
| `search_r1/kernel_rl/scoring/benchmark.py` | CUDA event 性能测量 |
| `search_r1/kernel_rl/advantage.py` | 自适应优势值计算 |
| `search_r1/kernel_rl/scorer.py` | Ray actor 入口 |
| `search_r1/kernel_rl/tree_manager.py` | 树搜索管理 + 日志 |
| `scripts/kernel_rl/run_e2e.sh` | E2E 启动脚本 |
| `scripts/kernel_rl/run_sft.sh` | SFT 启动脚本 |
| `scripts/kernel_rl/gen_sft_data_v3.py` | SFT 数据生成 |

---

## 8. H200 × 8 适配

### 8.1 配置变更

| 参数 | RTX 4090 × 4 | H200 × 8 |
|------|------|------|
| 单卡显存 | 48 GB | 141 GB |
| `gpu_memory_utilization` | 0.6 | 0.85 |
| `tensor_model_parallel_size` | 1 | 2⁻4 |
| `ppo_max_token_len_per_gpu` | 8192 | 32768 |
| `train_batch_size` | 4 | 8⁻16 |
| `max_num_seqs` | 32 | 128 |
| `max_num_batched_tokens` | 4096 | 16384 |

### 8.2 H200 额外能力

- **NVLink**: `tp_size ≥ 2` 时 KV cache 跨 GPU 共享
- **FP8**: vLLM 可加 `quantization: fp8` 加速推理
- **FlashAttention 3**: H200 H100 架构原生支持
- **torch.compile mode="max-autotune"**: 更多 tuning 空间

---

## 9. 输出与存储

| 位置 | 用途 | 文件系统 | 持久化 |
|------|------|------|------|
| `models/` | 模型权重 | GPFS (共享) | ✅ 长期 |
| `data/` | 训练数据 | GPFS (共享) | ✅ 长期 |
| `outputs/` | 训练日志 (scorer/tree) | GPFS (共享) | 运行后保留 |
| `/tmp/` | Checkpoint, Ray 临时文件 | overlay (1.5TB) | 自动清理 |

---

## 10. 当前状态与已知限制

### 10.1 已完成

- ✅ 树搜索 + vLLM rollout + scorer 评测 + FSDP 训练的完整链路
- ✅ SFT v3：86 个手写 Triton 样例，零 inductor 模式
- ✅ 三段独立 reward (code S-curve + design 一致性 + predict 准确度)
- ✅ AST 防作弊 (作弊/寄生/未尝试三层检测)
- ✅ 富环境反馈 (编译 stderr + 行号 + 性能分级提示)
- ✅ 自适应归一化优势值 (GRPO + Generational 等权合并)
- ✅ 绝对 baseline 保底

### 10.2 已知限制

- SFT 数据量偏少 (86 例)，覆盖率有限。后续需扩展至 200+ 例
- 种子任务仅 6 个，需扩充 KernelBench Level 1/2/3
- 多步 RL 训练尚未执行
- 训练指标仅 1 步验证
- H200 适配未实际测试
- GPFS 共享存储配额紧张 (~368MB 空闲)

---

## 11. 设计决策记录

| 决策 | 选择的方案 | 理由 |
|------|------|------|
| Advantage 信号源 | GRPO + Generational 双源 | GRPO 区分好坏，Generational 奖励改进 |
| Generational 公式 | `(r_child - r_parent) / 0.3` | 绝对差值比比值稳定，不爆炸 |
| 归一化方式 | 各自除 std 后等权 | 防止一方方差独大 |
| 保底 advantage | `(r - 0.15) / 0.3` | 不需要 critic model |
| Code reward 曲线 | S-curve (sigmoid) | 低加速比平滑鼓励，高加速比封顶防 hack |
| 惩罚设计 | 只两种例外给负分 | 不给平庸代码惩罚，避免抑制输出 |
| 防作弊 | AST (`ast.parse()`) | 比正则可靠，不受字符串/注释干扰 |
| 寄生检测 | kernel body < 3 行 | 防止塞 @triton.jit 空壳 |
| Predict 生效时机 | 总是生效 (三段独立) | code 差不会拖累 predict |
| Design 评分 | 性能指标，非关键词 | 避免模板化 design 刷分 |
| SFT 数据 | 手写 Triton，不用 inductor | inductor 代码不可独立运行 |
| SFT 含 revision | 3 个错误→反馈→改进样例 | 模型需要学会使用 scorer 反馈 |
| 树迭代 | BFS 逐层，不设折扣 | 子节点改进父节点，不是一次性写好 |
| GPU 调度 | scorer num_gpus=0 | 避免 GPU 碎片化导致 veRL 调度死锁 |
