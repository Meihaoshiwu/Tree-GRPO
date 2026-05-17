# RL 训练操作指南 & 奖励/优势值设计方案

> 本文档保留在本地，不提交 git。供离线 H200 服务器手动操作参考。

---

## 1. 手动启动训练

### 1.1 准备环境

```bash
# 激活环境（根据实际服务器调整）
cd /inspire/qb-ilm/project/wuliqifa/public/sdt/Tree-GRPO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
```

### 1.2 启动训练

```bash
# 单步验证（同 E2E）
bash scripts/kernel_rl/run_e2e.sh

# 多步 RL 训练（修改配置后）
python -m verl.trainer.main_ppo_kernel \
    hydra.run.dir=./outputs/hydra/$(date +%Y%m%d_%H%M%S)
```

### 1.3 监控训练进展

**查看训练 metrics（每步更新）：**
```bash
tail -f scripts/kernel_rl/e2e_run_*.log | grep "actor/pg_loss\|critic/rewards\|response_length"
```

**关键指标含义：**

| Metric | 含义 | 健康范围 |
|------|------|------|
| `response_length/mean` | 模型平均生成长度 | 500-1024 |
| `critic/rewards/mean` | 平均 reward | 逐步上升 |
| `actor/pg_loss_code` | code 段 PPO loss | 负值为好（advantage 为正） |
| `actor/grad_norm` | 梯度范数 | 0.01-1.0 |
| `timing_s/step` | 每步耗时 | 60-120s |

**查看 tree 日志（每个节点）：**
```bash
# 列出所有树
ls outputs/kernel_tree_logs/

# 查看某个节点的完整 prompt + response + scorer 反馈
cat outputs/kernel_tree_logs/tree_kernel_00/d1_n0.txt
```

**查看 scorer 日志（执行结果）：**
```bash
cat outputs/kernel_scorer_logs/scorer_pid*.jsonl | python3 -m json.tool | head -30
```

### 1.4 中断和恢复

```bash
# 停止训练
pkill -f main_ppo_kernel
ray stop -f

# 清理临时文件
rm -rf /tmp/kernel_rl_checkpoints /tmp/ray
```

---

## 2. H200 × 8 适配

### 2.1 关键差异

| 参数 | RTX 4090 × 4（当前） | H200 × 8（目标） |
|------|------|------|
| 单卡显存 | 48 GB | 141 GB |
| 显存带宽 | 1.0 TB/s | 4.8 TB/s |
| FP16 TFLOPS | 82.6 | 990 |
| NVLink | 无 | 有 (900 GB/s) |
| `tensor_model_parallel_size` | 1 | 可设为 2 或 4 |
| `gpu_memory_utilization` | 0.6 | 0.8⁻0.9 |
| `ppo_max_token_len_per_gpu` | 8192 | 32768 |
| `train_batch_size` | 4 | 8⁻16 |
| `ppo_micro_batch_size` | 4 | 8⁻16 |
| `max_num_seqs` | 32 | 128 |

### 2.2 需要修改的配置

```yaml
# ppo_trainer_kernel.yaml (H200 版本)
actor_rollout_ref:
  rollout:
    gpu_memory_utilization: 0.85     # H200 显存大，多给 vLLM
    tensor_model_parallel_size: 2     # 启用 TP 分布式 KV cache
    max_num_batched_tokens: 16384
    max_num_seqs: 128
  actor:
    ppo_max_token_len_per_gpu: 32768
    ppo_micro_batch_size: 8
    ppo_micro_batch_size_fixed: 8
  ref:
    log_prob_micro_batch_size: 8      # 与 actor 对齐

critic:
  ppo_max_token_len_per_gpu: 32768
  forward_max_token_len_per_gpu: 32768

data:
  train_batch_size: 8                 # 8 GPUs × 1 sample = 8

trainer:
  n_gpus_per_node: 8
```

### 2.3 H200 额外优化

- **FlashAttention**: H200 的 H100 架构支持 FA2/FA3，配置中已启用
- **NVLink**: `tensor_model_parallel_size >= 2` 时 KV cache 跨 GPU 共享
- **FP8**: H200 支持 FP8 推理加速，vLLM 配置可加 `quantization: fp8`
- **torch.compile 模式**: H200 上 `torch.compile(mode="max-autotune")` 效果更好

---

## 3. 种子任务：KernelBench Level 1

### 3.1 任务来源

从 KernelBench Level 1（100 问题）精选 20-30 个适合 Triton 的任务：

**Element-wise（8 个）:**
19_ReLU, 20_LeakyReLU, 25_Swish, 26_GELU, 27_SELU, 21_Sigmoid, 22_Tanh, 29_Softplus

**Normalization（6 个）:**
36_RMSNorm, 40_LayerNorm, 33_BatchNorm, 34_InstanceNorm, 35_GroupNorm, 39_L2Norm

**Reduction（6 个）:**
47_Sum_reduction, 48_Mean_reduction, 49_Max_reduction, 23_Softmax, 51_Argmax, 89_cumsum

**Matmul（5 个）:**
1_Square_matmul, 2_Standard_matmul, 3_Batched_matmul, 16_Matmul_transposed_A, 17_Matmul_transposed_B

**Pooling（3 个）:**
42_Max_Pooling_2D, 45_Average_Pooling_2D, 44_Average_Pooling_1D

**Loss/Attention（2 个）:**
94_MSELoss, 97_ScaledDotProductAttention

### 3.2 数据格式

每个 KernelBench 问题需转换为我们的 `train.parquet` 格式：

```python
{
    "task_spec": {
        "name": "26_GELU",
        "description": "GELU activation function...",
        # ... structured fields
    },
    "bench_spec": {
        "metric": "speedup",
        "target_speedup": 2.0,
        "input_gen": "torch.randn(4096, dtype=torch.float32)",
        "ref_code": "class Model(nn.Module):\n    def forward(self, x):\n        return F.gelu(x)\n\ndef get_inputs():\n    return [torch.randn(4096)]\n\ndef get_init_inputs():\n    return []",
    },
    "reference_python": "def gelu(x):\n    return F.gelu(x)",
    "extra_info": {"index": "kernel_KB_26", "difficulty": "easy", "category": "activation"}
}
```

`bench_spec.ref_code` 会被 scorer 的 `_build_ref_code()` 加载为参考实现 + 输入生成器。

### 3.3 torch.compile baseline 实现

对于每个 KernelBench 任务，scorer 会自动计算 torch.compile 版本的性能作为 baseline：

```python
# 在 scorer 中对比
baseline_fn = torch.compile(ref_model, mode="reduce-overhead")
baseline_time = measure(baseline_fn, inputs)
triton_time = measure(triton_model, inputs)
speedup = baseline_time / triton_time
```

这确保我们始终与 PyTorch 官方最优性能对比。

---

## 4. 奖励函数设计

### 4.1 设计原则

1. **不给平庸代码惩罚**：编译通过 + 正确但性能差的代码，至少不扣分。避免抑制模型探索。
2. **两种必须惩罚的情况**：编译失败且输出很短（< ref 代码长度）→ 明显作弊或放弃；调用了 PyTorch 但 Triton kernel 从未被调用 → 假 Triton
3. **加速比非线性映射**：低加速比平滑（鼓励尝试），中等加速比陡峭（区分好坏），极高加速比封顶（防 reward hacking）
4. **三段独立**：各自用自己的指标，避免一段的失败拖累其他段

### 4.2 Code Reward — 基于加速比的非线函数

```python
def code_reward(compiled, correctness, speedup, is_cheating=False, code_len=0, ref_len=0):
    """Code reward: only positive rewards, two penalty exceptions."""
    
    # Penalty 1: compiled but calling PyTorch instead of Triton
    if is_cheating:
        return -0.3
    
    # Penalty 2: compile failed + output suspiciously short
    if not compiled and code_len < ref_len and code_len > 0:
        return -0.2
    
    # Not compiled but at least tried → tiny reward for effort
    if not compiled:
        return 0.05
    
    # Compiled but incorrect → small reward for compiling
    if not correctness:
        return 0.15
    
    # Compiled + correct → speedup-based reward
    # S-curve: flat at low speedup, steep around target, capped at ~2.5x
    target = 2.0        # target speedup
    k = 4.0             # steepness
    midpoint = 1.2      # center of steep region
    
    if speedup <= 0.5:
        # Very slow → barely better than incorrect (reward was 0.15)
        return 0.2
    elif speedup < 0.8:
        # Below PyTorch but correct → modest reward
        return 0.25
    else:
        # S-curve for speedup >= 0.8
        sigmoid = 1.0 / (1.0 + math.exp(-k * (speedup - midpoint)))
        # Map [~0.15, 0.98] → [0.3, 0.7]
        return 0.3 + sigmoid * 0.4
    
    # Max reward: ~0.7 (speedup ~3.0x)
    # Never exceeds 0.7 — cap prevents reward hacking
```

**曲线特点：**
- speedup=0.5: r≈0.2（慢但正确 → 比编译失败好一点）
- speedup=1.0: r≈0.3（持平 PyTorch → 基线）
- speedup=1.5: r≈0.52（明显更快 → 陡峭上升）
- speedup=2.0: r≈0.65（2倍加速 → 高分）
- speedup=3.0: r≈0.69（接近上限）
- speedup=10+: r≈0.70（封顶，防 reward hacking）

### 4.3 Design Reward — 基于策略与实现的一致性

```python
def design_reward(design_text, code_text, actual_metrics):
    """Design reward: does the claimed strategy match implementation?"""
    score = 0.0
    
    # 1. Block size claim vs actual (0.1)
    claimed_bs = extract_block_size_from_design(design_text)
    actual_bs = extract_block_size_from_code(code_text)
    if claimed_bs and actual_bs and abs(claimed_bs - actual_bs) <= 0:
        score += 0.1
    
    # 2. Memory pattern claim vs actual (0.1)
    if "coalesced" in design_text.lower() and has_coalesced_access(code_text):
        score += 0.1
    
    # 3. Bottleneck identification (0.1)
    claimed_bottleneck = extract_bottleneck(design_text)
    actual_bottleneck = actual_metrics.get("bottleneck", "unknown")
    if claimed_bottleneck == actual_bottleneck:
        score += 0.1
    
    # 4. Has specific optimization rationale (not generic) (0.1)
    if design_is_specific(design_text):
        score += 0.1
    
    return min(score, 0.4)  # cap at 0.4
```

### 4.4 Predict Reward — 逐指标给分

```python
def predict_reward(predict_text, actual_metrics):
    """Predict reward: binary correctness per predicted metric."""
    score = 0.0
    predictions = parse_predictions(predict_text)
    
    # Metric 1: compile prediction correct (0.1)
    if "will_compile" in predictions:
        score += 0.1 if predictions["will_compile"] == actual_metrics["compiled"] else -0.05
    
    # Metric 2: speedup prediction accuracy (0.15)
    if "speedup" in predictions:
        pred_sp = predictions["speedup"]
        actual_sp = actual_metrics.get("speedup", 0)
        if actual_sp > 0:
            error_ratio = abs(pred_sp - actual_sp) / max(pred_sp, actual_sp, 1.0)
            if error_ratio < 0.2:
                score += 0.15       # very accurate
            elif error_ratio < 0.5:
                score += 0.08       # somewhat accurate
            elif error_ratio > 2.0:
                score -= 0.05       # wildly wrong → penalty
            # else: 0 (neutral)
    
    # Metric 3: bottleneck identification (0.05)
    if "bottleneck" in predictions:
        if predictions["bottleneck"] == actual_metrics.get("bottleneck"):
            score += 0.05
    
    return max(min(score, 0.3), -0.1)  # cap at 0.3, floor at -0.1
```

### 4.5 奖励函数汇总

| 段 | 正常范围 | 惩罚（两种例外） | 上限 |
|------|------|------|------|
| code | 0.05-0.70 | -0.3（调用 PyTorch 作弊）, -0.2（compile fail + 代码过短） | 0.70 |
| design | 0.0-0.4 | 无惩罚 | 0.40 |
| predict | -0.1-0.3 | 偏差>2× 给 -0.05 | 0.30 |

---

## 5. 优势值设计

### 5.1 两部分优势值

```
最终的 token-level advantage = w_grpo × A_grpo + w_gen × A_generational
```

**w_grpo = 0.5, w_gen = 0.5**（等权重，确保数量级一致）

### 5.2 GRPO 同深度优势

GRPO 在同一深度的所有节点之间计算相对优势：

```python
# 对于深度 d 的所有节点 {n_1, n_2, ..., n_k}
# reward = {r_1, r_2, ..., r_k}
# group_mean = mean(r_1...r_k)
# group_std = std(r_1...r_k)

# GRPO advantage: (r_i - group_mean) / (group_std + epsilon)
A_grpo_i = (r_i - group_mean) / (group_std + 1e-8)
```

特点：同深度表现好的节点得到正优势值，差的得到负优势值，自动归一化。

### 5.3 代际优势

子节点相对于父节点的改进比例：

```python
# 对于节点 n 和其父节点 parent(n):
# r_child = reward(n)
# r_parent = reward(parent(n))

# Generational advantage:
A_gen = (r_child - r_parent) / (r_parent + epsilon)
```

特点：
- 子节点比父节点好 → 正优势值
- 子节点不如父节点 → 负优势值
- 根节点（无父节点）→ 代际优势 = 0

### 5.4 权重平衡

两个优势值必须在同一数量级。GRPO 的归一化后范围约为 [-2, +2]，代际比值范围约为 [-1, +1]。

```python
# 在 advantage.py 中实现
def compute_combined_advantages(rewards, node_depths, parent_map):
    # GRPO: same-depth comparison
    for depth in unique_depths:
        depth_rewards = [r for r, d in zip(rewards, node_depths) if d == depth]
        depth_mean = mean(depth_rewards)
        depth_std = std(depth_rewards)
        for i in depth_indices:
            adv_grpo[i] = (rewards[i] - depth_mean) / (depth_std + 1e-8)
    
    # Generational: child vs parent
    for i, node in enumerate(nodes):
        if node.parent_uid is not None:
            parent_reward = parent_reward_map[node.parent_uid]
            adv_gen[i] = (rewards[i] - parent_reward) / (parent_reward + 1e-8)
        else:
            adv_gen[i] = 0.0  # root nodes
    
    # Combined (balanced weights)
    total_adv = 0.5 * adv_grpo + 0.5 * adv_gen
    
    # Epsilon-clip to prevent extreme values
    return clip(total_adv, -1.0, 1.0)
```

### 5.5 所有节点都给出错误代码时的处理

当同一深度的所有节点 score=0 时，GRPO 的 std=0，无法计算优势值。处理方式：

```python
if group_std < 1e-6:
    # All nodes same reward → GRPO advantage = 0
    # Rely on generational advantage (which may also be 0 if all are roots)
    # Add small exploration bonus: tiny random noise
    adv_grpo = torch.zeros_like(rewards) + 0.01 * torch.randn_like(rewards)
```

这确保即使全部节点都差，仍有一点随机梯度来驱动探索。

---

## 6. 防作弊检测

### 6.1 Python 调用检测

```python
def detect_python_cheating(code_text):
    """Return True if the code claims to be Triton but uses PyTorch ops."""
    # Check 1: code has @triton.jit decorated function
    has_triton_kernel = bool(re.search(r'@triton\.jit', code_text))
    
    # Check 2: wrapper function that's supposed to call the Triton kernel
    # actually calls PyTorch ops instead
    wrapper_fn = extract_wrapper_function(code_text)
    if wrapper_fn:
        pytorch_calls = re.findall(r'torch\.(nn\.|functional\.|Tensor\.|sum\(|mean\(|matmul\()', wrapper_fn)
        triton_launch = re.findall(r'\[grid\]\(|kernel\[', wrapper_fn)
        # Has PyTorch calls but no Triton kernel launch → cheating
        if pytorch_calls and not triton_launch:
            return True
    
    return False
```

### 6.2 代码长度检测

```python
def is_too_short(code_text, reference_python):
    """Return True if code is suspiciously short."""
    # Less than reference code length AND compile failed
    code_lines = [l for l in code_text.split('\n') if l.strip() and not l.strip().startswith('#')]
    ref_lines = [l for l in reference_python.split('\n') if l.strip()]
    return len(code_lines) < len(ref_lines)
```

### 6.3 成功范式参考

**TritonForge 的做法（RLsys-Foundation）：**
- SFT 阶段过滤掉所有编译失败的样本
- RL 阶段用 `compilation + correctness + speedup` 三信号 reward
- 多轮迭代（最多 3 轮），每轮都会收到编译错误反馈
- 折扣因子 γ=0.4 让模型倾向于在前面轮次就解决问题

**SPIN / DPO 范式的启示：**
- 不给负 reward（负 reward 会让模型减少输出，而非改进质量）
- 用 relative comparison（组内对比）代替 absolute scoring 来避免 reward hacking

我们综合两者：不给负 reward（除作弊和放弃外），用 GRPO 组内对比 + 代际对比提供优势值差异。

---

## 7. 训练目标验证

### 7.1 期望的模型行为

经过 RL 训练后，每个树节点应能输出：
- Design：包含具体策略描述（block size 选择依据、内存布局决策）
- Code：独立的 Triton kernel，不使用 PyTorch 回退
- Predict：包含编译预测、加速比预估、瓶颈分析

性能目标：
- 编译通过率 > 80%
- 数值正确率 > 50%
- 平均加速比 > 1.0（至少不比 PyTorch 差）
- 最高加速比 > 2.0（部分 kernel 明显优于 PyTorch）

### 7.2 验证流程

```bash
# 1. 启动训练
bash scripts/kernel_rl/run_e2e.sh

# 2. 监控指标
tail -f scripts/kernel_rl/e2e_run_*.log | grep "actor/pg_loss\|rewards/mean"

# 3. 查看生成代码质量
cat outputs/kernel_tree_logs/tree_kernel_00/d1_n0.txt

# 4. 检查作弊率
grep -l "is_cheating" outputs/kernel_scorer_logs/*.jsonl | wc -l
```

---

## 8. 文件清单

| 文件 | 作用 |
|------|------|
| `scripts/kernel_rl/run_e2e.sh` | 启动训练（单步验证） |
| `outputs/kernel_tree_logs/` | 每个节点的完整 prompt + response + scorer 反馈 |
| `outputs/kernel_scorer_logs/` | JSONL 执行结果（node_uid + rewards + metrics） |
| `docs/RL_TRAINING_GUIDE.md` | 本文档（本地，不提交） |
| `docs/vllm_kvcache_memory.md` | vLLM KV cache 分析（已提交） |
