# SFT & RL 策略深度讨论

> 2026-05-18 | 基于 Q&A 讨论记录

---

## 1. PPO 绝对打分作为 GRPO 失败时的保底

### 用户问题
能否不依赖 critic 模型给出绝对打分，作为 GRPO 失败时（all-zero group）的保底选项？

### 回答

**可以，且不需要 critic network。** 关键洞察：我们的 scorer 本身就能给出绝对 reward（编译分数的 0.05/0.15/0.30 本来就是绝对值）。GRPO 的组内对比 `(r - mean)/std` 在 std=0 时失效，但 scorer 的原始 reward 仍然有意义。

```python
# 保底方案（无需 critic）
if group_std < 0.05:
    # GRPO 失效 → 回退到绝对 reward
    adv = (sample_reward - 0.15) / 0.3  # 0.15=编译OK但不正确, 0.3=baseline
else:
    adv = (sample_reward - group_mean) / (group_std + eps)  # GRPO
```

这本质上是混合 advantage——在高方差时用 GRPO（组内对比），低方差时用绝对 baseline 对比（scorer reward - baseline）。不需要 critic model 学习 value function，比 PPO 轻量。

SCoRe 论文（2409.12917）的做法类似——他们用 reward bonus 来放大 self-correction 信号，防止 behavior collapse。

---

## 2. 模型初始生成能力与 RL 改进

### 用户问题
如果模型基线很差，RL 无法改进。我承认如果 baseline 差，RL 在错误代码间摆动。

### 回答

这正是你观察到的现象——所有节点 code=0.3（全编译失败），GRPO 无信号。这不是 RL 算法的问题，是**初始策略分布**的问题。

RL 的探索范围 = 初始分布 + 温度控制的采样噪声。如果初始分布里找不到任何正确代码，采样噪声（即使 temperature=2.0）也几乎不可能偶然生成正确代码——Triton 的语法空间太大。

**解决方案层次：**
1. **SFT 打高基线** → 让初始分布里 ≥30% 的样本能编译通过
2. **GRPO 开始工作** → 组内有好有坏，优势值有区分
3. **Generational advantage 兜底** → 即使全部编译失败，子节点比父节点稍微好一点也有信号

---

## 3. 衰减项与树搜索的兼容性

### 用户问题
γ=0.4 折扣不适合我的设计。我期望模型学习"改进错误"——在浅层输出的基础上给更好的代码，不是一次性写好。

### 回答

完全同意。TritonForge 用 γ=0.4 是因为他们的多轮迭代是**同一个 prompt 反复 refine**（第 2 轮看到第 1 轮的编译错误后修正），不是树形结构。

在你的树搜索中：
- 深度 0 → 深度 1 → 深度 2 代表迭代改进
- 子节点看到父节点的 response + scorer feedback 后才生成
- 正确定义：**子节点的 reward 应该与父节点比较，而不是打折**

所以你的代际优势值设计 `(r_child - r_parent)` 是正确的，折扣项 γ 不需要。

SCoRe 论文的发现支持这一点：他们强调**多轮 RL** 中，奖励应该激励两轮之间的**改进**（delta reward），而不仅仅是最终结果。这与你"在浅层基础上给更好代码"的理念一致。

---

## 4. SCoRe 论文核心洞见

**Training Language Models to Self-Correct via Reinforcement Learning** (DeepMind, 2024)

核心发现：
1. **SFT 无法教 self-correction**：因为 SFT 用离线数据训练，模型学的是"别人的错误→别人的修正"，而不是"我自己的错误→我的修正"。分布不匹配导致 collapse。
2. **RL 可以教 self-correction**：模型在自己的 generation 分布上做 multi-turn RL，第一轮生成 → 第二轮修正，RL 奖励改进幅度。
3. **关键 reward 设计**：奖励**修正幅度**而非最终得分。如果第一轮是错的，第二轮纠正了 → 高奖励。如果第一轮就对了，第二轮没改 → 低奖励（你没学到任何东西）。
4. **两阶段训练**：Phase 1 做基础 multi-turn RL 防止 collapse；Phase 2 用 reward bonus 放大 self-correction。

对我们的启示：
```
你的树搜索就是天然的多轮 self-correction 结构：
  深度 0: 模型第一次尝试 → 编译失败
  深度 1: 看到 feedback 后修正 → 编译通过但不正确
  深度 2: 再次修正 → 正确且性能好
  
  reward 应该奖励: d1 比 d0 好, d2 比 d1 好
  这正是你的 generational advantage 设计！
```

---

## 5. KernelBench Levels 与当前 SFT 数据

### 用户问题
KernelBench 有几个 level？我们用了哪个 level？能否只收录编译成功且性能好的大模型输出？

### 回答

KernelBench 有 **4 个 Level**：

| Level | 名称 | 问题数 | 难度 |
|------|------|------|------|
| 1 | 单算子（activations, norms, matmul, pooling） | 100 | 简单-中等 |
| 2 | 简单融合模式（Conv+ReLU, Matmul+Scale+Sigmoid） | 100 | 中等 |
| 3 | 完整模型架构（MobileNet, VGG, MiniGPT） | 50 | 困难 |
| 4 | HuggingFace 模型 | 可变 | 极难 |

**当前状态**：我们的 SFT（v1/v2/v3）都没直接用 KernelBench——v1 是手写 20 例，v2 是从 KernelBook (18k 例，inductor 代码) 生成的。

**你提出的 Code 数据策略是可行的**：
```
Step 1: 收集人类手写 Triton 代码（triton 官方 tutorial, KernelBench baseline 实现）
Step 2: 不足以覆盖的 kernel 类型 → 用大模型生成候选
Step 3: 拒绝采样：只保留 "编译通过 + 数值正确 + speedup > 0.5" 的大模型输出
Step 4: 组合人类数据 + 大模型高质量输出 → SFT code 数据集
```

Design 数据确实更稀缺——大模型可以生成，但质量难以自动验证。初期可以用我们的 auto-generator + 人工抽检。

---

## 6. 冷启动与泛化

### 用户问题
为什么叫"冷启动"？如果模型从未见过某个算子，RL 能学会吗？

### 回答

**为什么叫冷启动（Cold Start）？**
比喻自引擎——冷引擎直接高负荷运转会损坏。DeepSeek-R1 的"冷启动 SFT"意思是：在 RL 之前先给模型注入几千条高质量 reasoning 示例，让模型先学会基本的 reasoning 格式和能力，再开始 RL 探索。跟"冷引擎先预热"一个道理。

**模型从未见过的算子，RL 能学会吗？**
不能从头学。RL 的探索基于初始分布的采样。如果模型在预训练/SFT 中从未见过 `cumsum` 的 Triton 实现，它连 `cumsum` 需要 parallel scan 都不知道，RL 的随机采样不可能碰巧生成正确的 parallel scan 算法。

**但可以泛化**：
- 如果 SFT 教了 softmax 的 Triton 实现（three-pass reduction 模式）
- 模型可能将 reduction 模式泛化到未见过但结构相似的算子（如 log_softmax）
- 泛化质量取决于 SFT 覆盖的**模式多样性**

### Cold-Start Drafting 论文核心思想

> "Towards Cold-Start Drafting and Continual Refining: A Value-Driven Memory Approach with Application to NPU Kernel Synthesis"

这篇工作针对 NPU kernel 生成（比 GPU 更难，因为软硬件协同设计更复杂）：

1. **Cold-Start 阶段**：用 memory bank 存储高质量的 "kernel 模板"——不是完整的 kernel 代码，而是可复用的代码片段（tile 策略、buffer 分配方案、loop 结构）。这些模板来自人类专家的历史设计。
2. **Drafting 阶段**：给定新算子，从 memory bank 检索相关模板，组合成初步 kernel。
3. **Refining 阶段**：RL 在初步 kernel 上迭代优化（类似你的树搜索）。
4. **Value-driven**：用 value network 评估每次修改的预期收益，只保留有改进的修改。

**对我们的启示**：
- 我们的 SFT 数据相当于他们的 "memory bank"——不需要覆盖所有算子，但需要覆盖足够多的**代码模式**（element-wise 模式、reduction 模式、matmul tiling 模式等）
- 20 个手写 kernel（v3）覆盖了 ~10 种模式，不够。需要扩展到至少 50+ 模式
- 可以利用 KernelBench Level 1 的 100 个问题作为 prompt，生成每个问题的 baseline 实现（哪怕性能一般），让模型见过每种算子的基本 Triton 写法

---

## 7. torch._inductor 的真实性质

### 用户问题
torch._inductor 是内部表示还是真实 Triton 代码？任何难度的 triton 都可以通过 inductor 得出吗？

### 回答

**torch._inductor 生成的 Triton 代码是真实的，不是伪代码。** 但有几个关键限制：

1. **它是真实 Triton**：`triton_poi_fused_*` 函数由 `@triton.jit` 装饰，包含 `tl.load`/`tl.store`/`tl.arange`，可以在 GPU 上运行。它**不是** IR 或伪代码。

2. **但它依赖 inductor runtime**：`empty_strided_cuda`、`assert_size_stride`、`libdevice.erf` 这些是 inductor 的 helper，在标准 triton 环境中不可用。所以 inductor 生成的 Triton 代码**不是可独立运行的**——它必须在 inductor 的 runtime 上下文中执行。

3. **是否任何 PyTorch op 都能转换？不是。** inductor 支持一个不断增长的算子集合，但并非所有 PyTorch op 都有 Triton 实现。复杂的控制流、动态 shape、非标准 stride 可能导致 inductor fallback 到 eager 执行。

4. **代码质量**：inductor 生成的 Triton 代码是**正确的**（经过 PyTorch 测试），但**不一定是性能最优的**。它的 block size 选择是启发式的，不针对特定硬件调优。

5. **人读性差**：`triton_poi_fused_add_div_erf_mul_0` 这种函数名、`XBLOCK: tl.constexpr` 这种 inductor 特有 pattern、`libdevice` 调用——都是 machine-generated 的痕迹，不适合作为 learning target。

**结论**：inductor 代码不适合作为 SFT 数据。我们需要的是独立、可读、模式清晰的手写 Triton kernel。这就是 v3 数据的作用。

---

## 8. 下一步行动建议

基于以上讨论，优先级排序：

1. **SFT 数据 v3 扩展**——17 个模式扩展到 50+ 模式（用不同 kernel 类型和参数变体），确保覆盖主要 kernel pattern
2. **SFT 训练**——用 v3 数据全量微调，目标是消除 inductor 模式、建立手写 Triton 风格
3. **Bootstrapping**——SFT 后用模型生成 KernelBench 100 题的候选，scorer 过滤，扩充数据集
4. **GRPO + 绝对 baseline 保底**——实现 group_std < 阈值时回退到绝对 reward 对比
5. **多步 RL**——≥100 步，观察 loss 下降和 code 质量变化
