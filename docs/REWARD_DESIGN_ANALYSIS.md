# Reward & Advantage 设计方案 — 深度分析与修正计划

> 2026-05-18 | 状态: 分析完成，待实现

---

## 1. 类似工作调研

### 1.1 TritonForge (RLsys-Foundation)

与我们的项目最接近——SFT + RL 训练模型写 Triton kernel。

- 用 SLIME 框架（PPO + KL），不是 GRPO
- Reward 三组件：compilation + correctness + speedup
- SFT 阶段过滤编译失败样本 → RL 基线编译率 >90%
- 每个 kernel 最多 3 轮迭代，折扣 γ=0.4——推动模型在早期轮次解决问题
- **不给负 reward**：编译失败给 0 分而非负值
- 启示：SFT 质量决定 RL 起点

> **Q1: SLIME 框架是什么？相比GRPO有什么优势？可以解决组内给出相同质量差代码无法区分的问题？**
>
> SLIME（THUDM 开源）是在标准 PPO 基础上专门为代码生成定制的 RL 框架。核心区别：
>
> | | SLIME (PPO-based) | GRPO |
> |------|------|------|
> | Advantage 计算 | 用 critic network 估计 value，`A = R - V(s)` | 组内对比，`A = (r - mean) / std` |
> | 是否需要 critic | 是 | 否（组内自归一化） |
> | 对 group 多样性依赖 | 低——critic 给出绝对值，不受组内其他样本影响 | 高——组内无差异则 A=0 |
> | All-zero group 处理 | critic 可以学习到"当前状态全是低分"，给出低 value | 无法处理，std=0 产生 NaN |
>
> **能解决组内相同质量差代码无法区分的问题吗？**
> 理论上可以——PPO 的 critic 模型学习的是"给定 prompt，预期的 return 是多少"。即使所有 response 都是零分，critic 也能输出一个低 value 估计，不会产生 NaN。但这是以增加一个 critic 模型（额外显存和训练开销）为代价的。我们当前用 GRPO 是为了简洁，代价就是组内无区分时失效。折中方案：all-zero group 时回退到 generational 优势值，避开 NaN。

> **Q2: TritonForge 的折扣机制是什么？为什么最多3轮？**
>
> 折扣 γ=0.4 是 RL 标准的累计回报折现：`G = r_1 + γ*r_2 + γ²*r_3`。在 TritonForge 的多轮迭代场景中，模型有 3 次机会改进同一个 kernel（第1轮→第2轮→第3轮），每往后一轮的 reward 权重打 4 折。设计目的：迫使模型在第一轮就尽量写好，而不是拖到最后一轮才给出正确代码。
>
> 为什么最多 3 轮？因为 γ³≈0.064 已经很小，第 4 轮的贡献微乎其微。而且树深度 3 对 kernel 任务来说已经足够——第一轮写基本实现，第二轮修正编译错误，第三轮优化性能。匹配我们的 tree depth=2-3 设计。

### 1.2 DeepSeek-R1

- SFT 质量决定 RL 收敛上限
- Verifiable reward only：只对可客观验证的结果给分
- GRPO 有效前提是 group 内多样性——所有样本同一模式错误时 GRPO 失效
- 冷启动 SFT 防止 RL 收敛到低质量局部最优

> **Q3: 我设计的reward还算客观吧？**
>
> 大部分是客观的，具体分析：
>
> | 段 | 信号来源 | 客观？ | 备注 |
> |------|------|------|------|
> | code (编译) | compiler 返回码 | ✅ 客观 | 编译成功/失败是二进制事实 |
> | code (正确性) | torch.allclose vs ref | ✅ 客观 | 数值误差是可计算的事实 |
> | code (性能) | CUDA event speedup | ✅ 客观 | 墙钟时间测量 |
> | design | 当前是正则匹配关键词 | ❌ 不客观 | 这是问题所在——需要改为用 code 的性能结果反推 design 质量 |
> | predict | 预测 vs 实测偏差 | ✅ 客观 | 偏差是可计算的 |
>
> design 的客观性需要改进：不靠正则匹配关键词，而是用 "design 中声称的策略是否在 code 中实现 + 该策略对应的性能指标是否达到 → design 质量"。比如 design 说了 block_size=256 且 code 确实用了 256，且 speedup>1.0 → design 有信息量 → 给分。

> **Q4: 能否通过增加输出多样性解决 all-zero group 问题？**
>
> 你的直觉是对的——仅靠提高温度解决不了根本问题。当模型的能力上限是"任何 prompt 下都写不出正确代码"时，提高温度只是让模型在"写错的代码A"和"写错的代码B"之间随机，不会出现正确的代码。多样性需要的是**样本空间中出现至少一个好的**，不是让差的样本之间更不同。
>
> GRPO 区分不开好坏的真正原因不是多样性不够，而是**基线太低**（SFT 没有让模型学会正确 Triton）。解决方案分两步：① 重新 SFT，用高质量数据打高基线；② RL 阶段即使全部编译失败，generational 优势值仍能从父子差异中给出信号。

> **Q5: 冷启动 SFT是什么意思？**
>
> 指在 RL 之前先做一次**高质量**的 SFT。DeepSeek-R1 的做法是：收集几千条高质量 chain-of-thought → 先 SFT → 再 RL。对比：直接从 base model 开始 RL（热启动），模型完全不会 reasoning，RL 从零探索。冷启动 = 先灌注基础知识，RL 在基础上优化。我们当前的 800 例 KernelBook SFT 就是冷启动，但数据质量不够高（含 inductor 模式）。

> **Q6: entropy bonus是专门给增加熵的奖励？**
>
> 是的。标准 PPO loss = policy_loss - entropy_coeff * H(π)。其中 H(π) 是策略的熵（token 分布的均匀程度）。entropy_coeff > 0 表示奖励模型保持高熵（不要过早确定性地选某个 token），鼓励探索。当前配置 `entropy_coeff: 0.001` 是 veRL 的默认值，很小，几乎不影响训练。如果模型总是选相同 token（低熵），可以增大这个系数。

### 1.3 通用 RL for Code 经验

| 问题 | 常见解决方案 |
|------|------|
| 编译失败 | 给 0 而非负值，用编译错误作 feedback text |
| Reward hacking | AST 检查确保使用目标框架 |
| 探索不足 | 高 temperature + entropy bonus |
| All-zero group | SFT 提高基线编译率 |

---

## 2. 当前输出的真实问题（5/16 E2E）

**问题 A：所有代码都是 inductor 模式**
`triton_poi_fused_*` + `libdevice.*` + `empty_strided_cuda`。根源：KernelBook SFT 数据的 triton 代码由 torch inductor 生成。

> **Q7: inductor 模式是什么？我们需要改成什么模式？**
>
> torch._inductor 是 PyTorch 2.0 的 JIT 编译器——它把 `torch.nn.functional.gelu(x)` 自动编译成 Triton kernel。这个生成的 kernel 使用 inductor 的内部运行时（`empty_strided_cuda`, `assert_size_stride`, `libdevice.erf` 等），**不是人类手写的 Triton 代码**。它的特点：
> - 函数名是自动生成的（`triton_poi_fused_add_div_erf_mul_0`）
> - 使用 `grid()` 宏而非手写 `grid=(n, )`
> - 包含 stride 断言、CUDA device guard 等 inductor runtime 代码
> - kernel 参数带有 `XBLOCK: tl.constexpr` 这种 inductor 特有模式
>
> 我们需要改成**手写 Triton 模式**：
> ```python
> @triton.jit
> def gelu_kernel(in_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
>     pid = tl.program_id(0)
>     offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
>     mask = offs < n
>     x = tl.load(in_ptr + offs, mask=mask)
>     # GELU: x * 0.5 * (1 + erf(x/sqrt(2)))
>     sqrt2 = 1.4142135623730951
>     y = 0.5 * x * (1.0 + tl.math.erf(x / sqrt2))
>     tl.store(out_ptr + offs, y, mask=mask)
>
> def gelu(x):
>     out = torch.empty_like(x)
>     grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']),)
>     gelu_kernel[grid](x, out, x.numel(), BLOCK_SIZE=1024)
>     return out
> ```
> 区别：手写模式有明确的 launch grid、wrapper 函数、无 inductor runtime 依赖。

**问题 B：正确性全部失败**
12/16 节点编译通过但 scorer 调用时签名不匹配。种子任务 `input_gen` 参数数量与 `reference_python` 不一致。

> **Q8: 签名不匹配能否在prompt中给出格式要求？**
>
> 可以，而且应该。KernelBench 的每个问题都有标准格式的 `reference_python`，包含了完整的函数签名和调用方式。在 prompt 中明确要求模型按照 reference 的函数签名来写 wrapper，比如：
> ```
> Your wrapper function MUST be named exactly as specified in the
> reference. It must accept the same arguments in the same order.
> The scorer will call: output = your_function(*gen_inputs())
> ```
> 这样模型知道 scorer 会怎么调用，避免签名不匹配。

**问题 C：Design 是 SFT 模板**
所有节点 design 几乎一模一样（auto-generated），RL 一步未改变。

**问题 D：所有 reward 相同**
code=0.3 无区分 → GRPO 无信号 → 梯度未改变模型行为。

---

## 3. 当前实现的具体问题

### Bug #1: gen_adv 双重 Masking

`advantage.py` L77-85：gen_adv 被 `section_mask` 乘两次，GRPO 只乘一次。

> **Q9: gen_adv 被 section_mask 乘两次，这是什么问题？**
>
> 看源码 `advantage.py`：
> ```python
> # L77-85: 在 section loop 内
> section_advantages = (
>     grpo_weight * section_advantages.float()
>     + gen_weight * gen_adv.float() * section_mask    # ← gen_adv × mask (第1次)
> )
>
> # L87-89: 在 section loop 外，merge 时
> merged_advantages = merged_advantages + section_advantages.float() * section_mask  # ← × mask (第2次！)
> ```
> 这意味着 gen_adv 的有效值是 `gen_weight * gen_adv * section_mask^2`，而 GRPO 的优势值只被乘了一次 mask (`grpo_weight * adv_grpo * section_mask^1`)。因为 section_mask 的值是 0/1 的二值矩阵，`mask^2 === mask`，所以**数学上不影响最终结果**——这是一个无害的冗余，不是真正的 bug。但它说明代码写得不够清晰，容易误解。应该改为只在 merge 时乘一次 mask，loop 内不提前乘。

### 问题 #2: 尺度不一致（核心）

- GRPO 返回 z-score：mean≈0, std≈1
- Generational 返回比值：`(r_child - r_parent) / (r_parent + eps)`，r_parent→0 时爆炸
- 固定 0.5/0.5 权重——一方方差可能完全主导

> **Q10: 方差修正这部分，我实际是希望方差大的不要影响过大，应该二者均衡，不要一家独大。**
>
> 理解了。你的意思是**阻尼**大方差信号，而不是放大。修正方案改为：
> ```python
> # 归一化双方到单位方差，然后等权合并（均分影响力）
> grpo_std = adv_grpo.std().clamp(min=1e-6)
> gen_std  = adv_gen.std().clamp(min=1e-6)
> adv_grpo_norm = adv_grpo / grpo_std          # 归一化：方差=1
> adv_gen_norm  = adv_gen / gen_std             # 归一化：方差=1
> adv_combined = 0.5 * adv_grpo_norm + 0.5 * adv_gen_norm
> ```
> 这样无论原始方差相差多少倍，归一化后双方精确均分影响力。可以用固定 0.5/0.5 权重，因为归一化已经消除了量纲差异。

### 问题 #3: Generational 比值爆炸

已确认：改用绝对差值 ÷ baseline。

### 问题 #4: Predict "预测编译失败"成为稳定策略

> **Q11: 用户意见：学到稳定预测编译失败不是问题，code和predict是分开的优势值。**
>
> 你说得对。GRPO 会分别计算 code_advantages 和 predict_advantages，三段独立。模型可以 code=0.05 + predict=+0.1 = 0.15，但另一个节点 code=0.7 + predict=0 = 0.7，GRPO 会给后者更高的 code 优势值，给前者更高的 predict 优势值。最终模型同时优化两段，不会因 predict 高而牺牲 code。
>
> 所以原来的"修正"是不必要的。Predict 可以在任何时候给分，不影响 code 段的独立优化。撤回 predict 的条件限制。

### 问题 #5: Design 脱离 code 独立高分

> **Q12: 用户意见：design给分是按性能指标给分，不是按关键词匹配给分。**
>
> 同意。之前的分析基于当前代码中 `design_reward()` 的正则匹配实现（检查 "coalesced", "BLOCK_SIZE" 等关键词）——这是实现问题，不是设计问题。Design 的正确做法是按客观性能指标给分：design 中声称的策略 → code 中是否实现 → 实测性能是否支持。所以 design_reward 的上限不需要人为加 `≤ 0.6 * code` 的约束——如果 design 真正描述了成功策略，它应该与 code 的质量自然相关。撤回这个约束。

### 问题 #6: All-Zero Group

暂不解决。保持现有 `0.01 * randn` 兜底 + future all-zero 回退方案。

### 问题 #7: 代码过短阈值不合理

> **Q13: AST检查是不是可以有比较好的结果？**
>
> 是的。Python 的 `ast` 模块可以精确解析代码结构，比正则可靠得多：
> ```python
> import ast
> tree = ast.parse(code_text)
> # 检查是否有 @triton.jit 装饰的函数
> has_triton_kernel = any(
>     isinstance(node, ast.FunctionDef) and
>     any('triton.jit' in ast.unparse(d) for d in node.decorator_list)
>     for node in ast.walk(tree)
> )
> # 检查 wrapper 函数是否包含 Triton kernel launch
> has_grid_launch = any(
>     'grid' in ast.unparse(node) for node in ast.walk(tree)
>     if isinstance(node, ast.Call)
> )
> ```
> AST 的优势：不受空格、注释、字符串内容干扰；可以精确判断函数调用关系；可以区分 "Triton kernel 被 launch 了" vs "Triton kernel 被定义了但没被调用"。建议用 AST 替代正则做 anti-cheat 检测。

---

## 4. 修正后的三部分 Reward（根据讨论更新）

| 段 | 正常范围 | 惩罚 | 生效条件 | 讨论结论 |
|------|------|------|------|------|
| code | 0.05-0.70 | -0.3 (AST detected Python cheat), -0.2 (compile fail + no @triton.jit) | 总是 | 增加 correctness 比例分（pass_count/total） |
| design | 0.0-0.4 | 无 | 总是 | 按性能指标给分，不设 ×0.6 上限 |
| predict | 0.0-0.3 | -0.05 (偏差>2×) | **总是**（撤回条件限制） | code/predict 独立优势值互不影响 |

**Code Reward S-Curve：**
```
compile fail (tried):      0.05
compile fail (no @triton): -0.20 (penalty)
compile OK + incorrect:     0.15
compile OK + correct:
  speedup=0.5 → 0.20     speedup=1.0 → 0.30     speedup=1.5 → 0.52
  speedup=2.0 → 0.65     speedup=3.0 → 0.69     speedup=10+ → 0.70 (cap)
```
公式：`r = 0.3 + sigmoid(4.0 * (speedup - 1.2)) * 0.4`

---

## 5. 修正后的 Advantage 流程（根据讨论更新）

```
Step 1: sample_rewards = (token_scores * mask).sum(-1) / mask.sum(-1)
Step 2: adv_grpo = compute_grpo_outcome_advantage(...)  [z-score]
Step 3: adv_gen = (r_child - r_parent) / 0.3            [绝对差值÷baseline]
Step 4: 归一化双方到单位方差，等权合并
        adv_grpo_norm = adv_grpo / adv_grpo.std()
        adv_gen_norm  = adv_gen / adv_gen.std()
        adv_combined  = 0.5 * adv_grpo_norm + 0.5 * adv_gen_norm
Step 5: 统一乘一次 section_mask（修双重masking冗余）
```

---

## 6. 待修改文件

| 文件 | 改动 |
|------|------|
| `scoring/reward.py` | predict 撤回条件限制（总是生效）; design 撤回 ×0.6; anti-cheat 改用 AST |
| `advantage.py` | 归一化到单位方差后等权合并; 清理双重masking冗余; gen 用绝对差值÷baseline; all-zero 保留 randn 兜底 |
| `prompt_builder.py` | 增加反 inductor 规则 + 函数签名匹配要求 |
| `data/kernel_rl/train.parquet` | 修 input_gen 匹配; 扩增 KernelBench 任务 |
| SFT 数据 | 重新生成，去 inductor 化，用 KernelBench 问题 + 手写 Triton 代码 |

## 7. 参考

- TritonForge: github.com/RLsys-Foundation/TritonForge
- KernelBench: github.com/ScalingIntelligence/KernelBench
- DeepSeek-R1: arXiv 2501.12948
- SLIME: github.com/THUDM/SLIME
