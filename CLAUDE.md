# CLAUDE.md

此文件为 Claude Code (claude.ai/code) 在此仓库中工作提供指导。

## 当前环境特殊配置
此环境是有root权限的docker环境，docker只有选择镜像、点击启动和保存权限，不能定制启动方式。环境由我个人使用，不需要用conda虚拟环境。

## 项目概述

本项目基于Tree-GRPO改造 — 主要改造方向为树状推理+面向算子开发的算子开发。基于veRL 构建。

**当前分支**：`feature/kernel-tree-ppo-phase1` — 将 Tree-GRPO 改造为 **Triton 算子开发的 RL 框架**，用真正的按层树推理替代原始 chain+采样的方式。

## 四进程架构（端到端设计）

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
│  ├─ Step3: CUDA event 性能测量 (speedup vs PyTorch)         │
│  ├─ (可选) SM 级采样: NSight Compute 硬件指标                │
│  └─ 返回 KernelScoreResult (feedback, scalar_rewards) → ①   │
│                                                            │
│  ① 主训练循环进程 (CPU, trainer 进程)                        │
│  ├─ KernelTreeSearchManager: 树结构维护                     │
│  │   ├─ BFS 逐层收集 prompt → tokenize → 组 batch            │
│  │   ├─ 调 ② vLLM 生成 → 解析 response → 创建子节点           │
│  │   ├─ 调 ③ scorer 评分 → 给节点赋 section rewards          │
│  │   └─ exporter: 每个非根节点 → 一个 PPO sample             │
│  ├─ compute_log_prob (当前策略 log prob)                    │
│  ├─ compute_multi_section_advantages (三路 advantage)       │
│  └─ 组装完整 DataProto → 送 ④                                │
│                                                            │
│  ④ 训练进程 (GPU, FSDP workers)                             │
│  ├─ 接收 DataProto (input_ids, masks, advantages, ...)     │
│  ├─ PPO 梯度更新，L = L_design + L_code + L_predict          │
│  └─ 权重同步，下一轮 rollout 自动使用新权重                    │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

**通信链路**：

| 步骤 | 发送方 | 接收方 | 内容 | 方式 |
|------|--------|--------|------|------|
| prompt → vLLM | ① tree_manager | ② vLLM workers | prompt batch + `n` | `wg.generate_sequences()` |
| response ← vLLM | ② vLLM workers | ① tree_manager | response_ids | 返回值 |
| code → scorer | ① tree_manager | ③ scorer actors | `KernelScoreRequest` | `pool.score_many()` |
| feedback ← scorer | ③ scorer actors | ① tree_manager | `KernelScoreResult` | Ray ObjectRef |
| node → sample | ① tree_manager | ① exporter | tree nodes | 本地调用 |
| sample → advantage | ① fit() | ① compute_multi_section_advantages | DataProto | 本地调用 |
| batch → training | ① fit() | ④ FSDP workers | DataProto | `wg.update_actor()` |

**关于②和④共享GPU**：在veRL的hybrid engine架构下，②rollout和④训练是同一组FSDP worker的不同方法调用。rollout时每个GPU持有完整模型副本做vLLM推理，训练时模型重新分片做FSDP梯度更新。这是veRL的设计，不是缺陷。

## 代码架构

### `search_r1/kernel_rl/` — 新增：Kernel RL 核心包

| 文件 | 作用 |
|------|------|
| `schema.py` | `KernelTreeNode`（稀疏树状态）、`KernelTrainSample`（dense PPO 样本）、`ParsedKernelResponse`、`KernelScoreResult` |
| `prompt_builder.py` | 构造 root/child prompt，强制 `design/code/predict` 三 section 输出 |
| `parser.py` | 解析 `<design>/<code>/<predict>` 标签，计算 char/token span，生成 section mask |
| `scorer.py` | `KernelScoringWorker` Ray actor + `KernelScoringPool`，调用 `scoring/` 模块执行真实评测 |
| `exporter.py` | 树节点 → PPO batch，每个非根节点导出一个独立样本 |
| `advantage.py` | Section-wise advantage：`design/code/predict` 分别算 advantage |
| `dataset.py` | Kernel dataset，支持 `task_spec/bench_spec/reference_python`，自动解析 JSON 字段 |
| `tree_manager.py` | Level-wise tree rollout manager，vLLM 原生 `n>1`，BFS 逐层推理，按树输出层级日志 |

### `search_r1/kernel_rl/scoring/` — Kernel 评分模块（独立目录）

基于 KernelBench (ICML'25, ScalingIntelligence) 评测框架改造，三步流水线：

| 文件 | 作用 |
|------|------|
| `__init__.py` | 导出 `KernelEvaluator`, `EvalConfig`, `EvalResult` |
| `evaluator.py` | **主评测器**：编译(subprocess) → 正确性(N 次随机输入) → 性能(CUDA event + speedup)。`compute_rewards()` 和 `build_feedback()` 方法 |
| `compiler.py` | Triton 编译检测，subprocess 隔离（crash 不杀 actor），tempfile + importlib 加载（`@triton.jit` 不支持 exec()） |
| `correctness.py` | 数值正确性验证：N 次随机输入 `torch.allclose()` vs PyTorch 参考实现，支持 shape/value/runtime 错误分类 |
| `benchmark.py` | 性能测量：CUDA event 计时，L2 cache 清理，speedup = ref_runtime / kernel_runtime |
| `profiler.py` | SM 级性能采样（可选，需 NVIDIA Nsight Compute）：SM cycles, memory bandwidth, Tensor Core 利用率等硬件指标 |

**评分信号设计**（参考 TritonForge）：
```python
reward_code    = 0.3 * compiled + 0.4 * correct + 0.3 * min(speedup, cap) / cap
reward_design  = 0.3 * min(speedup, cap) / cap      # 仅正确时给分
reward_predict = 0.3 * min(speedup, cap) / cap      # 仅正确时给分
```

### 新增 Trainer 文件

| 文件 | 作用 |
|------|------|
| `verl/trainer/main_ppo_kernel.py` | Kernel PPO 训练入口 |
| `verl/trainer/ppo/ray_trainer_kernel.py` | `RayKernelPPOTrainer` — 使用 kernel dataset + tree manager |
| `verl/trainer/config/ppo_trainer_kernel.yaml` | Kernel 训练配置 |

### 修改的 verl 文件

| 文件 | 修改内容 |
|------|----------|
| `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` | 动态 `n>1` 同步展开：按 runtime `kwargs['n']` 扩 prompt tensors |
| `verl/workers/fsdp_workers.py` | `generate_sequences` 透传 `sampling_kwargs` |
| `verl/workers/megatron_workers.py` | `generate_sequences` 透传 `sampling_kwargs` |
| `verl/workers/actor/dp_actor.py` | Multi-advantage mode：按 section 单独算 PPO loss 后求和 |

### `verl/` — 原始 veRL 训练框架

| 文件 | 作用 |
|------|------|
| `verl/trainer/main_ppo_format.py` | GRPO（链式）训练入口 |
| `verl/trainer/main_ppo_format_ts.py` | Tree-GRPO（树搜索）训练入口 |
| `verl/trainer/ppo/ray_trainer.py` | 链式 rollout 的 Ray PPO 训练器 |
| `verl/trainer/ppo/ray_trainer_ts.py` | 树搜索 rollout 的 Ray PPO 训练器 |
| `verl/trainer/ppo/core_algos.py` | PPO/GRPO 优势估计、KL 控制、损失函数 |
| `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd_ts.py` | 树搜索 vLLM rollout（返回 `infer_log_probs`） |

### `search_r1/` — 原始搜索增强 RL Agent 逻辑

| 文件 | 作用 |
|------|------|
| `search_r1/llm_agent/generation.py` | `LLMGenerationManager` — 多轮链式 rollout（ReAct） |
| `search_r1/llm_agent/generation_ts.py` | `LLMGenerationTreeSearchManager` — 树搜索 rollout（原始 QA） |
| `search_r1/llm_agent/tree_node.py` | `TreeNode` — 树数据结构 |
| `search_r1/llm_agent/tensor_helper.py` | `TensorHelper` — padding/截断/attention mask |

## Kernel RL 核心设计

### 节点状态分离
- `prompt_text`：人类可读 prompt 字符串（日志/复现/调试）
- `prompt_ids`：真正送入模型的 token 张量
- `response_text/response_ids`：本节点本轮新增生成内容，不包含祖先
- `env_feedback_text`：评分进程返回的环境反馈

### 输出格式
模型输出固定三个 section：`<design>...</design>` `<code>...</code>` `<predict>...</predict>`

### 树推理策略
- BFS / level-wise rollout，每层同深度 prompt 组 batch
- vLLM 原生 `n > 1` 多分支（不重复 prompt）
- 生成 child nodes 后可 reward 剪枝

### 训练样本
- `KernelTreeNode`：稀疏 rollout 状态，不存 dense token advantage
- `KernelTrainSample`：dense PPO 实体，每个非根节点导出一个独立 PPO sample
- 端到端训练1步仅用于验证数据链路，后续再开启完整训练

### PPO Sample 组成

每个 `KernelTrainSample`（即 exporter 导出的一个训练样本）包含：

```
┌─ 模型输入 ─────────────────────────────────────────────┐
│  input_ids         = cat([prompt_ids, response_ids])    │  ← 完整 token 序列
│  attention_mask    = cat([prompt_attn, response_attn])  │
│  position_ids      = 从 prompt 末位连续编号               │
│  responses         = response_ids                       │  ← 仅新增生成部分
├─ 训练目标 ─────────────────────────────────────────────┤
│  loss_mask         = OR(design_mask, code_mask, predict_mask) │  ← 哪些 token 参与 loss
│  design_mask       = parser 标记的 design token 位置     │
│  code_mask         = parser 标记的 code token 位置       │
│  predict_mask      = parser 标记的 predict token 位置    │
├─ Token 级评分 ─────────────────────────────────────────┤
│  design_token_scores  = design_mask × scalar_design_reward   │
│  code_token_scores    = code_mask × scalar_code_reward       │
│  predict_token_scores = predict_mask × scalar_predict_reward │
│  token_level_scores   = 三路求和                             │
├─ Advantage (由 advantage.py 计算后加入) ────────────────│
│  design_advantages  / code_advantages  / predict_advantages │
│  advantages (三路合并，兼容旧训练栈)                         │
│  returns (GAE returns，no_estimator 时=scores)               │
├─ 元数据 (non_tensor) ──────────────────────────────────│
│  uid / tree_uid / node_uid / parent_uid / depth         │
└────────────────────────────────────────────────────────┘
```

**构建流程**：
```
scorer返回 scalar_rewards → _attach_score_result → node.scalar_design/code/predict_reward
    → exporter._build_sample()
        → section_mask × scalar_reward → token_level_scores
        → cat(prompt_ids, response_ids) → input_ids
    → fit() 中 compute_log_prob → old_log_probs
    → fit() 中 compute_multi_section_advantages → design/code/predict_advantages
    → fit() 中 wg.update_actor() → 梯度更新
```

**Fallback 机制**：当 parser 解析不出任何 section 时（模型不按格式输出），exporter 将所有有效 response token 视为 code section，避免 `loss_mask` 全零导致 loss=0 训练中断。

### Reward / Advantage
- 节点级 scalar reward：`scalar_design/code/predict_reward`
- Exporter 投影到 token 空间，形成 section-wise token scores
- Actor loss 三路分开算：`L = L_design + L_code + L_predict`

### Scorer
- `KernelScoringWorker` Ray actor，默认 `mode: eval`
- 三步评测流水线：编译(subprocess隔离) → 正确性(N次随机输入) → 性能(CUDA event + speedup)
- `log_only` 模式保留用于调试（不执行真实 benchmark）
- SM 级性能采样通过 `scoring/profiler.py` 可选扩展（需 ncu）

## 模型

Qwen2.5-Coder-3B-Instruct，路径：`/inspire/qb-ilm/project/wuliqifa/public/sdt/models/Qwen2.5-Coder-3B-Instruct`

## 数据集

种子任务（任务描述+参考实现，不包含训练信号），路径：`data/kernel_rl/train.parquet`

| 类别 | 任务 | 难度 |
|------|------|------|
| elementwise | `vector_add`, `gelu_activation` | easy |
| reduction | `softmax` | medium |
| normalization | `layer_norm` | medium |
| gemm | `matmul_naive` | hard |
| attention | `attention_score` | hard |

## 测试与端到端验证

### 冒烟脚本（CPU 可跑，已全部通过）
```bash
python scripts/kernel_rl/smoke_parser_exporter.py   # parser + exporter + prompt builder（二叉树）
python scripts/kernel_rl/smoke_advantage.py          # section-wise advantage（4/4 PASS）
python scripts/kernel_rl/smoke_scorer.py             # scorer actor（PID + JSONL）
python scripts/kernel_rl/smoke_scoring.py            # 评分模块：编译+正确性+性能（6/6 PASS，需 GPU）
```

### SFT 数据生成
```bash
# 从 KernelBook 生成 800 条三段式 SFT 数据（含 auto-generated design/predict）
python scripts/kernel_rl/gen_sft_data_v2.py --num_samples 800 --output data/kernel_rl/sft_train_v2.jsonl
```

### SFT 训练（需要 4×GPU，全量微调）
```bash
bash scripts/kernel_rl/run_sft.sh
```
预期：800 样例，5 epochs，~500 步，~40 分钟。输出到 `models/Qwen2.5-Coder-7B-Instruct-SFT-kernel-v2/`

### 端到端验证（需要 GPU，二叉树 depth=2）
```bash
bash scripts/kernel_rl/run_e2e.sh
```
预期：root → 2 children → 4 leaves，6 个 PPO 样本，1 步训练。日志输出到 `scripts/kernel_rl/e2e_run_*.log`。

## 当前分支与提交历史

```
7f7ec88 update smoke_parser_exporter (二叉树 + prompt累积 + 日志输出)
8f87971 Add kernel PPO trainer entrypoint and config (+ smoke_scorer)
d4a3d6a Enable native vLLM branching and multi-advantage actor loss (+ smoke_advantage)
4e177dc Add kernel tree data model and smoke scripts
```

## 当前状态

### 已完成
- ✅ SFT 训练（800 KernelBook 样例, 5 epochs, avg loss 0.09, ~75min, 4×GPU FSDP）
- ✅ SFT 模型格式验证：未见任务上 100% 输出三段式 + 有效 Triton 代码
- ✅ RL 端到端链路确认打通（vLLM → tree rollout → scorer → advantage → PPO update）
- ✅ 评分模块：编译 + 正确性 + 性能，subprocess 隔离，reward 三段可配置
- ✅ vLLM KV cache 内存分析文档：`docs/vllm_kvcache_memory.md`
- ✅ 输出目录重定向到 `/tmp`（GPFS 配额仅 368MB，全组共享）
- 模型：`models/Qwen2.5-Coder-7B-Instruct-SFT-kernel-v2/checkpoint-500/`

### 已修复的关键阻塞问题
| 问题 | 根因 | 修复 |
|------|------|------|
| E2E 卡在 Ray init | GPFS overlay quota 写满 | Hydra/checkpoint/log 重定向到 `/tmp` |
| GPU 调度死锁 | scorer `num_gpus=0.2` 碎片化 GPU0，veRL 4×STRICT_PACK 永远等不到 | scorer `num_gpus=0.0` |
| response_length=1 | SFT 用 chat template 训练，但 tree_manager 发裸 prompt 给 vLLM | `_build_prompt_batch` 包裹 `apply_chat_template(add_generation_prompt=True)` |
| vLLM KV cache OOM | 7B 模型需更多显存 | `gpu_memory_utilization=0.6`, `ppo_max_token_len=8192` |

### 当前 RL 链路状态（已验证）
```
SFT Model → chat template prompt → vLLM 生成 ~962 tokens → parser 三段
→ scorer 编译+正确性+性能 → reward (max=194) → advantage (max=0.3)
→ pg_loss_code=-0.3 → FSDP gradient → grad_norm=0.133
```
- `response_length/mean`: 961.6（从 1.0 修复后）
- `critic/rewards/mean`: 78.0, max: 194.1
- `actor/pg_loss_code`: -0.30（非零训练信号）
- 三段 loss 均计算：`pg_loss_design`, `pg_loss_code`, `pg_loss_predict`

### 已知限制
- Design 和 predict 的 reward 目前与 code 同向（`scoring/reward.py`），需独立设计
- 种子任务仅 6 个（`train.parquet`），需扩展到 KernelBench Level 1 规模
- GRPO advantage estimator 未启用（当前 `no_estimator` = 原始 scores）
- 训练仅 1 步验证，未做多步 RL
- SFT 模型输出含 `torch._inductor` 模板代码（KernelBook 数据特征），需 RL 优化去掉

## 下一步：Phase 2 真实 RL 训练

### 目标
写出至少比 `torch.compile` 性能更高的 Triton 算子，在 KernelBench Level 1 上验证。

### 需要的工作

**1. 种子任务扩展**
- 从 KernelBench Level 1（100 问题）精选 20-30 个适合 Triton 的问题
- 每个任务格式：`task_spec(reference_python, ...)` + `bench_spec(input_gen, target_speedup, ...)`
- 确保 `input_gen` 参数签名与 `reference_python` 一致

**2. Reward 设计——代际差异 + 同深度节点对比**
```
当前 reward = 0.3*compile + 0.4*correct + 0.3*perf（三段同向）
目标 reward:
  - code:    compile + correctness + speedup（已有，保留）
  - design:  [父节点 design 策略] 与 [子节点 code 实现] 的一致性
             + design 中声称的 block_size 是否与 code 中实际的一致
             + 同级节点间 design 多样性（鼓励探索）
  - predict: |predicted_speedup - actual_speedup| / max(predicted, actual)
             越接近实测值分越高
```

**3. 代际差异 reward（inter-generational）**
- 子节点 speedup > 父节点 speedup → 正奖励（改进了）
- 子节点 speedup < 父节点 speedup → 负奖励（退步了）
- 基于 tree 结构自然获得：child.speedup - parent.speedup

**4. GRPO 同深度对比**
- 启用 `adv_estimator: grpo`（替代 `no_estimator`）
- 同一深度节点的 reward 相互对比，计算 group-wise advantage
- 更平稳的训练：advantage 不再只看绝对值，而是看组内相对表现

**5. 训练配置**
- Multi-step RL（≥100 步），逐步提升
- `adv_estimator: grpo`
- 每步对比训练前后的 KernelBench 指标（compile rate, correctness rate, avg speedup）
- 日志保留关键 metrics，不自动清理

### 关键文件
| 文件 | 需修改内容 |
|------|------|
| `data/kernel_rl/train.parquet` | 替换为 20-30 个 KernelBench 问题 |
| `scoring/reward.py` | 新增 design-prediction 一致性 reward、代际差异 reward |
| `verl/trainer/config/ppo_trainer_kernel.yaml` | `adv_estimator: grpo`、多步训练 |
| `search_r1/kernel_rl/advantage.py` | 确认 GRPO 同深度节点组正确 |
| `docs/vllm_kvcache_memory.md` | vLLM KV cache 分析文档 |

## 存储说明
- 代码、数据、模型权重：GPFS `/inspire/qb-ilm/project/wuliqifa/public/sdt/`（长期保留）
- 训练输出（checkpoint、日志）：`/tmp/`（overlay，1.5TB，运行完需分析后清理）
- GPFS 配额仅 368MB 剩余（全组 40+ 人共享），禁止写入大文件

## Git 作者
- sudetong <sudetong@local>

## 本地开发规则

- **永远不推送到远程仓库**，只做本地 `git commit`
- 提交由用户手动完成，Claude 仅做本地 commit（不 push）
- 危险操作（`rm -rf`、`git push`、`git reset --hard`）已被 `.claude/settings.json` 拦截
- 常规开发操作（python、git 本地、文件读写）无需重复确认，已加入 allowlist
