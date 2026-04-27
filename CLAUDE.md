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
│  ③ Scorer 评分进程 (CPU, Ray actor)                         │
│  ├─ 接收 KernelScoreRequest (node_uid, code_text, ...)     │
│  ├─ Phase1: log_only, 写入 JSONL 日志                       │
│  ├─ Phase2+: 子进程执行 benchmark/compile/runtime           │
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
| `scorer.py` | `KernelScoringWorker` Ray actor + `KernelScoringPool`，phase1 仅 log-only |
| `exporter.py` | 树节点 → PPO batch，每个非根节点导出一个独立样本 |
| `advantage.py` | Section-wise advantage：`design/code/predict` 分别算 advantage |
| `dataset.py` | Kernel dataset，支持 `task_spec/bench_spec/reference_python`，自动解析 JSON 字段 |
| `tree_manager.py` | Level-wise tree rollout manager，vLLM 原生 `n>1`，BFS 逐层推理 |

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
- `KernelTrainSample`：dense PPO 实体，含 `loss_mask`、`design/code/predict_mask`、section token scores
- 每个非根节点导出一个独立 PPO sample

### Reward / Advantage
- 节点级 scalar reward：`scalar_design/code/predict_reward`
- Exporter 投影到 token 空间，形成 section-wise token scores
- Actor loss 三路分开算：`L = L_design + L_code + L_predict`

### Scorer
- `KernelScoringWorker` Ray actor，phase1 仅 `log_only`
- 预埋 subprocess helper，后续可执行 benchmark/compile/runtime

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
```

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

## 当前状态与已知限制

- 三个冒烟测试全部通过，数据结构通路验证完毕
- 端到端脚本已就绪，待 GPU 执行验证
- scorer 仅 log_only，未执行真实 benchmark
- validation 为 placeholder
- `algorithm.kernel_adv_estimator` 为 `no_estimator`，后续可切 `grpo`
- Git author: Sdt <sdt@local>
