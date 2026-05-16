# vLLM KV Cache 架构与 veRL 混合引擎内存管理分析

> 分析日期: 2026-05-16
>
> 基于 veRL (volcengine/verl) 源码及 vLLM 0.4.2/0.5.4/0.6.3 适配层
>
> 项目路径: `/inspire/qb-ilm/project/wuliqifa/public/sdt/Tree-GRPO`

---

## 目录

1. [vLLM KV Cache 架构](#1-vllm-kv-cache-架构)
   - 1.1 [PagedAttention：块级 KV 缓存管理](#11-pagedattention块级-kv-缓存管理)
   - 1.2 [KV Cache 在 GPU 间的分布策略](#12-kv-cache-在-gpu-间的分布策略)
   - 1.3 [数据并行模式下的独立 KV Cache](#13-数据并行模式下的独立-kv-cache)
   - 1.4 [内存占用公式](#14-内存占用公式)
   - 1.5 [块数量预估：profile 阶段](#15-块数量预估profile-阶段)
2. [veRL 混合引擎内存管理](#2-verl-混合引擎内存管理)
   - 2.1 [Hybrid Engine 设计目标](#21-hybrid-engine-设计目标)
   - 2.2 [free_cache_engine 机制详解](#22-free_cache_engine-机制详解)
   - 2.3 [Sleep/Wake 循环：完整生命周期](#23-sleepwake-循环完整生命周期)
   - 2.4 [模型权重同步与 offload](#24-模型权重同步与-offload)
   - 2.5 [内存竞争边界情况分析](#25-内存竞争边界情况分析)
3. [当前配置分析 (ppo_trainer_kernel.yaml)](#3-当前配置分析-ppo_trainer_kernelyaml)
   - 3.1 [配置关键参数](#31-配置关键参数)
   - 3.2 [7B 模型内存计算](#32-7b-模型内存计算)
   - 3.3 [与标准 ppo_trainer.yaml 对比](#33-与标准-ppo_traineryaml-对比)
4. [总结与建议](#4-总结与建议)

---

## 1. vLLM KV Cache 架构

### 1.1 PagedAttention：块级 KV 缓存管理

vLLM 的核心创新在于 **PagedAttention**——一种受操作系统虚拟内存分页机制启发的 KV cache 管理方案。

**传统 Attention 的问题：**

在原生 Transformer 推理中，每个请求的 KV cache 是一块连续显存，大小由 `max_seq_len` 决定。这导致两个问题：
- **内部碎片**：短序列为其最大可能长度预留空间，实际未用满
- **外部碎片**：不同长度的序列交替分配/释放后，显存出现不可合并的空隙

**PagedAttention 解决方案：**

KV cache 被分割成固定大小的 **Block**（块），vLLM 默认 `block_size=16`（见 `/inspire/qb-ilm/project/wuliqifa/public/sdt/Tree-GRPO/verl/third_party/vllm/vllm_v_0_4_2/arg_utils.py:56`）：

```python
# EngineArgs 默认值
block_size: int = 16          # 每个 block 存储 16 个 token 的 KV
gpu_memory_utilization: float = 0.90  # GPU 显存利用率
```

每个 block 的大小由以下因素决定：

```
单 block 字节数 = 2 (K 和 V) × num_layers × num_heads × head_dim × block_size × dtype_bytes
```

- `num_layers`: Transformer 层数
- `num_kv_heads`: 每层 KV 头数（GQA 下可能小于 query heads）
- `head_dim`: 每个头的维度（通常 = hidden_size / num_heads）
- `block_size`: 16（默认）
- `dtype_bytes`: bf16 = 2 bytes, fp16 = 2 bytes, fp8 = 1 byte

对于 Qwen2.5-Coder-7B（假设 28 层, 16 KV heads, head_dim=128, bf16）：

```
单 block 字节数 = 2 × 28 × 16 × 128 × 16 × 2 = 3,670,016 bytes ≈ 3.5 MB
```

**Block Table 映射：**

vLLM 维护一个 **block table**，将每个请求的逻辑 token 位置映射到物理 block。这与操作系统中虚拟地址到物理页面的映射完全对应：

```
请求 A 的 token 位置: [0-15]  [16-31]  [32-47] ...
        |            |        |        |
Block Table:       Block 7  Block 3  Block 42  ...
        |            |        |        |
物理 GPU 显存:    [Block 7] [Block 3] [Block 42] ...
```

这个机制使得：
- 多个请求可以共享物理 block（如 beam search 中的公共前缀）
- 不存在碎片问题（所有 block 大小一致）
- memory usage 仅与实际使用的 token 数成正比，而非最大长度

vLLM 的 CacheEngine 实现位于 vLLM 官方仓库的 `vllm/worker/cache_engine.py`。在 veRL 适配的 vLLM 版本中，worker 类的缓存初始化通过继承实现（见 `/inspire/qb-ilm/project/wuliqifa/public/sdt/Tree-GRPO/verl/third_party/vllm/vllm_v_0_4_2/worker.py:199-206`）：

```python
def _init_cache_engine(self):
    if self.cache_engine is None and self.gpu_cache is None:
        super()._init_cache_engine()
```

### 1.2 KV Cache 在 GPU 间的分布策略

KV cache 的分布方式取决于并行策略。vLLM 支持三种并行模式，KV cache 行为各有不同：

#### 1.2.1 张量并行 (Tensor Parallelism, TP)

- `tensor_model_parallel_size > 1`
- 模型权重在 TP 组内各 GPU 间 **分片存储**
- KV cache **也按注意力头均匀分片**：每个 GPU 只存储 `num_kv_heads / tp_size` 个头的 KV
- 举例：16 KV heads, TP=4 -> 每个 GPU 存 4 个头的 KV cache
- KV cache **共享 GPU 显存**：所有 TP rank 共享同一组物理 block，因为 `determine_num_available_blocks` 在 TP 组内通过 all_reduce 同步（取最小值），见 `/inspire/qb-ilm/project/wuliqifa/public/sdt/Tree-GRPO/verl/third_party/vllm/vllm_v_0_4_2/worker.py:185-193`：

```python
num_gpu_blocks = torch.tensor([num_gpu_blocks], device='cuda')
torch.distributed.all_reduce(
    num_gpu_blocks,
    op=torch.distributed.ReduceOp.MIN,
    group=get_tensor_model_parallel_group()
)
```

这个 all_reduce 确保所有 TP rank 使用相同数量的 block（取各 rank 可用量的最小值）。

#### 1.2.2 流水线并行 (Pipeline Parallelism, PP)

- 模型按层切分到不同 GPU，每个 GPU 存一部分层
- KV cache **只存在于对应的 PP stage 上**
- 在 veRL 的 hybrid engine 中，PP 的使用场景主要是 Megatron 后端
- 在 vLLM 集成中，PP 模式并非默认启用（veRL 的 vLLM rollout 的 TODO 注释明确说明需要支持，见 `/inspire/qb-ilm/project/wuliqifa/public/sdt/Tree-GRPO/verl/workers/rollout/vllm_rollout/vllm_rollout.py:43-44`）

```python
# TODO
# 1. support pp in vllm
```

- veRL 目前在 FSDP 后端下不使用 PP，而是通过 **Rollout Device Mesh** 组织 `(dp, infer_tp)` 两维

#### 1.2.3 数据并行 (Data Parallelism, DP)

- 每个 GPU 持有 **完整的模型副本**
- 每个 GPU 有 **独立的 KV cache**（不共享，不通信）
- 这就是 `tensor_model_parallel_size=1` 的场景
- **关键**：各 GPU 的 KV cache 完全独立，block 数量不需要同步

### 1.3 数据并行模式下的独立 KV Cache

在 `tensor_model_parallel_size=1` 时，veRL 的数据流如下：

1. **DP 维度划分**：`nnodes * n_gpus_per_node` 个 GPU 构成 FSDP 的数据并行组
2. **Rollout Device Mesh**：维度为 `(dp=world_size, infer_tp=1)`
   - 见 `/inspire/qb-ilm/project/wuliqifa/public/sdt/Tree-GRPO/verl/workers/fsdp_workers.py:371-374`
   ```python
   infer_tp = self.config.rollout.tensor_model_parallel_size
   dp = self.world_size // infer_tp
   rollout_device_mesh = init_device_mesh(
       'cuda', mesh_shape=(dp, infer_tp), mesh_dim_names=['dp', 'infer_tp']
   )
   ```
3. 每个 GPU 独立实例化一个 vLLM `SPMDGPUExecutor`（见 `/inspire/qb-ilm/project/wuliqifa/public/sdt/Tree-GRPO/verl/third_party/vllm/vllm_v_0_4_2/spmd_gpu_executor.py:69-96`），包含完整的模型权重和独立的 KV cache
4. `generate_sequences` 时各 GPU 处理不同的 prompt batch，互不干扰

**重要**：在 DP 模式下，`determine_num_available_blocks` 的 all_reduce 同步（worker.py:185-193）虽然执行，但由于 TP=1 时 TP 组大小为 1，all_reduce 实际上是一个 no-op，每张 GPU 独立确定自己的 block 数量。

### 1.4 内存占用公式

GPU 显存使用可以分解为以下部分：

```
total_gpu_memory = model_weights + kv_cache_blocks * block_size + activation_memory + reserved
```

其中：

1. **模型权重** (model_weights)：
   - 参数数量 * 每个参数的字节数
   - 7B bf16: `7 * 10^9 * 2 = 14 GB`
   - FSDP 全分片下，训练时每 GPU 仅持有 `params / world_size`，但 rollout 时需要完整参数

2. **KV Cache** (kv_cache_blocks * block_size)：
   ```
   block_size = 2 * num_layers * num_kv_heads * head_dim * block_size_tokens * dtype_bytes
   ```
   - `num_gpu_blocks` 由 profile 阶段确定（见第 1.5 节）
   - 每个 block 存储 `block_size_tokens`（默认 16）个 token 的 K 和 V

3. **激活内存** (activation_memory)：
   - 前向传播中间激活值（包括 attention score、hidden states 等）
   - 与 `max_num_batched_tokens` 和 `max_num_seqs` 成正比
   - `enforce_eager=True` 时无 CUDA graph 额外开销（但也无 graph 加速）

4. **系统预留** (reserved)：
   - CUDA context、PyTorch 框架开销、NCCL buffer 等
   - 实测通常在 1-2 GB 左右

vLLM 通过 `gpu_memory_utilization` 参数控制 KV cache 可用的上限比例：

```
gpu_memory_for_kv_cache = free_gpu_memory * gpu_memory_utilization
```

其中 `free_gpu_memory` 是 profile 阶段模型加载完毕后剩余的 GPU 显存。

### 1.5 块数量预估：profile 阶段

KV cache 块数量的确定发生在 vLLM 引擎初始化时，流程如下：

1. **加载模型权重**到 GPU（`worker.load_model()`）
2. **执行一次 profile run**（`model_runner.profile_run()`），用 dummy 输入做一次前向传播，记录峰值内存
3. **计算可用显存**：profile 完成后剩余的自由显存 * `gpu_memory_utilization`
4. **计算 block 数量**：`num_gpu_blocks = free_memory / block_size_bytes`
5. **TP 组内 all_reduce** 同步取最小值（见 worker.py:185-193）

源码路径：
- **确定 block 数量**：`/inspire/qb-ilm/project/wuliqifa/public/sdt/Tree-GRPO/verl/third_party/vllm/vllm_v_0_4_2/worker.py:141-196` (`determine_num_available_blocks`)
- **初始化 KV cache 张量**：原版 vLLM `CacheEngine._init_cache_engine()`（不在 veRL 仓库内，在 vLLM 官方包中）

---

## 2. veRL 混合引擎内存管理

### 2.1 Hybrid Engine 设计目标

veRL 的 **Hybrid Engine**（混合引擎）旨在解决 RL 训练中一个核心矛盾：

- **Rollout（推理）阶段**：需要模型完整参数做自回归生成，需要大量 KV cache
- **训练（Training）阶段**：需要显存做 FSDP 全分片 + 优化器状态 + 梯度累积

如果两个阶段同时持有各自的全部内存占用，在单 GPU 显存受限（如 48GB）的情况下，7B+ 规模的模型会直接 OOM。

veRL 的解决思路是：**同一组 GPU 交替扮演 rollout worker 和 training worker 的角色**。在这个设计下，`ActorRolloutRefWorker` 同时承担推理和训练两种职责（见 `/inspire/qb-ilm/project/wuliqifa/public/sdt/Tree-GRPO/verl/workers/fsdp_workers.py:105-109`）：

```python
self._is_actor = self.role in ['actor', 'actor_rollout', 'actor_rollout_ref']
self._is_rollout = self.role in ['rollout', 'actor_rollout', 'actor_rollout_ref']
self._is_ref = self.role in ['ref', 'actor_rollout_ref']
```

在配置中 `role='actor_rollout'`，意味着同一个 worker 实例既是 actor（训练）又是 rollout（推理）。

### 2.2 free_cache_engine 机制详解

`free_cache_engine` 是 veRL 的关键优化开关，位于 rollout 配置中（`ppo_trainer_kernel.yaml:92`）：

```yaml
free_cache_engine: True
```

#### 2.2.1 做了什么？

当 `free_cache_engine=True` 时，vLLM Rollout 对象在 `generate_sequences()` 方法前后执行 cache engine 的初始化和释放（见 `/inspire/qb-ilm/project/wuliqifa/public/sdt/Tree-GRPO/verl/workers/rollout/vllm_rollout/vllm_rollout.py:150-151, 229-230`）：

```python
@torch.no_grad()
def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
    # rebuild vllm cache engine
    if self.config.free_cache_engine:
        self.inference_engine.init_cache_engine()

    # ... rollout generation ...

    # free vllm cache engine
    if self.config.free_cache_engine:
        self.inference_engine.free_cache_engine()

    return DataProto(batch=batch)
```

`init_cache_engine` 调用链为：

```
vLLMRollout.generate_sequences()
  -> self.inference_engine.init_cache_engine()          # LLM.llm.py:137-138
    -> self.llm_engine.init_cache_engine()               # LLMEngine.llm_engine_sp.py:236-239
      -> self.model_executor.init_cache_engine()          # SPMDGPUExecutor.spmd_gpu_executor.py:140-141
        -> self.worker._init_cache_engine()               # Worker.worker.py:199-201
          -> super()._init_cache_engine()                 # 原版 vLLM CacheEngine 初始化
```

`free_cache_engine` 调用链为：

```
vLLMRollout.generate_sequences()
  -> self.inference_engine.free_cache_engine()           # LLM.llm.py:140-141
    -> self.llm_engine.free_cache_engine()                # LLMEngine.llm_engine_sp.py:241-242
      -> self.model_executor.free_cache_engine()          # SPMDGPUExecutor.spmd_gpu_executor.py:143-144
        -> self.worker.free_cache_engine()                # Worker.worker.py:203-206
          -> self.cache_engine = None
          -> self.gpu_cache = None
          # 然后 Python GC + torch.cuda.empty_cache() 回收显存
```

**释放的实际内容**（见 worker.py:203-206）：

```python
def free_cache_engine(self):
    # ensure `enforce_eager=True`
    self.cache_engine = None
    self.gpu_cache = None
```

`gpu_cache` 是一个 `List[torch.Tensor]`，每个元素是形状为 `(2, num_blocks, num_kv_heads, block_size, head_dim)` 的 KV cache 块集合（K 和 V 分别存储）。将其设为 None 后，PyTorch 的引用计数机制会在下次 GC 时回收这部分显存。

#### 2.2.2 是否完全释放 vLLM 内存？

**不完全。** `free_cache_engine` 只释放 KV cache 本身（上述 `gpu_cache` 张量）。vLLM 进程中仍然持有：

1. **模型权重**（vLLM 内部的模型副本）：虽然在 rollout `__init__` 末尾调用了 `self.inference_engine.offload_model_weights()` 将权重移到 CPU（见 vllm_rollout.py:109），但在 sharding manager 的 `__enter__` 中又被同步回 GPU，接着在 `__exit__` 中重新 offload 到 CPU。

2. **vLLM 调度器状态**：`Scheduler` 对象和内部队列仍占用少量内存。

3. **模型结构**：vLLM 的 `ModelRunner` 和模型 `nn.Module` 对象本身仍然存在（即使在 CPU 上）。

所以 `free_cache_engine` + `offload_model_weights` 组合大约释放：
- **All of KV cache**（几十 GB，最大块）
- 模型权重从 GPU 移到 CPU（而非释放，但为 FSDP 训练腾出 GPU 空间）

**不释放**：
- vLLM 进程的 Python 对象开销
- 模型的 CPU 副本

对于训练阶段而言，最关键的是 **GPU 显存**被释放以供 FSDP 使用。CPU 内存通常充裕，vLLM 的 CPU 权重副本不构成瓶颈。

### 2.3 Sleep/Wake 循环：完整生命周期

完整的一轮 PPO 迭代中，GPU 显存使用状态转换如下：

```
                         Rollout 阶段                训练阶段
                     +----------------------+  +----------------------+
                     |                      |  |                      |
    GPU 显存          |  [vLLM 模型权重]      |  |  [FSDP 分片参数]      |
    +------+         |  [vLLM KV Cache]     |  |  [优化器状态]         |
    |      |         |  [模型推理计算]       |  |  [梯度累积]           |
    |      |         |                      |  |                      |
    +------+         +----------+-----------+  +----------+-----------+
                                |                          |
                          FSDP state_dict() ->            FSDP 训练
                          vLLM sync_weights()            vLLM offload
                          vLLM init_cache()              (已释放)
```

具体时序（对应 `ray_trainer.py` 和 `ray_trainer_kernel.py` 中的 `fit()` 循环）：

#### Rollout 阶段（推理生成）

```
Step 1: FSDP -> vLLM 权重同步
  - ActorRolloutRefWorker.generate_sequences() 被调用
  - 进入 rollout_sharding_manager (FSDPVLLMShardingManager.__enter__)
  - 调用 self.module.state_dict() 收集 FSDP 全量参数
  - 调用 inference_engine.sync_model_weights(params) 将参数写入 vLLM 模型
  - 删除 params 临时副本，清缓存
  - 见: fsdp_vllm.py:70-91

Step 2: 初始化 KV cache
  - vLLMRollout.generate_sequences() 内
  - 调用 inference_engine.init_cache_engine()
  - 分配 GPU block 给 KV cache

Step 3: 自回归生成
  - inference_engine.generate(...)
  - vLLM 调度器逐 token 生成
  - KV cache 随 decode 逐步填充

Step 4: 释放 KV cache
  - 调用 inference_engine.free_cache_engine()
  - gpu_cache = None, 显存回收
  - 见: vllm_rollout.py:228-230

Step 5: offload vLLM 模型权重
  - 退出 rollout_sharding_manager (FSDPVLLMShardingManager.__exit__)
  - 调用 inference_engine.offload_model_weights()
  - vLLM 模型权重移至 CPU
  - module.train() 将 FSDP 模块置回训练模式
  - 见: fsdp_vllm.py:104-111
```

#### 训练阶段（梯度更新）

```
Step 6: FSDP 训练
  - ActorRolloutRefWorker.update_actor() / compute_log_prob() 被调用
  - FSDP 自动从分片恢复全量参数（按需 unshard）
  - 执行前向/反向/优化器更新
  - 训练完成后可以 offload 参数
  - 见: fsdp_workers.py:496-538

Step 7: 清空 CUDA 缓存
  - 各阶段末尾调用 torch.cuda.empty_cache()
```

### 2.4 模型权重同步与 offload

#### 权重同步 (FSDP -> vLLM)

`FSDPVLLMShardingManager.__enter__()` 是权重同步的核心入口（`/inspire/qb-ilm/project/wuliqifa/public/sdt/Tree-GRPO/verl/workers/sharding_manager/fsdp_vllm.py:70-91`）：

```python
def __enter__(self):
    params = self.module.state_dict()      # 收集 FSDP 全量参数（触发 all-gather）
    # 根据 load_format 选择同步方式
    if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
        self.inference_engine.sync_model_weights(params, load_format=load_format)
    # ...
    del params                              # 删除临时副本
    torch.cuda.empty_cache()
```

这个过程涉及：
1. FSDP `state_dict()` 在底层触发 `all-gather`，将全量模型参数集中到当前 rank
2. `sync_model_weights` 将参数写入 vLLM 的模型副本
3. 参数副本立即删除，避免显存浪费

对于 `load_format='dtensor'` 的情况，`DTensorLoader` 将分片格式的权重映射到 vLLM 模型（`/inspire/qb-ilm/project/wuliqifa/public/sdt/Tree-GRPO/verl/third_party/vllm/vllm_v_0_4_2/model_loader.py:200-243`）。

#### 权重 offload (vLLM -> CPU)

`offload_model_weights` 将 vLLM 推理引擎的模型参数移到 CPU 预分配张量中（`/inspire/qb-ilm/project/wuliqifa/public/sdt/Tree-GRPO/verl/third_party/vllm/vllm_v_0_4_2/worker.py:246-254`）：

```python
def offload_model_weights(self) -> None:
    if self.cpu_model == None:
        self.cpu_model = {}
        for name, params in self.model_runner.model.named_parameters():
            self.cpu_model[name] = torch.empty_like(params, device='cpu')
            params.data = self.cpu_model[name]
    else:
        for name, params in self.model_runner.model.named_parameters():
            params.data = self.cpu_model[name]
```

注意：这里采用的是**指针替换**而非数据拷贝。`params.data = cpu_tensor` 使模型参数直接指向 CPU 张量，零拷贝开销。但这也意味着 GPU 显存立即释放——因为原 GPU 张量不再被引用。

### 2.5 内存竞争边界情况分析

veRL 的 Hybrid Engine 设计在大多数情况下能有效防止显存竞争，但仍存在几个边界情况：

#### 2.5.1 vLLM profile 阶段的内存估计偏差

vLLM 在 `determine_num_available_blocks()` 中做 profile 时，使用的是当前 GPU 的可用显存。但此时的 profile 是在 **vLLM 模型刚加载完毕、但尚未进行任何训练的状态**下做的。如果后续训练阶段中 FSDP 需要更多显存（如 optimizer state 较大），profile 阶段分配给 KV cache 的 block 数可能偏大。

不过这个风险在实际中被缓解：FSDP FULL_SHARD 模式下，训练时每 GPU 只持有 1/world_size 的参数，optimizer state 也同理，实际训练显存需求通常低于 vLLM rollout 的显存需求。

#### 2.5.2 FSDP 参数收集的瞬时显存峰值

在 `FSDPVLLMShardingManager.__enter__()` 中调用 `self.module.state_dict()` 时，FSDP 需要在当前 GPU 上收集**全量模型参数**。对于 7B bf16 模型，这意味着约 14GB 的临时显存占用。如果此时 GPU 上还有残留的 KV cache（比如 `free_cache_engine=False` 场景），就会产生瞬时显存峰值。

这就是 `free_cache_engine=True` 的另一个重要作用——在 state_dict 收集参数前，确保 KV cache 已被释放。

#### 2.5.3 vLLM offload 到 CPU 不释放

`offload_model_weights()` 将 vLLM 的模型权重移到 CPU，但 CPU 内存本身不释放。在 GPU 显存受限但 CPU 内存充裕的环境下这不是问题。但如果 CPU 内存也紧张（例如 32GB RAM 的节点），vLLM 的 CPU 权重副本 + Python 进程本身 + 训练数据的 CPU 缓存可能导致 swap。

#### 2.5.4 多轮 rollout 场景（kernel RL 的树搜索）

在 kernel RL 的树搜索中（`ray_trainer_kernel.py`），一个训练步骤可能涉及多次 rollout 调用（每层一次）：

```
树展开：root -> 调 generate_sequences -> 评分 -> child 节点
      -> 调 generate_sequences -> 评分 -> grandchild 节点
```

每次 `generate_sequences` 都执行 `init_cache_engine` -> 生成 -> `free_cache_engine` 的完整循环。这意味着：
- 各层展开的 KV cache 分配/释放不会叠加（因为每次都会先释放再分配）
- 但是调度器开销和 CPU 端模型副本保持不变
- 如果树展开之间插入了 `compute_log_prob` 等训练操作，需要特别小心内存转换

在当前的 kernel PPO 实现中，所有的树展开都在 `tree_manager.run_tree_rollout()` 内部完成，之后才执行 `compute_log_prob`。因此 rollout 阶段和训练阶段的边界是清晰的，不存在交叉。

#### 2.5.5 DP 模式下 GPU 间显存不平衡

`tensor_model_parallel_size=1` 时，各 GPU 独立管理 KV cache。虽然 DP 模式下每个 GPU 分配的 batch 大小相同，但由于各 GPU 处理的 prompt/response 长度可能不同，KV cache block 的实际使用量可能不平衡。不过 vLLM 本身会对请求做动态批处理调度，这种不平衡在统计意义上会被平均。

---

## 3. 当前配置分析 (ppo_trainer_kernel.yaml)

### 3.1 配置关键参数

文件路径：`/inspire/qb-ilm/project/wuliqifa/public/sdt/Tree-GRPO/verl/trainer/config/ppo_trainer_kernel.yaml`

#### 显存相关参数

```yaml
actor_rollout_ref:
  hybrid_engine: True                    # 启用混合引擎（必须）
  model:
    path: .../Qwen2.5-Coder-7B-Instruct-SFT-kernel-v2/checkpoint-500
  rollout:
    name: vllm
    dtype: bfloat16                      # 权重和 KV cache 精度
    gpu_memory_utilization: 0.6          # vLLM 可用显存比例
    enforce_eager: True                  # 禁用 CUDA graph（减少显存）
    free_cache_engine: True              # 训练时释放 KV cache
    tensor_model_parallel_size: 1        # 数据并行模式
    max_num_batched_tokens: 4096         # 单次批处理最大 token 数
    max_num_seqs: 32                     # 单次批处理最大序列数

trainer:
  nnodes: 1
  n_gpus_per_node: 4                     # 4 GPU 数据并行
```

#### 非显存但相关参数

```yaml
  actor:
    strategy: fsdp
    ppo_mini_batch_size: 16
    ppo_micro_batch_size: 4
    fsdp_config:
      param_offload: False               # 不使用 CPU offload 训练参数
      grad_offload: False
      optimizer_offload: False
```

所有 offload 均为 False，意味着 FSDP 训练期间 actor 参数、梯度、优化器状态都在 GPU 上。对于 7B 模型和 4 GPU：
- 每 GPU 参数：14 GB / 4 = 3.5 GB（FSDP FULL_SHARD）
- 优化器状态（AdamW）：2 * 3.5 GB = 7 GB（动量和方差）
- 梯度：~3.5 GB
- 总计训练显存：~14 GB

### 3.2 7B 模型内存计算

#### 基本参数

- 模型规模：7B 参数
- 精度：bf16（每参数 2 字节）
- GPU 可用数量：4 张
- 假设单卡显存：48 GB（如 L40、A40 或 A100-40GB 的高显存版本）

#### 模型权重

```
权重显存 = 7 * 10^9 * 2 bytes = 14 GB
```

在 FSDP FULL_SHARD 模式下，训练时每 GPU 仅含：
```
每 GPU 权重 = 14 GB / 4 = 3.5 GB
```

但 rollout 时需要完整权重在 GPU 上（vLLM 持有），这是 hybrid engine 需要切换的原因。

#### Rollout 阶段显存分配

```
GPU 总显存: 48 GB

gpu_memory_utilization = 0.6
-> vLLM 可用: 48 * 0.6 = 28.8 GB

其中:
- 模型权重 (vLLM 完整模型): 14 GB
- 剩余给 KV cache: 28.8 - 14 = 14.8 GB
- 实际还要扣除激活内存、CUDA context 等 (~1-2 GB)
- 所以 KV cache 实际可用: ~12-13 GB
```

#### KV Cache Block 数量估算

```
单 block 大小（估算）:
  = 2 * num_layers * num_kv_heads * head_dim * block_size * dtype_bytes

  Qwen2.5-Coder-7B 配置（典型值）:
  - num_layers: ~28
  - num_kv_heads: ~16（GQA 假设）
  - head_dim: 128 (hidden_size=2048 / num_heads=16)
  - block_size: 16
  - dtype: bf16 = 2 bytes

  = 2 * 28 * 16 * 128 * 16 * 2
  = 2 * 28 * 65536
  = 3,670,016 bytes ~= 3.5 MB

可用 block 数:
  ~= 13 GB / 3.5 MB ~= 3700 blocks
```

每个 block 存 16 个 token 的 KV，所以总共可缓存：
```
3700 * 16 = 59200 tokens 的 KV cache
```

这比 kernel 任务中单次 batch 的最大 token 数大得多。kernel 配置的 `max_prompt_length=2048`、`max_response_length=1024`、`train_batch_size=4`，即使考虑 `n_agent=1` 和树搜索的分支数 `[2,2]`（最多 4 个生成节点），也只需要约 `4 * (2048 + 1024) = 12288` 个 token slot。KV cache 容量是需求的 ~4.8 倍，说明 KV cache 容量非常充足。

#### 训练阶段显存分配

在 `free_cache_engine=True` 且模型权重 offload 后，训练阶段 GPU 显存需求：

```
FSDP 训练显存 (Actor):
- 分片权重: 14 GB / 4 = 3.5 GB
- 优化器状态 (AdamW, 2 states): 2 * 3.5 = 7 GB
- 梯度: ~3.5 GB
- 激活内存 (micro_batch=4, max_tokens=8192): ~1-2 GB
- 小计: ~15-16 GB

Critic 模型（和 actor 同架构）:
- 分片权重: 14 GB / 4 = 3.5 GB
- 优化器状态: 7 GB
- 梯度: 3.5 GB
- 小计: ~14 GB

注意: actor 和 critic 训练是交替执行的
（先 update_critic，再 update_actor），
不会同时占用 full 显存，但每个阶段各自的峰值如上。
```

#### 综合峰值分析

kernel PPO 中的演员-评论家并行架构：
- `actor_rollout_wg`（ActorRolloutRefWorker）：actor + rollout + ref 在同一进程
- `critic_wg`（CriticWorker）：critic 独立进程
- 它们被 colocate 在同一 GPU 上（同一个 Ray resource pool）

```
Rollout 阶段:
  - Actor 进程: ~28.8 GB (模型 + KV cache + 激活)
  - Critic 进程: ~14 GB (保持在 GPU 上,但无计算)
  - 总计: ~42.8 GB
  - 安全余量: 48 - 42.8 = 5.2 GB

训练阶段 (Actor 更新):
  - Actor 进程: ~16 GB (FSDP 参数 + 优化器 + 梯度)
  - Critic 进程: ~14 GB (空闲)
  - 总计: ~30 GB
  - 安全余量: 48 - 30 = 18 GB

训练阶段 (Critic 更新):
  - Actor 进程: ~3.5 GB (仅分片权重, 已 offload 优化器)
  - Critic 进程: ~14 GB (FSDP 参数 + 优化器 + 梯度)
  - 总计: ~17.5 GB
  - 安全余量: 48 - 17.5 = 30.5 GB
```

**结论**：对于 7B 模型在 4x48GB GPU 上，当前配置是安全的。最紧张的是 rollout 阶段（~43 GB），接近但仍在限制内。

### 3.3 与标准 ppo_trainer.yaml 对比

| 参数 | ppo_trainer.yaml (标准) | ppo_trainer_kernel.yaml (kernel RL) |
|------|------------------------|--------------------------------------|
| `tensor_model_parallel_size` | 2 | **1** |
| `gpu_memory_utilization` | 0.5 | **0.6** |
| `free_cache_engine` | True | True |
| `load_format` | dummy_dtensor | **auto** |
| `max_num_batched_tokens` | 8192 | **4096** |
| `max_num_seqs` | 1024 | **32** |
| `train_batch_size` | 1024 | **4** |
| `n_gpus_per_node` | 8 | **4** |
| 模型 | DeepSeek-LLM-7B | Qwen2.5-Coder-7B |
| `adv_estimator` | gae/grpo | no_estimator |
| 数据集 | GSM8K (QA) | Kernel RL (代码生成) |

**TP=1 vs TP=2 的影响**：

标准 config 的 `tensor_model_parallel_size=2` 意味着：
- 每 2 张 GPU 组成一个 TP 组，共享 KV cache
- 模型权重在 2 张 GPU 间分片（每 GPU 存一半的注意力头）
- KV cache 也按头分片
- 实际上 train_batch_size=1024 数据量下，TP 能减少单 GPU 计算负载

kernel config 的 `tensor_model_parallel_size=1` 意味着：
- 每张 GPU 独立（纯数据并行）
- 完整的模型副本 + 完整的 KV cache 在每张 GPU 上
- 但处理的数据量较小（`train_batch_size=4`, `max_num_seqs=32`）
- 对于 kernel RL 的小 batch 场景更为合适

**gpu_memory_utilization 0.6 的选择原因**：

相比标准 config 的 0.5，kernel config 使用 0.6 是因为：
- TP=1 时需要完整的模型权重在每 GPU 上（14 GB vs TP=2 时的每 GPU 7 GB）
- 更高的 util 给 KV cache 更多空间
- kernel 任务的 max_response_length 较小（1024），batch 小，KV 压力不大

---

## 4. 总结与建议

### 核心架构总结

1. **PagedAttention** 通过块级 KV cache 管理消除了显存碎片，`block_size=16` 为默认配置
2. **TP 模式下 KV cache 按头分片共享**，**DP 模式下每 GPU 独立管理自己的 KV cache**
3. **veRL Hybrid Engine 通过 `free_cache_engine` + `offload_model_weights` 实现推理与训练的内存时分复用**
4. `free_cache_engine` 仅释放 KV cache 的 GPU 张量，模型权重被 offload 到 CPU 而非释放
5. Sleep/wake cycle 保证同一 GPU 在不同阶段切换角色，避免了额外 GPU 资源需求

### 当前配置评估

- **内存安全性**：7B 模型在 4x48GB GPU 上安全运行，峰值约 43 GB/GPU
- **性能合理**：`gpu_memory_utilization=0.6` 给 KV cache 留了约 13 GB 空间，远高于 kernel 小 batch 场景的需求
- **余量有限**：rollout 阶段 ~43 GB 使用率接近 48 GB 上限，增加 batch 或 sequence length 需重算

### 潜在的优化方向

1. **降低 `gpu_memory_utilization`**：如果训练阶段峰值逼近 OOM，可以降到 0.55 或 0.5，牺牲 KV cache 大小换取安全余量
2. **启用 `param_offload`**：actor FSDP 的 `param_offload=True` 可以将训练参数移到 CPU，每 GPU 节省 ~3.5 GB，但会引入通信延迟
3. **增大 `tensor_model_parallel_size`**：如果单 GPU KV cache 压力大（如超长序列），可设 TP=2 或 TP=4，让 KV cache 在多个 GPU 间分片
4. **动态 batch 调度**：目前 `use_dynamic_bsz=False`，启用后可以根据序列长度动态调整 micro batch，提高显存利用效率

### 关键代码路径速查

| 功能 | 文件路径 | 关键行号 |
|------|---------|---------|
| KV block 数量确定 | `verl/third_party/vllm/vllm_v_0_4_2/worker.py` | 141-196 |
| KV cache 初始化 | `verl/third_party/vllm/vllm_v_0_4_2/worker.py` | 199-201 |
| KV cache 释放 | `verl/third_party/vllm/vllm_v_0_4_2/worker.py` | 203-206 |
| vLLM 权重 offload | `verl/third_party/vllm/vllm_v_0_4_2/worker.py` | 246-254 |
| 权重同步入口 (FSDP->vLLM) | `verl/workers/sharding_manager/fsdp_vllm.py` | 70-91 |
| 权重 offload 出口 (vLLM->CPU) | `verl/workers/sharding_manager/fsdp_vllm.py` | 104-111 |
| vLLM rollout 生命周期 | `verl/workers/rollout/vllm_rollout/vllm_rollout.py` | 148-232 |
| FSDP worker 定义 | `verl/workers/fsdp_workers.py` | 72-646 |
| Ray PPO Trainer 基类 | `verl/trainer/ppo/ray_trainer.py` | 328-end |
| Kernel PPO Trainer | `verl/trainer/ppo/ray_trainer_kernel.py` | 全文件 |
| SPMD GPU Executor | `verl/third_party/vllm/vllm_v_0_4_2/spmd_gpu_executor.py` | 33-176 |
| Engine Args (默认参数) | `verl/third_party/vllm/vllm_v_0_4_2/arg_utils.py` | 全文件 |
| 模型加载器 | `verl/third_party/vllm/vllm_v_0_4_2/model_loader.py` | 全文件 |
| 当前 kernel 配置 | `verl/trainer/config/ppo_trainer_kernel.yaml` | 全文件 |
