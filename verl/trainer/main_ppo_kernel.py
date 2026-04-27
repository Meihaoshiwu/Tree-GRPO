# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Kernel-development PPO main entrypoint.
"""

import hydra
import ray

from search_r1.kernel_rl.prompt_builder import KernelPromptBuilder
from search_r1.kernel_rl.scorer import KernelScoringPool
from verl.trainer.ppo.ray_trainer_kernel import RayKernelPPOTrainer


@hydra.main(config_path='config', config_name='ppo_trainer_kernel', version_base=None)
def main(config):
    if not ray.is_initialized():
        print("Ray is not initialized! run ray.init()...")
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})
    print("Ray already initialized. Get remote ...")
    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.utils import hf_tokenizer

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(local_path)

    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup
    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup
    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer_kernel import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
    }

    if config.algorithm.adv_estimator == 'gae':
        role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)

    if config.actor_rollout_ref.actor.use_kl_loss or config.algorithm.use_kl_in_reward:
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {role: global_pool_id for role in role_worker_mapping.keys()}

    scoring_pool = KernelScoringPool.create(
        num_workers=config.kernel.scorer.num_workers,
        log_dir=config.kernel.scorer.log_dir,
        mode=config.kernel.scorer.mode,
        timeout_s=config.kernel.scorer.timeout_s,
    )

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayKernelPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=None,
        val_reward_fn=None,
        scoring_pool=scoring_pool,
        prompt_builder=KernelPromptBuilder(),
    )
    print(f"=================init_workers================")
    trainer.init_workers()
    print(f"=================fit================")
    trainer.fit()


if __name__ == '__main__':
    main()
