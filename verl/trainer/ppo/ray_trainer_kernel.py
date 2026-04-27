from __future__ import annotations

from pprint import pprint
from typing import Optional

import torch
from omegaconf import OmegaConf, open_dict
from verl import DataProto

from search_r1.kernel_rl.advantage import compute_multi_section_advantages
from search_r1.kernel_rl.dataset import KernelRLDataset
from search_r1.kernel_rl.prompt_builder import KernelPromptBuilder
from search_r1.kernel_rl.tree_manager import KernelTreeSearchConfig, KernelTreeSearchManager
from search_r1.kernel_rl.scorer import KernelScoringPool
from verl.trainer.ppo.ray_trainer_ts import (
    RayPPOTrainer as LegacyRayPPOTrainer,
    ResourcePoolManager,
    Role,
    _timer,
    compute_data_metrics,
    compute_timing_metrics,
    reduce_metrics,
)


class RayKernelPPOTrainer(LegacyRayPPOTrainer):
    """
    Kernel-development PPO trainer.

    This trainer reuses the existing veRL actor / rollout / checkpointing stack,
    but replaces the QA-oriented tree-search loop with a true level-wise tree
    rollout driven by ``search_r1.kernel_rl``.
    """

    def __init__(
        self,
        *args,
        scoring_pool: KernelScoringPool,
        prompt_builder: Optional[KernelPromptBuilder] = None,
        **kwargs,
    ):
        self.scoring_pool = scoring_pool
        self.prompt_builder = prompt_builder or KernelPromptBuilder()
        super().__init__(*args, **kwargs)

    def _create_dataloader(self):
        """
        Create kernel-oriented train / validation dataloaders.

        The dataset keeps ``task_spec`` and ``bench_spec`` as structured fields,
        while the prompt builder decides how the root prompt should be composed.
        """
        from torch.utils.data import DataLoader
        from verl.utils.dataset.rl_dataset import collate_fn

        self.train_dataset = KernelRLDataset(
            parquet_files=self.config.data.train_files,
            tokenizer=self.tokenizer,
            prompt_builder=self.prompt_builder,
            prompt_key=self.config.data.prompt_key,
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get('return_raw_chat', False),
            truncation=self.config.data.get('prompt_truncation', 'left'),
            task_spec_key=self.config.data.get('task_spec_key', 'task_spec'),
            bench_spec_key=self.config.data.get('bench_spec_key', 'bench_spec'),
            reference_python_key=self.config.data.get('reference_python_key', 'reference_python'),
        )
        if self.config.data.train_data_num is not None:
            if self.config.data.train_data_num > len(self.train_dataset.dataframe):
                print(
                    "[WARNING] training dataset size is smaller than desired size. "
                    f"Using the original size {len(self.train_dataset.dataframe)}."
                )
            else:
                self.train_dataset.dataframe = self.train_dataset.dataframe.sample(
                    self.config.data.train_data_num,
                    random_state=42,
                )
        print(f"filtered training dataset size: {len(self.train_dataset.dataframe)}")

        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            shuffle=self.config.data.shuffle_train_dataloader,
            drop_last=True,
            collate_fn=collate_fn,
        )

        val_files = self.config.data.get('val_files')
        self.val_dataloader = None
        if val_files:
            self.val_dataset = KernelRLDataset(
                parquet_files=val_files,
                tokenizer=self.tokenizer,
                prompt_builder=self.prompt_builder,
                prompt_key=self.config.data.prompt_key,
                max_prompt_length=self.config.data.max_prompt_length,
                filter_prompts=True,
                return_raw_chat=self.config.data.get('return_raw_chat', False),
                truncation=self.config.data.get('prompt_truncation', 'left'),
                task_spec_key=self.config.data.get('task_spec_key', 'task_spec'),
                bench_spec_key=self.config.data.get('bench_spec_key', 'bench_spec'),
                reference_python_key=self.config.data.get('reference_python_key', 'reference_python'),
            )
            if self.config.data.val_data_num is not None:
                if self.config.data.val_data_num > len(self.val_dataset.dataframe):
                    print(
                        "[WARNING] validation dataset size is smaller than desired size. "
                        f"Using the original size {len(self.val_dataset.dataframe)}."
                    )
                else:
                    self.val_dataset.dataframe = self.val_dataset.dataframe.sample(
                        self.config.data.val_data_num,
                        random_state=42,
                    )
            print(f"filtered validation dataset size: {len(self.val_dataset.dataframe)}")
            self.val_dataloader = DataLoader(
                dataset=self.val_dataset,
                batch_size=self.config.data.val_batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn,
            )
            print(f'Size of val dataloader: {len(self.val_dataloader)}')

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        assert len(self.train_dataloader) >= 1

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            if hasattr(self.config, "critic") and hasattr(self.config.critic, "optim"):
                self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self):
        """
        Phase-1 validation placeholder.

        The kernel scorer is still log-only in this phase, so validation metrics
        would be misleading. The hook stays here so later phases can extend it
        without changing the trainer interface again.
        """
        return {}

    def _build_tree_manager(self) -> KernelTreeSearchManager:
        """Construct the kernel tree manager from config once per training run."""
        tree_config = KernelTreeSearchConfig(
            max_depth=self.config.kernel.max_depth,
            branch_factors=list(self.config.kernel.branch_factors),
            keep_per_depth=list(self.config.kernel.keep_per_depth),
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            prompt_truncation=self.config.data.get('prompt_truncation', 'left'),
        )
        return KernelTreeSearchManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=tree_config,
            scoring_pool=self.scoring_pool,
            prompt_builder=self.prompt_builder,
        )

    def fit(self):
        """
        Train with true level-wise tree rollout and node-level PPO samples.

        Rollout ownership:
        - ``KernelTreeSearchManager`` owns tree expansion, parser calls, and
          scorer requests.
        - The exporter returns one PPO sample per generated node.
        - veRL actor / ref / critic workers still own log-prob recomputation and
          gradient updates.
        """
        pprint(
            "Kernel trainer config check: "
            f"adv_estimator={self.config.algorithm.adv_estimator}, "
            f"kernel_adv_estimator={self.config.algorithm.kernel_adv_estimator}"
        )

        if self.config.algorithm.use_kl_in_reward:
            raise NotImplementedError(
                "Phase1 kernel trainer does not support KL folded into token rewards. "
                "Keep algorithm.use_kl_in_reward=False."
            )

        logger = self.logger
        self.global_steps = 0

        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            if val_metrics:
                pprint(f'Initial validation metrics: {val_metrics}')
                logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        self.global_steps += 1
        tree_manager = self._build_tree_manager()

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                print(f'[########] kernel epoch {epoch}, step {self.global_steps}')
                metrics = {}
                timing_raw = {}

                batch = DataProto.from_single_dict(batch_dict)

                with _timer('step', timing_raw):
                    with _timer('gen', timing_raw):
                        _, train_batch = tree_manager.run_tree_rollout(batch)

                    # Keep id / mask tensors in integer dtype. Scores and log-probs
                    # stay float because they participate in reward / advantage math.
                    float_keys = {
                        'old_log_probs',
                        'ref_log_prob',
                        'token_level_scores',
                        'token_level_rewards',
                        'advantages',
                        'returns',
                        'design_token_scores',
                        'code_token_scores',
                        'predict_token_scores',
                        'design_advantages',
                        'code_advantages',
                        'predict_advantages',
                        'design_returns',
                        'code_returns',
                        'predict_returns',
                    }
                    for key in train_batch.batch.keys():
                        if key not in float_keys:
                            train_batch.batch[key] = train_batch.batch[key].long()

                    with torch.no_grad():
                        output = self.actor_rollout_wg.compute_log_prob(train_batch)
                        train_batch = train_batch.union(output)

                    self._balance_batch(train_batch, metrics=metrics)
                    train_batch.meta_info['global_token_num'] = torch.sum(
                        train_batch.batch['attention_mask'],
                        dim=-1,
                    ).tolist()

                    if self.use_reference_policy:
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(train_batch)
                            train_batch = train_batch.union(ref_log_prob)

                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(train_batch)
                            train_batch = train_batch.union(values)

                    with _timer('adv', timing_raw):
                        train_batch = compute_multi_section_advantages(
                            train_batch,
                            adv_estimator=self.config.algorithm.kernel_adv_estimator,
                        )

                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(train_batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(train_batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics = self._validate()
                        if val_metrics:
                            metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                        self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                metrics.update(
                    compute_data_metrics(
                        batch=train_batch,
                        use_critic=self.use_critic,
                        reward_mode='kernel',
                    )
                )
                metrics.update(compute_timing_metrics(batch=train_batch, timing_raw=timing_raw))
                print(self.global_steps, metrics)
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1
                if self.global_steps >= self.total_training_steps:
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        if val_metrics:
                            pprint(f'Final validation metrics: {val_metrics}')
                            logger.log(data=val_metrics, step=self.global_steps)

                    with _timer('save_checkpoint', timing_raw):
                        self._save_checkpoint()
                    return
