from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F

from .exporter import KernelTrainSampleExporter
from .parser import KernelOutputParser
from .prompt_builder import KernelPromptBuilder
from .scorer import KernelScoreRequest, KernelScoringPool
from .schema import KernelScoreResult, KernelTreeNode


@dataclass
class KernelTreeSearchConfig:
    max_depth: int
    branch_factors: Sequence[int]
    keep_per_depth: Sequence[int]
    max_prompt_length: int
    max_response_length: int
    prompt_truncation: str = "left"
    prune_strategy: str = "reward_desc_then_fifo"

    def branch_factor_for_parent_depth(self, parent_depth: int) -> int:
        if parent_depth < len(self.branch_factors):
            return int(self.branch_factors[parent_depth])
        return int(self.branch_factors[-1])

    def keep_count_for_child_depth(self, child_depth: int) -> int:
        if child_depth - 1 < len(self.keep_per_depth):
            return int(self.keep_per_depth[child_depth - 1])
        return int(self.keep_per_depth[-1])


class KernelTreeSearchManager:
    """Level-wise tree rollout manager for kernel-development tasks."""

    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: KernelTreeSearchConfig,
        scoring_pool: KernelScoringPool,
        prompt_builder: Optional[KernelPromptBuilder] = None,
        output_parser: Optional[KernelOutputParser] = None,
        exporter: Optional[KernelTrainSampleExporter] = None,
        tree_log_dir: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.scoring_pool = scoring_pool
        self.prompt_builder = prompt_builder or KernelPromptBuilder()
        self.output_parser = output_parser or KernelOutputParser()
        self.exporter = exporter or KernelTrainSampleExporter(parser=self.output_parser)
        self.tree_log_dir = tree_log_dir or "./kernel_tree_logs"
        os.makedirs(self.tree_log_dir, exist_ok=True)

        # Per-run counter for intra-level node indices
        self._level_counters: Dict[int, int] = {}

    # ------------------------------------------------------------------
    def run_tree_rollout(self, batch: DataProto) -> Tuple[List[KernelTreeNode], DataProto]:
        roots = self._build_root_nodes(batch)
        active_nodes = roots

        for parent_depth in range(self.config.max_depth):
            if not active_nodes:
                break

            prompt_batch, prompt_states = self._build_prompt_batch(active_nodes)
            branch_factor = self.config.branch_factor_for_parent_depth(parent_depth)
            if branch_factor <= 0:
                break

            prompt_batch.meta_info["n"] = branch_factor
            rollout_output = self.actor_rollout_wg.generate_sequences(
                prompts=prompt_batch,
            )

            new_children = self._materialize_children(
                active_nodes=active_nodes,
                prompt_states=prompt_states,
                rollout_output=rollout_output,
                branch_factor=branch_factor,
                child_depth=parent_depth + 1,
            )

            if parent_depth + 1 >= self.config.max_depth:
                active_nodes = []
            else:
                keep_count = self.config.keep_count_for_child_depth(parent_depth + 1)
                active_nodes = self._select_active_children(new_children, keep_count=keep_count)

        # Write per-tree hierarchy log
        self._write_tree_logs(roots)

        train_batch = self.exporter.export_nodes(roots)
        return roots, train_batch

    # ------------------------------------------------------------------
    # Node naming
    # ------------------------------------------------------------------

    @staticmethod
    def _make_node_uid(depth: int, index: int) -> str:
        """Hierarchical node identifier: ``d{depth}_n{index}``."""
        return f"d{depth}_n{index}"

    def _next_node_uid(self, depth: int) -> str:
        idx = self._level_counters.get(depth, 0)
        self._level_counters[depth] = idx + 1
        return self._make_node_uid(depth, idx)

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def _build_root_nodes(self, batch: DataProto) -> List[KernelTreeNode]:
        roots: List[KernelTreeNode] = []
        for idx in range(len(batch)):
            data_item = batch[idx]
            tree_uid = str(data_item.non_tensor_batch.get("index", f"kernel_{idx}"))
            prompt_text = data_item.non_tensor_batch.get("prompt_text")
            if not prompt_text:
                prompt_ids = data_item.batch["input_ids"]
                prompt_mask = data_item.batch["attention_mask"]
                valid_prompt_ids = prompt_ids[prompt_mask.bool()]
                prompt_text = self.tokenizer.decode(valid_prompt_ids)

            roots.append(
                KernelTreeNode(
                    tree_uid=tree_uid,
                    node_uid=self._next_node_uid(0),
                    parent_uid=None,
                    depth=0,
                    task_spec=data_item.non_tensor_batch.get("task_spec", {}) or {},
                    bench_spec=data_item.non_tensor_batch.get("bench_spec", {}) or {},
                    prompt_text=prompt_text,
                    prompt_ids=data_item.batch["input_ids"].cpu(),
                    prompt_attention_mask=data_item.batch["attention_mask"].cpu(),
                    prompt_position_ids=data_item.batch["position_ids"].cpu(),
                    status="root",
                )
            )
        return roots

    def _build_prompt_batch(
        self,
        active_nodes: Sequence[KernelTreeNode],
    ) -> Tuple[DataProto, List[Dict[str, torch.Tensor]]]:
        prompt_ids_list = []
        prompt_attention_mask_list = []
        prompt_position_ids_list = []
        prompt_states: List[Dict[str, torch.Tensor]] = []

        for node in active_nodes:
            prompt_text = self._build_prompt_for_expansion(node)

            # Wrap in chat template so the SFT model sees <|im_start|>assistant\n
            # and knows to generate the three-section response.
            messages = [{"role": "user", "content": prompt_text}]
            chat_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            prompt_ids, prompt_attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=chat_prompt,
                tokenizer=self.tokenizer,
                max_length=self.config.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.config.prompt_truncation,
            )
            prompt_position_ids = compute_position_id_with_mask(prompt_attention_mask)

            prompt_ids = prompt_ids[0].cpu()
            prompt_attention_mask = prompt_attention_mask[0].cpu()
            prompt_position_ids = prompt_position_ids[0].cpu()

            prompt_ids_list.append(prompt_ids)
            prompt_attention_mask_list.append(prompt_attention_mask)
            prompt_position_ids_list.append(prompt_position_ids)
            prompt_states.append(
                {
                    "prompt_text": prompt_text,
                    "prompt_ids": prompt_ids,
                    "prompt_attention_mask": prompt_attention_mask,
                    "prompt_position_ids": prompt_position_ids,
                }
            )

        prompt_batch = DataProto.from_dict(
            tensors={
                "input_ids": torch.stack(prompt_ids_list, dim=0),
                "attention_mask": torch.stack(prompt_attention_mask_list, dim=0),
                "position_ids": torch.stack(prompt_position_ids_list, dim=0),
            },
            meta_info={
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "do_sample": True,
            },
        )
        return prompt_batch, prompt_states

    def _build_prompt_for_expansion(self, node: KernelTreeNode) -> str:
        if node.is_root:
            return node.prompt_text
        return self.prompt_builder.build_child_prompt(
            parent_prompt_text=node.prompt_text,
            parent_response_text=node.response_text,
            env_feedback_text=node.env_feedback_text,
            depth=node.depth,
        )

    def _materialize_children(
        self,
        active_nodes: Sequence[KernelTreeNode],
        prompt_states: Sequence[Dict[str, torch.Tensor]],
        rollout_output: DataProto,
        branch_factor: int,
        child_depth: int,
    ) -> List[KernelTreeNode]:
        requests: List[KernelScoreRequest] = []
        children: List[KernelTreeNode] = []
        prompt_length = prompt_states[0]["prompt_ids"].shape[-1]

        for flat_index in range(len(rollout_output)):
            parent_index = flat_index // branch_factor
            parent_node = active_nodes[parent_index]
            prompt_state = prompt_states[parent_index]

            data_item = rollout_output[flat_index]
            response_ids = data_item.batch["responses"].cpu()
            full_attention_mask = data_item.batch["attention_mask"].cpu()
            response_attention_mask = full_attention_mask[prompt_length:]

            valid_response_ids = response_ids[response_attention_mask.bool()]
            response_text = self.tokenizer.decode(valid_response_ids)
            parsed_response = self.output_parser.parse(response_text=response_text, tokenizer=self.tokenizer)

            child = KernelTreeNode(
                tree_uid=parent_node.tree_uid,
                node_uid=self._next_node_uid(child_depth),
                parent_uid=parent_node.node_uid,
                depth=child_depth,
                task_spec=parent_node.task_spec,
                bench_spec=parent_node.bench_spec,
                prompt_text=prompt_state["prompt_text"],
                prompt_ids=prompt_state["prompt_ids"],
                prompt_attention_mask=prompt_state["prompt_attention_mask"],
                prompt_position_ids=prompt_state["prompt_position_ids"],
                response_text=response_text,
                response_ids=response_ids,
                response_attention_mask=response_attention_mask,
                parsed_response=parsed_response,
                status="generated",
            )
            parent_node.add_child(child)
            children.append(child)

            requests.append(
                KernelScoreRequest(
                    tree_uid=child.tree_uid,
                    node_uid=child.node_uid,
                    parent_uid=child.parent_uid or "",
                    depth=child.depth,
                    task_spec=child.task_spec,
                    bench_spec=child.bench_spec,
                    prompt_text=child.prompt_text,
                    response_text=child.response_text,
                    code_text=parsed_response.get_text("code"),
                    design_text=parsed_response.get_text("design"),
                    predict_text=parsed_response.get_text("predict"),
                    metadata={
                        "missing_sections": parsed_response.missing_sections,
                    },
                )
            )

        results = self.scoring_pool.score_many(requests)
        for child, score_result in zip(children, results):
            self._attach_score_result(child, score_result)
        return children

    def _attach_score_result(self, node: KernelTreeNode, score_result: KernelScoreResult) -> None:
        node.score_result = score_result
        node.status = score_result.status
        node.env_feedback_text = score_result.feedback_text
        node.metrics = dict(score_result.metrics)
        node.scalar_design_reward = score_result.get_reward("design")
        node.scalar_code_reward = score_result.get_reward("code")
        node.scalar_predict_reward = score_result.get_reward("predict")

    def _select_active_children(self, children: List[KernelTreeNode], keep_count: int) -> List[KernelTreeNode]:
        if keep_count <= 0 or keep_count >= len(children):
            return children
        ranked = sorted(children, key=lambda node: node.total_reward(), reverse=True)
        return ranked[:keep_count]

    # ------------------------------------------------------------------
    # Per-tree logging
    # ------------------------------------------------------------------

    def _write_tree_logs(self, roots: List[KernelTreeNode]) -> None:
        """Write one directory per tree, one file per node, showing full prompt chain."""
        for root in roots:
            tree_dir = os.path.join(self.tree_log_dir, f"tree_{root.tree_uid}")
            os.makedirs(tree_dir, exist_ok=True)
            self._write_node_file(root, tree_dir)

    def _write_node_file(self, node: KernelTreeNode, tree_dir: str) -> None:
        path = os.path.join(tree_dir, f"{node.node_uid}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"=== NODE: {node.node_uid} ===\n")
            f.write(f"=== TREE: {node.tree_uid} ===\n")
            f.write(f"=== PARENT: {node.parent_uid or 'ROOT'} ===\n")
            f.write(f"=== DEPTH: {node.depth} ===\n")
            f.write(f"=== STATUS: {node.status} ===\n")
            if not node.is_root:
                f.write(f"=== REWARDS: design={node.scalar_design_reward:.3f} "
                        f"code={node.scalar_code_reward:.3f} "
                        f"predict={node.scalar_predict_reward:.3f} ===\n")
            f.write("\n")

            # Full prompt sent to model
            f.write("--- PROMPT (sent to model) ---\n")
            f.write(node.prompt_text)
            f.write("\n\n")

            # Model response
            if node.response_text:
                f.write("--- MODEL RESPONSE ---\n")
                f.write(node.response_text)
                f.write("\n\n")

            # Scorer feedback
            if node.env_feedback_text:
                f.write("--- SCORER FEEDBACK ---\n")
                f.write(node.env_feedback_text)
                f.write("\n\n")

            # Metrics
            if node.metrics:
                f.write("--- METRICS ---\n")
                for k, v in sorted(node.metrics.items()):
                    f.write(f"  {k}: {v}\n")
                f.write("\n")

        for child in node.children:
            self._write_node_file(child, tree_dir)
