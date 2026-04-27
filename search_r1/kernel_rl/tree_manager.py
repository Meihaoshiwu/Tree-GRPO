from __future__ import annotations

import uuid
from dataclasses import dataclass, field
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
    """
    Config for true level-wise tree rollout.

    ``max_depth`` follows the user-facing tree depth convention:
    root depth is 0, and the deepest generated nodes are at ``max_depth``.
    """

    max_depth: int
    branch_factors: Sequence[int]
    keep_per_depth: Sequence[int]
    max_prompt_length: int
    max_response_length: int
    prompt_truncation: str = "left"
    prune_strategy: str = "reward_desc_then_fifo"

    def branch_factor_for_parent_depth(self, parent_depth: int) -> int:
        """Return how many children each active node should sample."""
        if parent_depth < len(self.branch_factors):
            return int(self.branch_factors[parent_depth])
        return int(self.branch_factors[-1])

    def keep_count_for_child_depth(self, child_depth: int) -> int:
        """Return how many nodes remain active after one level is expanded."""
        if child_depth - 1 < len(self.keep_per_depth):
            return int(self.keep_per_depth[child_depth - 1])
        return int(self.keep_per_depth[-1])


class KernelTreeSearchManager:
    """
    Level-wise tree rollout manager for kernel-development tasks.

    Each depth is rolled out as one batched vLLM step. Native ``n > 1`` sampling
    is used for multi-branch generation instead of duplicating prompts by hand.
    """

    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: KernelTreeSearchConfig,
        scoring_pool: KernelScoringPool,
        prompt_builder: Optional[KernelPromptBuilder] = None,
        output_parser: Optional[KernelOutputParser] = None,
        exporter: Optional[KernelTrainSampleExporter] = None,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.scoring_pool = scoring_pool
        self.prompt_builder = prompt_builder or KernelPromptBuilder()
        self.output_parser = output_parser or KernelOutputParser()
        self.exporter = exporter or KernelTrainSampleExporter(parser=self.output_parser)

    def run_tree_rollout(self, batch: DataProto) -> Tuple[List[KernelTreeNode], DataProto]:
        """
        Run the full tree rollout and export node-level PPO samples.

        Returns:
            roots: the sparse tree for logging / debugging.
            train_batch: dense PPO samples exported from every generated node.
        """
        roots = self._build_root_nodes(batch)
        active_nodes = roots

        for parent_depth in range(self.config.max_depth):
            if not active_nodes:
                break

            prompt_batch, prompt_states = self._build_prompt_batch(active_nodes)
            branch_factor = self.config.branch_factor_for_parent_depth(parent_depth)
            if branch_factor <= 0:
                break

            rollout_output = self.actor_rollout_wg.generate_sequences(
                prompt_batch,
                {"n": branch_factor},
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

        train_batch = self.exporter.export_nodes(roots)
        return roots, train_batch

    def _build_root_nodes(self, batch: DataProto) -> List[KernelTreeNode]:
        """Build one synthetic root node per original training example."""
        roots: List[KernelTreeNode] = []
        for idx in range(len(batch)):
            data_item = batch[idx]
            tree_uid = str(data_item.non_tensor_batch.get("index", idx))
            prompt_text = data_item.non_tensor_batch.get("prompt_text")
            if not prompt_text:
                prompt_ids = data_item.batch["input_ids"]
                prompt_mask = data_item.batch["attention_mask"]
                valid_prompt_ids = prompt_ids[prompt_mask.bool()]
                prompt_text = self.tokenizer.decode(valid_prompt_ids)

            roots.append(
                KernelTreeNode(
                    tree_uid=tree_uid,
                    node_uid=f"root-{uuid.uuid4()}",
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
        """
        Build a batch of same-depth prompts for one vLLM generation step.

        ``prompt_states`` mirrors the batch order and is later attached to the
        generated children so that each child stores the exact prompt it used.
        """
        prompt_ids_list = []
        prompt_attention_mask_list = []
        prompt_position_ids_list = []
        prompt_states: List[Dict[str, torch.Tensor]] = []

        for node in active_nodes:
            prompt_text = self._build_prompt_for_expansion(node)
            prompt_ids, prompt_attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_text,
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
        """
        Build the next-round prompt for one active node.

        Root nodes already own the original prompt. Non-root nodes must append
        their previous output and environment feedback before the next rollout.
        """
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
        """Create child nodes from one batched rollout output."""
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
                node_uid=str(uuid.uuid4()),
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
        """Store scorer output on the node in a field-by-field explicit way."""
        node.score_result = score_result
        node.status = score_result.status
        node.env_feedback_text = score_result.feedback_text
        node.metrics = dict(score_result.metrics)
        node.scalar_design_reward = score_result.get_reward("design")
        node.scalar_code_reward = score_result.get_reward("code")
        node.scalar_predict_reward = score_result.get_reward("predict")

    def _select_active_children(self, children: List[KernelTreeNode], keep_count: int) -> List[KernelTreeNode]:
        """
        Keep the best children for the next depth.

        Phase 1 uses a simple reward-descending heuristic. When scores are tied,
        insertion order is preserved so the rollout stays deterministic.
        """
        if keep_count <= 0 or keep_count >= len(children):
            return children
        ranked = sorted(children, key=lambda node: node.total_reward(), reverse=True)
        return ranked[:keep_count]
