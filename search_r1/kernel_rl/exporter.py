from __future__ import annotations

from typing import Iterable, List

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto

from .parser import KernelOutputParser
from .schema import KERNEL_SECTION_NAMES, KernelTrainSample, KernelTreeNode


class KernelTrainSampleExporter:
    """
    Export one PPO sample per non-root node.

    This exporter is intentionally node-level rather than leaf-level. It avoids
    over-counting ancestor tokens across many leaves and matches the intended
    training behavior: given the current prompt and feedback, improve the next
    round output.
    """

    def __init__(self, parser: KernelOutputParser):
        self.parser = parser

    def iter_trainable_nodes(self, roots: Iterable[KernelTreeNode]) -> Iterable[KernelTreeNode]:
        """Depth-first traversal that yields every non-root node exactly once."""
        for root in roots:
            stack = list(reversed(root.children))
            while stack:
                node = stack.pop()
                yield node
                if node.children:
                    stack.extend(reversed(node.children))

    def export_nodes(self, roots: Iterable[KernelTreeNode]) -> DataProto:
        """
        Convert tree nodes to a batched ``DataProto`` ready for PPO.

        The exporter keeps tensors on CPU. The worker-side training code already
        moves them to GPU at update time.
        """
        samples = [self._build_sample(node) for node in self.iter_trainable_nodes(roots)]
        if not samples:
            raise ValueError("Tree rollout produced no trainable nodes.")

        tensors = {
            "input_ids": torch.stack([sample.input_ids for sample in samples], dim=0),
            "attention_mask": torch.stack([sample.attention_mask for sample in samples], dim=0),
            "position_ids": torch.stack([sample.position_ids for sample in samples], dim=0),
            "responses": torch.stack([sample.responses for sample in samples], dim=0),
            "loss_mask": torch.stack([sample.loss_mask for sample in samples], dim=0),
            "design_mask": torch.stack([sample.design_mask for sample in samples], dim=0),
            "code_mask": torch.stack([sample.code_mask for sample in samples], dim=0),
            "predict_mask": torch.stack([sample.predict_mask for sample in samples], dim=0),
            "design_token_scores": torch.stack([sample.design_token_scores for sample in samples], dim=0),
            "code_token_scores": torch.stack([sample.code_token_scores for sample in samples], dim=0),
            "predict_token_scores": torch.stack([sample.predict_token_scores for sample in samples], dim=0),
            "token_level_scores": torch.stack([sample.token_level_scores for sample in samples], dim=0),
        }
        non_tensors = {
            "uid": np.array([sample.uid for sample in samples], dtype=object),
            "tree_uid": np.array([sample.tree_uid for sample in samples], dtype=object),
            "node_uid": np.array([sample.node_uid for sample in samples], dtype=object),
            "parent_uid": np.array([sample.parent_uid for sample in samples], dtype=object),
            "depth": np.array([sample.depth for sample in samples], dtype=object),
        }
        batch = TensorDict(source=tensors, batch_size=(len(samples),))
        return DataProto(batch=batch, non_tensor_batch=non_tensors)

    def _build_sample(self, node: KernelTreeNode) -> KernelTrainSample:
        """Create one dense sample from one node."""
        if node.prompt_ids is None or node.prompt_attention_mask is None or node.prompt_position_ids is None:
            raise ValueError(f"Node {node.node_uid} is missing prompt tensors.")
        if node.response_ids is None or node.response_attention_mask is None:
            raise ValueError(f"Node {node.node_uid} is missing response tensors.")

        response_length = node.response_ids.shape[-1]
        token_spans = {}
        if node.parsed_response is not None:
            token_spans = node.parsed_response.section_token_spans
        section_masks = self.parser.build_section_masks(response_length=response_length, token_spans=token_spans)

        input_ids = torch.cat([node.prompt_ids, node.response_ids], dim=-1)
        attention_mask = torch.cat([node.prompt_attention_mask, node.response_attention_mask], dim=-1)

        # Position ids are continued from the padded prompt representation.
        prompt_last_position = int(node.prompt_position_ids[-1].item())
        response_delta = torch.arange(1, response_length + 1, dtype=node.prompt_position_ids.dtype)
        response_position_ids = node.prompt_position_ids[-1:] + response_delta
        position_ids = torch.cat([node.prompt_position_ids, response_position_ids], dim=-1)

        design_mask = section_masks["design"].to(dtype=torch.int64)
        code_mask = section_masks["code"].to(dtype=torch.int64)
        predict_mask = section_masks["predict"].to(dtype=torch.int64)
        loss_mask = torch.clamp(design_mask + code_mask + predict_mask, max=1)

        # Early in RL, the model may fail to respect the required output format.
        # We still export a trainable sample by falling back to "all valid tokens
        # belong to the code section". This keeps PPO numerically stable while
        # preserving the parser's missing-section metadata for later analysis.
        if int(loss_mask.sum().item()) == 0:
            valid_response_length = int(node.response_attention_mask.sum().item())
            if valid_response_length > 0:
                code_mask[:valid_response_length] = 1
                loss_mask = code_mask.clone()

        design_token_scores = design_mask.float() * float(node.scalar_design_reward)
        code_token_scores = code_mask.float() * float(node.scalar_code_reward)
        predict_token_scores = predict_mask.float() * float(node.scalar_predict_reward)
        token_level_scores = design_token_scores + code_token_scores + predict_token_scores

        return KernelTrainSample(
            input_ids=input_ids.long(),
            attention_mask=attention_mask.long(),
            position_ids=position_ids.long(),
            responses=node.response_ids.long(),
            loss_mask=loss_mask.long(),
            design_mask=design_mask,
            code_mask=code_mask,
            predict_mask=predict_mask,
            design_token_scores=design_token_scores.float(),
            code_token_scores=code_token_scores.float(),
            predict_token_scores=predict_token_scores.float(),
            token_level_scores=token_level_scores.float(),
            uid=node.tree_uid,
            tree_uid=node.tree_uid,
            node_uid=node.node_uid,
            parent_uid=node.parent_uid or "",
            depth=node.depth,
        )
