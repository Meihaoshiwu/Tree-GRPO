from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

# The three semantic sections of a kernel-development response.
KERNEL_SECTION_NAMES: Tuple[str, str, str] = ("design", "code", "predict")


@dataclass
class KernelScoreResult:
    """
    Standardized scorer output shared between rollout and training.

    ``feedback_text`` is the natural-language environment feedback that may be
    appended to the next-round prompt.
    ``scalar_rewards`` stores the node-level rewards before they are projected
    to token space for PPO.
    """

    status: str = "pending"
    feedback_text: str = ""
    scalar_rewards: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    error_type: str = ""

    def get_reward(self, section: str) -> float:
        """Return the scalar reward for one section with a safe default."""
        return float(self.scalar_rewards.get(section, 0.0))


@dataclass
class ParsedKernelResponse:
    """
    Parsed ``design / code / predict`` response and its spans.

    ``section_char_spans`` and ``section_token_spans`` both use half-open
    intervals ``[start, end)``.
    """

    raw_text: str
    section_texts: Dict[str, str] = field(default_factory=dict)
    section_char_spans: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    section_token_spans: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    missing_sections: List[str] = field(default_factory=list)

    def get_text(self, section: str) -> str:
        """Return the text of a section, or an empty string if missing."""
        return self.section_texts.get(section, "")


@dataclass
class KernelTreeNode:
    """
    One rollout node in the kernel-development reasoning tree.

    Important variable semantics:

    - ``prompt_text`` is the human-readable prompt string used in *this* round.
      It is kept for logging, debugging, and exact prompt reconstruction.
    - ``prompt_ids`` is the tokenized and padded tensor actually fed into the
      model in *this* round. It is not identical to ``prompt_text`` because
      tokenization, truncation, and left-padding may change the concrete tensor
      representation.
    - ``response_*`` fields only describe the *newly generated increment* of
      this node. They do not contain ancestor outputs.
    - The next round prompt is built from this node's
      ``prompt_text + response_text + env_feedback_text`` on demand. It is not
      stored here because that future prompt belongs to this node's children,
      not to this node's own PPO sample.
    """

    tree_uid: str
    node_uid: str
    parent_uid: Optional[str]
    depth: int

    # Original task description and benchmark metadata.
    task_spec: Dict[str, Any] = field(default_factory=dict)
    bench_spec: Dict[str, Any] = field(default_factory=dict)

    # Current-round model input.
    prompt_text: str = ""
    prompt_ids: Optional[torch.Tensor] = None
    prompt_attention_mask: Optional[torch.Tensor] = None
    prompt_position_ids: Optional[torch.Tensor] = None

    # Current-round model output increment.
    response_text: str = ""
    response_ids: Optional[torch.Tensor] = None
    response_attention_mask: Optional[torch.Tensor] = None
    old_log_probs: Optional[torch.Tensor] = None

    # Parsed output and environment result for this node.
    parsed_response: Optional[ParsedKernelResponse] = None
    env_feedback_text: str = ""
    score_result: Optional[KernelScoreResult] = None

    # Scalar node-level rewards. These are semantic rewards, not token vectors.
    scalar_design_reward: float = 0.0
    scalar_code_reward: float = 0.0
    scalar_predict_reward: float = 0.0

    # Structured metadata used for analysis / pruning / debugging.
    status: str = "pending"
    metrics: Dict[str, Any] = field(default_factory=dict)
    children: List["KernelTreeNode"] = field(default_factory=list)
    parent: Optional["KernelTreeNode"] = field(default=None, repr=False)

    @property
    def is_root(self) -> bool:
        """Whether this node is the synthetic root that owns the original task."""
        return self.parent_uid is None

    def add_child(self, child: "KernelTreeNode") -> None:
        """Link a new child node while preserving a direct parent pointer."""
        child.parent = self
        self.children.append(child)

    def path_from_root(self) -> List["KernelTreeNode"]:
        """Return the ordered root-to-self path for logging or DFS export."""
        nodes: List["KernelTreeNode"] = []
        current: Optional["KernelTreeNode"] = self
        while current is not None:
            nodes.append(current)
            current = current.parent
        return list(reversed(nodes))

    def total_reward(self) -> float:
        """Convenience scalar used by simple pruning heuristics."""
        return self.scalar_design_reward + self.scalar_code_reward + self.scalar_predict_reward


@dataclass
class KernelTrainSample:
    """
    Dense PPO sample exported from one tree node.

    This object is intentionally separate from ``KernelTreeNode``:
    tree nodes keep sparse rollout state, while train samples keep the dense
    tensors needed only during PPO update.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    responses: torch.Tensor

    # ``loss_mask`` is the union of all trainable sections for this sample.
    loss_mask: torch.Tensor

    # Section masks are aligned with ``responses`` and mark the newly generated
    # tokens of each semantic section.
    design_mask: torch.Tensor
    code_mask: torch.Tensor
    predict_mask: torch.Tensor

    # Token-level scores are the vectorized training projection of node-level
    # scalar rewards. They are created at export time, not stored on the node.
    design_token_scores: torch.Tensor
    code_token_scores: torch.Tensor
    predict_token_scores: torch.Tensor
    token_level_scores: torch.Tensor

    # Metadata used by grouping-based estimators such as GRPO.
    uid: str
    tree_uid: str
    node_uid: str
    parent_uid: str
    depth: int

