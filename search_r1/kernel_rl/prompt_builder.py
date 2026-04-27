from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class KernelPromptBuilder:
    """
    Build root prompts and iterative refinement prompts for kernel development.

    The builder always asks the model to emit three explicit sections so that
    later scoring and advantage assignment can stay section-aware.
    """

    design_open_tag: str = "<design>"
    design_close_tag: str = "</design>"
    code_open_tag: str = "<code>"
    code_close_tag: str = "</code>"
    predict_open_tag: str = "<predict>"
    predict_close_tag: str = "</predict>"

    def output_format_instructions(self) -> str:
        """Return the exact response contract expected by the parser."""
        return (
            "Respond with exactly three sections in this order:\n"
            f"{self.design_open_tag}\n"
            "Explain the optimization idea, correctness constraints, and memory / launch tradeoffs.\n"
            f"{self.design_close_tag}\n"
            f"{self.code_open_tag}\n"
            "Provide the Triton kernel code or the edited code block.\n"
            f"{self.code_close_tag}\n"
            f"{self.predict_open_tag}\n"
            "Estimate performance, expected bottlenecks, and confidence in the estimate.\n"
            f"{self.predict_close_tag}"
        )

    def build_root_prompt(
        self,
        task_spec: Dict[str, Any],
        reference_python: str,
        extra_instruction: str = "",
    ) -> str:
        """
        Build the root prompt from task description and reference Python code.

        ``task_spec`` is serialized as readable JSON so that dataset authors can
        pass structured fields without manually flattening them into one string.
        """
        task_text = json.dumps(task_spec, ensure_ascii=False, indent=2, sort_keys=True)
        extra_block = f"\n\nAdditional instruction:\n{extra_instruction}" if extra_instruction else ""
        return (
            "You are improving a Triton operator implementation.\n"
            "Read the task carefully, preserve correctness, and optimize for real GPU performance.\n\n"
            "Task specification:\n"
            f"{task_text}\n\n"
            "Reference Python implementation:\n"
            "```python\n"
            f"{reference_python}\n"
            "```\n\n"
            f"{self.output_format_instructions()}{extra_block}"
        )

    def build_child_prompt(
        self,
        parent_prompt_text: str,
        parent_response_text: str,
        env_feedback_text: str,
        depth: int,
    ) -> str:
        """
        Build the next-round prompt from the previous round state.

        The child prompt keeps the original instruction, appends the parent's
        generated result, and injects environment feedback for iterative
        refinement.
        """
        feedback = env_feedback_text or "No benchmark feedback is available yet."
        return (
            f"{parent_prompt_text}\n\n"
            f"Previous round result at depth {depth}:\n"
            f"{parent_response_text}\n\n"
            "Environment feedback from the previous round:\n"
            f"{feedback}\n\n"
            "Revise the design and code using the feedback above.\n"
            f"{self.output_format_instructions()}"
        )

    def build_prompt_from_row(self, row: Dict[str, Any]) -> str:
        """
        Build one root prompt from a dataset row.

        If the row already provides a fully formatted ``prompt`` string, it is
        used directly. Otherwise the prompt is composed from ``task_spec`` and
        ``reference_python``.
        """
        prompt = row.get("prompt")
        if prompt:
            return prompt
        task_spec = row.get("task_spec", {})
        reference_python = row.get("reference_python", "")
        extra_instruction = row.get("extra_instruction", "")
        return self.build_root_prompt(task_spec, reference_python, extra_instruction=extra_instruction)
