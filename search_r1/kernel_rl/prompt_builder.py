from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class KernelPromptBuilder:
    design_open_tag: str = "<design>"
    design_close_tag: str = "</design>"
    code_open_tag: str = "<code>"
    code_close_tag: str = "</code>"
    predict_open_tag: str = "<predict>"
    predict_close_tag: str = "</predict>"

    # ── shared rules ─────────────────────────────────────────────────

    _HARD_RULES = (
        "CRITICAL RULES — our automated benchmark will enforce these:\n"
        "1. Write a STANDALONE Triton kernel with explicit grid launch.\n"
        "   DO NOT call torch.nn.functional or any PyTorch op inside your kernel.\n"
        "   DO NOT use torch._inductor patterns. DO NOT wrap a PyTorch call.\n"
        "2. COMPILATION: if your code fails to compile → code reward = 0.\n"
        "3. CORRECTNESS: if output does not match reference → code reward = 0.\n"
        "4. PERFORMANCE: timed against torch.compile.\n"
        "   - Slower than torch.compile → PENALTY (low reward)\n"
        "   - Faster than torch.compile → BONUS (full reward)\n"
        "   - 2× faster than torch.compile → MAX BONUS\n"
        "5. Do not write empty <code> sections. A missing kernel is worse than\n"
        "   a buggy kernel — at least a buggy kernel gives us something to debug.\n"
        "6. Your <predict> section MUST include an estimated speedup number vs PyTorch."
    )

    # ── format instructions ──────────────────────────────────────────

    def output_format_instructions(self) -> str:
        return (
            "Respond with exactly three sections in this order:\n\n"
            f"{self.design_open_tag}\n"
            "Optimization strategy. Be specific: which block size and why, "
            "what memory layout, what tiling strategy. If this is a revision "
            "round, start by briefly critiquing the PREVIOUS attempt.\n"
            f"{self.design_close_tag}\n\n"
            f"{self.code_open_tag}\n"
            "Complete standalone Triton kernel: @triton.jit decorated function\n"
            "PLUS a wrapper function with explicit grid=(...) launch.\n"
            f"{self.code_close_tag}\n\n"
            f"{self.predict_open_tag}\n"
            "Expected speedup vs PyTorch, bottleneck analysis, confidence level.\n"
            f"{self.predict_close_tag}"
        )

    # ── root prompt ──────────────────────────────────────────────────

    def build_root_prompt(
        self,
        task_spec: Dict[str, Any],
        reference_python: str,
        extra_instruction: str = "",
    ) -> str:
        task_text = json.dumps(task_spec, ensure_ascii=False, indent=2, sort_keys=True)
        extra = f"\n\n{extra_instruction}" if extra_instruction else ""
        return (
            "You are writing a high-performance Triton GPU kernel.\n\n"
            f"{self._HARD_RULES}\n\n"
            "Task specification:\n"
            f"{task_text}\n\n"
            "Reference PyTorch implementation (what you must beat):\n"
            "```python\n"
            f"{reference_python}\n"
            "```\n\n"
            f"{self.output_format_instructions()}{extra}"
        )

    # ── child prompt (iterative refinement) ──────────────────────────

    def build_child_prompt(
        self,
        parent_prompt_text: str,
        parent_response_text: str,
        env_feedback_text: str,
        depth: int,
    ) -> str:
        feedback = env_feedback_text or "No benchmark feedback is available yet."
        return (
            f"{self._HARD_RULES}\n\n"
            "Original task:\n"
            f"{parent_prompt_text}\n\n"
            f"--- Previous attempt at depth {depth} ---\n"
            f"{parent_response_text}\n"
            "--- End of previous attempt ---\n\n"
            "Benchmark results from automated scorer:\n"
            f"{feedback}\n\n"
            "Your task: Analyze what went wrong above. In your <design> section:\n"
            "1. Critique the previous version (what failed, why)\n"
            "2. Propose your improved strategy\n"
            "Then write a NEW kernel that fixes the issues.\n\n"
            f"{self.output_format_instructions()}"
        )

    # ── dataset row → prompt ─────────────────────────────────────────

    def build_prompt_from_row(self, row: Dict[str, Any]) -> str:
        prompt = row.get("prompt")
        if prompt:
            return prompt
        task_spec = row.get("task_spec", {})
        reference_python = row.get("reference_python", "")
        extra_instruction = row.get("extra_instruction", "")
        return self.build_root_prompt(
            task_spec, reference_python, extra_instruction=extra_instruction
        )
