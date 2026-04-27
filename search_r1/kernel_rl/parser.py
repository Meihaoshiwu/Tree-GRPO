from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from .schema import KERNEL_SECTION_NAMES, ParsedKernelResponse


@dataclass
class KernelOutputParser:
    """
    Parse the model response into ``design / code / predict`` sections.

    The parser also derives token spans when a fast tokenizer is available. The
    token spans are later transformed into section masks for multi-advantage PPO.
    """

    design_open_tag: str = "<design>"
    design_close_tag: str = "</design>"
    code_open_tag: str = "<code>"
    code_close_tag: str = "</code>"
    predict_open_tag: str = "<predict>"
    predict_close_tag: str = "</predict>"

    def parse(self, response_text: str, tokenizer=None) -> ParsedKernelResponse:
        """Parse the raw response text and optionally compute token spans."""
        patterns = {
            "design": self._compile_section_pattern(self.design_open_tag, self.design_close_tag),
            "code": self._compile_section_pattern(self.code_open_tag, self.code_close_tag),
            "predict": self._compile_section_pattern(self.predict_open_tag, self.predict_close_tag),
        }

        section_texts: Dict[str, str] = {}
        char_spans: Dict[str, Tuple[int, int]] = {}
        missing_sections = []

        for section, pattern in patterns.items():
            match = pattern.search(response_text)
            if match is None:
                section_texts[section] = ""
                missing_sections.append(section)
                continue
            content = match.group("content").strip()
            span = match.span("content")
            section_texts[section] = content
            char_spans[section] = span

        token_spans = self._compute_token_spans(
            tokenizer=tokenizer,
            response_text=response_text,
            char_spans=char_spans,
        )

        return ParsedKernelResponse(
            raw_text=response_text,
            section_texts=section_texts,
            section_char_spans=char_spans,
            section_token_spans=token_spans,
            missing_sections=missing_sections,
        )

    @staticmethod
    def _compile_section_pattern(open_tag: str, close_tag: str) -> re.Pattern:
        """Compile one non-greedy section pattern."""
        return re.compile(
            re.escape(open_tag) + r"\s*(?P<content>.*?)\s*" + re.escape(close_tag),
            flags=re.DOTALL,
        )

    def build_section_masks(
        self,
        response_length: int,
        token_spans: Dict[str, Tuple[int, int]],
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert token spans into fixed-length binary masks aligned with responses.
        """
        masks: Dict[str, torch.Tensor] = {}
        for section in KERNEL_SECTION_NAMES:
            mask = torch.zeros(response_length, dtype=torch.int64, device=device)
            span = token_spans.get(section)
            if span is not None:
                start, end = span
                start = max(start, 0)
                end = min(end, response_length)
                if start < end:
                    mask[start:end] = 1
            masks[section] = mask
        return masks

    def _compute_token_spans(
        self,
        tokenizer,
        response_text: str,
        char_spans: Dict[str, Tuple[int, int]],
    ) -> Dict[str, Tuple[int, int]]:
        """
        Map section character spans to token spans.

        Fast tokenizers provide ``offset_mapping`` directly. When the tokenizer
        does not support offsets we fall back to a prefix-tokenization strategy.
        The fallback is less exact around token-boundary merges but is still
        sufficient for a phase-1 pipeline bring-up.
        """
        if tokenizer is None:
            return {}

        try:
            tokenized = tokenizer(
                response_text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            offsets = tokenized["offset_mapping"]
            spans: Dict[str, Tuple[int, int]] = {}
            for section, (char_start, char_end) in char_spans.items():
                token_start = None
                token_end = None
                for token_index, (tok_start, tok_end) in enumerate(offsets):
                    if tok_end <= char_start:
                        continue
                    if token_start is None and tok_start < char_end and tok_end > char_start:
                        token_start = token_index
                    if token_start is not None and tok_start < char_end:
                        token_end = token_index + 1
                if token_start is not None and token_end is not None:
                    spans[section] = (token_start, token_end)
            return spans
        except Exception:
            spans = {}
            for section, (char_start, char_end) in char_spans.items():
                prefix_ids = tokenizer(response_text[:char_start], add_special_tokens=False)["input_ids"]
                section_prefix_ids = tokenizer(response_text[:char_end], add_special_tokens=False)["input_ids"]
                spans[section] = (len(prefix_ids), len(section_prefix_ids))
            return spans
