from __future__ import annotations

import json
from typing import Any, Dict

from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F

from .prompt_builder import KernelPromptBuilder


class KernelRLDataset(RLHFDataset):
    """
    RL dataset variant for kernel-development tasks.

    Unlike the original QA datasets, rows may provide ``task_spec`` and
    ``bench_spec`` separately. The prompt builder decides how much of that
    structured data should be surfaced to the model.
    """

    def __init__(
        self,
        *args,
        prompt_builder: KernelPromptBuilder,
        task_spec_key: str = "task_spec",
        bench_spec_key: str = "bench_spec",
        reference_python_key: str = "reference_python",
        **kwargs,
    ):
        self.prompt_builder = prompt_builder
        self.task_spec_key = task_spec_key
        self.bench_spec_key = bench_spec_key
        self.reference_python_key = reference_python_key
        super().__init__(*args, **kwargs)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """
        Return tokenized prompt tensors plus original task / benchmark metadata.
        """
        row_dict = self.dataframe.iloc[item].to_dict()
        if self.prompt_key in row_dict and "prompt" not in row_dict:
            row_dict["prompt"] = row_dict[self.prompt_key]
        prompt_text = self.prompt_builder.build_prompt_from_row(row_dict)

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_text,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["prompt_text"] = prompt_text
        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]
        row_dict["task_spec"] = _parse_json_field(row_dict.get(self.task_spec_key, {}))
        row_dict["bench_spec"] = _parse_json_field(row_dict.get(self.bench_spec_key, {}))
        row_dict["reference_python"] = row_dict.get(self.reference_python_key, "")
        row_dict["index"] = row_dict.get("extra_info", {}).get("index", row_dict.get("index", item))
        return row_dict


def _parse_json_field(value: Any) -> Dict[str, Any]:
    """Parse a JSON string field from parquet into a dict, or pass through."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {}
    return value if isinstance(value, dict) else {}
