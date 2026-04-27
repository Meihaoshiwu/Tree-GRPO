"""
CPU-only smoke test for multi-section advantage preparation.

Usage:
    python scripts/kernel_rl/smoke_advantage.py
"""

import numpy as np
import torch

from tensordict import TensorDict

from search_r1.kernel_rl.advantage import compute_multi_section_advantages
from verl import DataProto


def main():
    batch = TensorDict(
        source={
            "responses": torch.tensor([[11, 12, 13, 0]]),
            "input_ids": torch.tensor([[1, 2, 3, 11, 12, 13, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0]]),
            "position_ids": torch.tensor([[0, 1, 2, 3, 4, 5, 6]]),
            "design_mask": torch.tensor([[1, 0, 0, 0]]),
            "code_mask": torch.tensor([[0, 1, 1, 0]]),
            "predict_mask": torch.tensor([[0, 0, 0, 0]]),
            "design_token_scores": torch.tensor([[0.5, 0.0, 0.0, 0.0]]),
            "code_token_scores": torch.tensor([[0.0, 1.0, 1.0, 0.0]]),
            "predict_token_scores": torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
        },
        batch_size=(1,),
    )
    proto = DataProto(
        batch=batch,
        non_tensor_batch={"uid": np.array(["sample-0"], dtype=object)},
    )
    proto = compute_multi_section_advantages(proto, adv_estimator="no_estimator")
    print("loss mask:", proto.batch["loss_mask"][0].tolist())
    print("advantages:", proto.batch["advantages"][0].tolist())
    print("design advantages:", proto.batch["design_advantages"][0].tolist())
    print("code advantages:", proto.batch["code_advantages"][0].tolist())


if __name__ == "__main__":
    main()
