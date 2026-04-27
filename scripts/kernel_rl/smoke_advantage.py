"""
CPU-only smoke test for multi-section advantage preparation.

Usage:
    python scripts/kernel_rl/smoke_advantage.py
    # then: cat scripts/kernel_rl/smoke_advantage.log
"""

from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from search_r1.kernel_rl.advantage import compute_multi_section_advantages
from verl import DataProto

LOG_PATH = Path(__file__).with_suffix(".log")


def log(msg: str = "") -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)


def main():
    LOG_PATH.write_text("", encoding="utf-8")

    # response tokens: 3 valid + 1 padding = 4
    # token0 → design, token1-2 → code, token3 → padding
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

    log("=" * 60)
    log("INPUT: manually constructed DataProto")
    log("=" * 60)
    log("  response tokens (3 valid + 1 pad): [11, 12, 13, 0]")
    log("  design_mask : [1, 0, 0, 0]  → token0 is design")
    log("  code_mask   : [0, 1, 1, 0]  → token1-2 are code")
    log("  predict_mask: [0, 0, 0, 0]  → no predict tokens")
    log("  design_token_scores : [0.5, 0, 0, 0]")
    log("  code_token_scores   : [0, 1.0, 1.0, 0]")
    log("  predict_token_scores: [0, 0, 0, 0]")
    log()

    proto = compute_multi_section_advantages(proto, adv_estimator="no_estimator")

    log("=" * 60)
    log("OUTPUT: after compute_multi_section_advantages (no_estimator)")
    log("=" * 60)
    log(f"  loss_mask           : {proto.batch['loss_mask'][0].tolist()}")
    log(f"  advantages (merged) : {[round(v, 4) for v in proto.batch['advantages'][0].tolist()]}")
    log(f"  design_advantages   : {[round(v, 4) for v in proto.batch['design_advantages'][0].tolist()]}")
    log(f"  code_advantages     : {[round(v, 4) for v in proto.batch['code_advantages'][0].tolist()]}")
    log(f"  predict_advantages  : {[round(v, 4) for v in proto.batch['predict_advantages'][0].tolist()]}")
    log(f"  token_level_scores  : {[round(v, 4) for v in proto.batch['token_level_scores'][0].tolist()]}")
    log()

    # Verify expectations
    log("=" * 60)
    log("VERIFICATION (no_estimator mode: scores = advantages directly)")
    log("=" * 60)

    checks = []
    checks.append(("loss_mask = [1,1,1,0] (OR of design+code+predict, masked by response_attn)",
                    proto.batch["loss_mask"][0].tolist() == [1, 1, 1, 0]))
    checks.append(("design_advantages = [0.5, 0, 0, 0]",
                    proto.batch["design_advantages"][0].tolist() == [0.5, 0.0, 0.0, 0.0]))
    checks.append(("code_advantages = [0, 1.0, 1.0, 0]",
                    proto.batch["code_advantages"][0].tolist() == [0.0, 1.0, 1.0, 0.0]))
    checks.append(("predict_advantages = [0, 0, 0, 0]",
                    proto.batch["predict_advantages"][0].tolist() == [0.0, 0.0, 0.0, 0.0]))

    all_ok = True
    for desc, ok in checks:
        status = "PASS" if ok else "FAIL"
        log(f"  [{status}] {desc}")
        if not ok:
            all_ok = False

    log()
    if all_ok:
        log("ALL CHECKS PASSED")
    else:
        log("SOME CHECKS FAILED — review above")

    log(f"\nLog written to: {LOG_PATH}")


if __name__ == "__main__":
    main()
