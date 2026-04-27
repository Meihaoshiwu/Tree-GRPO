"""
CPU-only smoke test for parser + exporter.

Usage:
    python scripts/kernel_rl/smoke_parser_exporter.py
"""

import torch

from search_r1.kernel_rl.exporter import KernelTrainSampleExporter
from search_r1.kernel_rl.parser import KernelOutputParser
from search_r1.kernel_rl.schema import KernelScoreResult, KernelTreeNode


def main():
    parser = KernelOutputParser()
    exporter = KernelTrainSampleExporter(parser=parser)

    root = KernelTreeNode(
        tree_uid="tree-0",
        node_uid="root",
        parent_uid=None,
        depth=0,
        prompt_text="root prompt",
        prompt_ids=torch.tensor([0, 0, 1, 2, 3]),
        prompt_attention_mask=torch.tensor([0, 0, 1, 1, 1]),
        prompt_position_ids=torch.tensor([0, 0, 0, 1, 2]),
        status="root",
    )

    response_text = (
        "<design>tile x dimension</design>"
        "<code>def kernel():\n    pass</code>"
        "<predict>~1.3x speedup</predict>"
    )
    parsed = parser.parse(response_text)
    child = KernelTreeNode(
        tree_uid="tree-0",
        node_uid="child-0",
        parent_uid="root",
        depth=1,
        prompt_text="root prompt",
        prompt_ids=torch.tensor([0, 0, 1, 2, 3]),
        prompt_attention_mask=torch.tensor([0, 0, 1, 1, 1]),
        prompt_position_ids=torch.tensor([0, 0, 0, 1, 2]),
        response_text=response_text,
        response_ids=torch.tensor([11, 12, 13, 14, 15, 16, 17, 0, 0]),
        response_attention_mask=torch.tensor([1, 1, 1, 1, 1, 1, 1, 0, 0]),
        parsed_response=parsed,
        score_result=KernelScoreResult(
            status="logged",
            feedback_text="log only",
            scalar_rewards={"design": 0.1, "code": 0.2, "predict": 0.3},
        ),
        scalar_design_reward=0.1,
        scalar_code_reward=0.2,
        scalar_predict_reward=0.3,
    )
    root.add_child(child)

    batch = exporter.export_nodes([root])
    print("exported batch keys:", list(batch.batch.keys()))
    print("response shape:", tuple(batch.batch["responses"].shape))
    print("loss mask:", batch.batch["loss_mask"][0].tolist())
    print("token scores:", batch.batch["token_level_scores"][0].tolist())


if __name__ == "__main__":
    main()
