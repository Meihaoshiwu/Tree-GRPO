"""
CPU-only smoke test for parser + exporter + prompt builder.

Demonstrates:
1. Root prompt construction from task_spec + reference_python
2. Multi-level prompt accumulation via KernelPromptBuilder.build_child_prompt()
3. Binary tree structure: root -> 2 children -> 4 grandchildren (depth=2)
4. Parser fallback behaviour (all-zero section masks without tokenizer → code fallback)
5. Node-level export: only non-root nodes become training samples (6 samples expected)
6. Output written to log file instead of stdout

Usage:
    python scripts/kernel_rl/smoke_parser_exporter.py
    # then: cat scripts/kernel_rl/smoke_parser_exporter.log
"""

import sys
from pathlib import Path

import torch

from search_r1.kernel_rl.exporter import KernelTrainSampleExporter
from search_r1.kernel_rl.parser import KernelOutputParser
from search_r1.kernel_rl.prompt_builder import KernelPromptBuilder
from search_r1.kernel_rl.schema import KernelScoreResult, KernelTreeNode

LOG_PATH = Path(__file__).with_suffix(".log")


def log(msg: str = "") -> None:
    """Append one line to the log file and also print it."""
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)


def make_node(
    tree_uid: str,
    node_uid: str,
    parent_uid: str | None,
    depth: int,
    prompt_text: str,
    response_text: str,
    response_ids: list[int],
    response_mask: list[int],
    env_feedback: str,
    parsed_response,
    scalar_rewards: dict[str, float],
    *,
    prompt_len: int = 12,
    response_len: int = 9,
) -> KernelTreeNode:
    """Create a node with fixed-size mock tensors so all samples stack cleanly."""
    assert len(response_ids) == response_len
    assert len(response_mask) == response_len

    prompt_ids = torch.arange(10, 10 + prompt_len, dtype=torch.long)
    prompt_attention_mask = torch.ones(prompt_len)
    prompt_position_ids = torch.arange(prompt_len)

    node = KernelTreeNode(
        tree_uid=tree_uid,
        node_uid=node_uid,
        parent_uid=parent_uid,
        depth=depth,
        prompt_text=prompt_text,
        prompt_ids=prompt_ids,
        prompt_attention_mask=prompt_attention_mask,
        prompt_position_ids=prompt_position_ids,
        response_text=response_text,
        response_ids=torch.tensor(response_ids, dtype=torch.long),
        response_attention_mask=torch.tensor(response_mask),
        parsed_response=parsed_response,
        env_feedback_text=env_feedback,
        score_result=KernelScoreResult(
            status="logged",
            feedback_text=env_feedback,
            scalar_rewards=scalar_rewards,
        ),
        scalar_design_reward=scalar_rewards.get("design", 0.0),
        scalar_code_reward=scalar_rewards.get("code", 0.0),
        scalar_predict_reward=scalar_rewards.get("predict", 0.0),
    )
    return node


def main():
    # Clear previous log
    LOG_PATH.write_text("", encoding="utf-8")

    parser = KernelOutputParser()
    exporter = KernelTrainSampleExporter(parser=parser)
    builder = KernelPromptBuilder()

    # ── root prompt ───────────────────────────────────────────────────
    task_spec = {"name": "vector_add", "input_shape": [1024], "dtype": "float32"}
    reference_python = "def vector_add(a, b):\n    return a + b"
    root_prompt = builder.build_root_prompt(task_spec, reference_python)

    log("=" * 70)
    log("ROOT PROMPT (depth=0)")
    log("=" * 70)
    log(root_prompt)

    root = KernelTreeNode(
        tree_uid="tree-0",
        node_uid="root",
        parent_uid=None,
        depth=0,
        prompt_text=root_prompt,
        prompt_ids=torch.arange(10, 22, dtype=torch.long),
        prompt_attention_mask=torch.ones(12),
        prompt_position_ids=torch.arange(12),
        status="root",
    )

    # ── helpers ───────────────────────────────────────────────────────
    def build_tree_branch(
        parent: KernelTreeNode,
        parent_prompt: str,
        child_id: str,
        response_text: str,
        response_ids: list[int],
        response_mask: list[int],
        env_feedback: str,
        scalar_rewards: dict[str, float],
    ) -> KernelTreeNode:
        """Build one child node with prompt accumulation, link to parent, return it."""
        new_depth = parent.depth + 1
        child_prompt = builder.build_child_prompt(
            parent_prompt_text=parent_prompt,
            parent_response_text=response_text,
            env_feedback_text=env_feedback,
            depth=new_depth,
        )
        parsed = parser.parse(response_text)
        child = make_node(
            tree_uid="tree-0",
            node_uid=child_id,
            parent_uid=parent.node_uid,
            depth=new_depth,
            prompt_text=child_prompt,
            response_text=response_text,
            response_ids=response_ids,
            response_mask=response_mask,
            env_feedback=env_feedback,
            parsed_response=parsed,
            scalar_rewards=scalar_rewards,
        )
        parent.add_child(child)
        return child, child_prompt

    # ── depth-1: two children ─────────────────────────────────────────
    RESP_LEN = 9  # all responses same length for torch.stack

    child0_text = (
        "<design>tile the x dimension to exploit memory coalescing</design>"
        "<code>def kernel_v1(x, y):\n    tl.store(y, tl.load(x))</code>"
        "<predict>~1.3x speedup over baseline</predict>"
    )
    child0_feedback = "Benchmark: compile error — 'return' outside function"
    child0_ids = [31, 32, 33, 34, 35, 36, 37, 0, 0]
    child0_mask = [1, 1, 1, 1, 1, 1, 1, 0, 0]
    child0_rewards = {"design": 0.1, "code": 0.2, "predict": 0.3}

    child1_text = (
        "<design>use vectorized loads with mask for boundary handling</design>"
        "<code>def kernel_alt(x, y, N):\n    offsets = tl.arange(0, 256)\n    mask = offsets < N\n    tl.store(y + offsets, tl.load(x + offsets, mask=mask))</code>"
        "<predict>~1.5x speedup, better boundary safety</predict>"
    )
    child1_feedback = "Benchmark PASS: 1.4x speedup, output correct"
    child1_ids = [41, 42, 43, 44, 45, 46, 47, 0, 0]
    child1_mask = [1, 1, 1, 1, 1, 1, 1, 0, 0]
    child1_rewards = {"design": 0.4, "code": 0.7, "predict": 0.5}

    (child0, child0_prompt) = build_tree_branch(
        root, root_prompt, "child-0", child0_text, child0_ids, child0_mask, child0_feedback, child0_rewards
    )
    (child1, child1_prompt) = build_tree_branch(
        root, root_prompt, "child-1", child1_text, child1_ids, child1_mask, child1_feedback, child1_rewards
    )

    log("\n" + "=" * 70)
    log("CHILD-0 PROMPT (depth=1)")
    log("=" * 70)
    log(child0_prompt)

    log("\n" + "=" * 70)
    log("CHILD-1 PROMPT (depth=1)")
    log("=" * 70)
    log(child1_prompt)

    # ── depth-2: four grandchildren ───────────────────────────────────
    gc_texts = [
        (
            "<design>add shared memory tiling to child-0 v1</design>"
            "<code>def kernel_v2(x, y):\n    tile = tl.zeros((256,), dtype=tl.float32)\n    tl.store(y, tile)</code>"
            "<predict>~2.1x speedup</predict>"
        ),
        (
            "<design>fix child-0 v1: precompute offsets, unroll inner loop</design>"
            "<code>def kernel_v3(x, y):\n    for i in range(0, N, 256):\n        tl.store(y + i, tl.load(x + i))</code>"
            "<predict>~1.8x speedup</predict>"
        ),
        (
            "<design>child-1 alt: fuse mask computation into pointer arithmetic</design>"
            "<code>def kernel_alt_v2(x, y, N):\n    pid = tl.program_id(0)\n    block_start = pid * 256\n    tl.store(y + block_start, tl.load(x + block_start))</code>"
            "<predict>~2.3x speedup</predict>"
        ),
        (
            "<design>child-1 alt: double-buffer shared memory for overlapped loads</design>"
            "<code>def kernel_alt_v3(x, y, N):\n    buf = tl.zeros((512,), dtype=tl.float32)\n    buf[:256] = tl.load(x + 0*256)\n    buf[256:] = tl.load(x + 1*256)\n    tl.store(y + 0*256, buf[:256])\n    tl.store(y + 1*256, buf[256:])</code>"
            "<predict>~2.8x speedup with double buffering</predict>"
        ),
    ]
    gc_feedbacks = [
        "Benchmark: compile error — 'tile' declared but shape mismatch",
        "Benchmark PASS: 1.75x speedup, correct results",
        "Benchmark FAIL: out-of-bounds access at N=1023, mask needed",
        "Benchmark PASS: 2.6x speedup, 5% numerical deviation in last element",
    ]
    gc_rewards_list = [
        {"design": 0.3, "code": 0.1, "predict": 0.2},
        {"design": 0.5, "code": 0.7, "predict": 0.4},
        {"design": 0.6, "code": 0.1, "predict": 0.5},
        {"design": 0.8, "code": 0.75, "predict": 0.7},
    ]
    gc_parents = [(child0, child0_prompt, "child-0"), (child0, child0_prompt, "child-0"),
                  (child1, child1_prompt, "child-1"), (child1, child1_prompt, "child-1")]
    gc_ids_template = [
        [51, 52, 53, 54, 55, 56, 57, 0, 0],
        [61, 62, 63, 64, 65, 66, 67, 0, 0],
        [71, 72, 73, 74, 75, 76, 77, 0, 0],
        [81, 82, 83, 84, 85, 86, 87, 0, 0],
    ]
    gc_mask = [1, 1, 1, 1, 1, 1, 1, 0, 0]

    for i in range(4):
        parent, parent_prompt, parent_name = gc_parents[i]
        gc_id = f"{parent_name}-{i % 2}"
        gc_node, gc_prompt = build_tree_branch(
            parent, parent_prompt, gc_id,
            gc_texts[i], gc_ids_template[i], gc_mask,
            gc_feedbacks[i], gc_rewards_list[i],
        )
        log("\n" + "=" * 70)
        log(f"GRANDCHILD {gc_id} PROMPT (depth=2, parent={parent_name})")
        log("=" * 70)
        log(gc_prompt)

    # ── export ────────────────────────────────────────────────────────
    batch = exporter.export_nodes([root])

    log("\n" + "=" * 70)
    log("EXPORTED PPO BATCH")
    log("=" * 70)
    log(f"total nodes in tree : 7  (1 root + 2 children + 4 grandchildren)")
    log(f"exported samples     : {batch.batch['responses'].shape[0]}  (expect 6: all non-root nodes)")

    for i in range(batch.batch["responses"].shape[0]):
        uid = batch.non_tensor_batch["node_uid"][i]
        depth = batch.non_tensor_batch["depth"][i]
        parent = batch.non_tensor_batch["parent_uid"][i]
        tree = batch.non_tensor_batch["tree_uid"][i]
        loss_mask = batch.batch["loss_mask"][i].tolist()
        scores = batch.batch["token_level_scores"][i].tolist()
        d_mask = batch.batch["design_mask"][i].tolist()
        c_mask = batch.batch["code_mask"][i].tolist()
        p_mask = batch.batch["predict_mask"][i].tolist()

        log(f"\n--- sample {i}: {uid}  tree={tree}  depth={depth}  parent={parent} ---")
        log(f"  response length           : {len(loss_mask)}")
        log(f"  loss_mask                  : {loss_mask}")
        log(f"  token_level_scores         : {[round(v, 2) for v in scores]}")
        log(f"  design_mask                : {d_mask}")
        log(f"  code_mask                  : {c_mask}")
        log(f"  predict_mask               : {p_mask}")

        has_design = any(v > 0 for v in d_mask)
        has_code = any(v > 0 for v in c_mask)
        has_predict = any(v > 0 for v in p_mask)
        if has_design and has_code and has_predict:
            log("  [OK] three-section masks active with tokenizer")
        elif has_design or has_code or has_predict:
            tag = " + ".join(
                t for t, m in [("design", has_design), ("code", has_code), ("predict", has_predict)] if m
            )
            log(f"  [OK] section masks: {tag}")
        else:
            log("  [NOTE] fallback: all active tokens → code section (no tokenizer in smoke test)")

    log(f"\nLog written to: {LOG_PATH}")


if __name__ == "__main__":
    main()
