"""
CPU-only smoke test for the log-only scorer actor.

Usage:
    python scripts/kernel_rl/smoke_scorer.py
"""

import ray

from search_r1.kernel_rl.scorer import KernelScoreRequest, KernelScoringPool


def main():
    if not ray.is_initialized():
        ray.init(num_cpus=2, ignore_reinit_error=True)

    pool = KernelScoringPool.create(
        num_workers=1,
        log_dir="./tmp_kernel_scorer_logs",
        mode="log_only",
        timeout_s=30,
    )
    result = pool.score_many(
        [
            KernelScoreRequest(
                tree_uid="tree-0",
                node_uid="node-0",
                parent_uid="root",
                depth=1,
                task_spec={"name": "vector add"},
                bench_spec={"metric": "runtime"},
                prompt_text="prompt",
                response_text="<design>x</design><code>y</code><predict>z</predict>",
                code_text="def kernel(): pass",
            )
        ]
    )[0]
    print("status:", result.status)
    print("feedback:", result.feedback_text)
    print("metrics:", result.metrics)


if __name__ == "__main__":
    main()
