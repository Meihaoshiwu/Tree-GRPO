"""
CPU-only smoke test for the log-only scorer actor.

Verifies:
- KernelScoringWorker as an independent Ray actor process
- Actor PID and worker ID are logged
- Request metadata + code_text are captured in JSONL
- Scorer runs decoupled from main process

Usage:
    python scripts/kernel_rl/smoke_scorer.py
    # then: cat scripts/kernel_rl/smoke_scorer.log
    #       cat ./tmp_kernel_scorer_logs/kernel_score_requests.jsonl  (from repo root)
"""

import os
from pathlib import Path

import ray

from search_r1.kernel_rl.scorer import KernelScoreRequest, KernelScoringPool

LOG_PATH = Path(__file__).with_suffix(".log")
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCORER_LOG_DIR = REPO_ROOT / "tmp_kernel_scorer_logs"


def log(msg: str = "") -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)


def main():
    LOG_PATH.write_text("", encoding="utf-8")

    log("=" * 60)
    log("ENVIRONMENT")
    log("=" * 60)
    log(f"  main PID      : {os.getpid()}")
    log(f"  repo root     : {REPO_ROOT}")
    log(f"  scorer log dir: {SCORER_LOG_DIR}")
    log()

    if not ray.is_initialized():
        ray.init(num_cpus=2, ignore_reinit_error=True)

    log("=" * 60)
    log("CREATING SCORER POOL")
    log("=" * 60)
    pool = KernelScoringPool.create(
        num_workers=1,
        log_dir=str(SCORER_LOG_DIR),
        mode="log_only",
        timeout_s=30,
    )
    log(f"  pool created with {len(pool.actors)} actor(s)")

    # Show Ray actors BEFORE scoring
    log("\n--- Ray actors (before scoring) ---")
    for actor_info in ray._private.state.actors().values():
        log(f"  ActorID={actor_info['ActorID'][:20]}...  State={actor_info['State']}  "
            f"Name={actor_info.get('Name', '(unnamed)')}  "
            f"PID={actor_info.get('Pid', 'N/A')}")
    log()

    # ── send a request with rich metadata ─────────────────────────────
    request = KernelScoreRequest(
        tree_uid="tree-0",
        node_uid="node-0",
        parent_uid="root",
        depth=1,
        task_spec={"name": "vector_add", "input_shape": [1024], "dtype": "float32"},
        bench_spec={"metric": "runtime", "target_speedup": 1.5},
        prompt_text="(root prompt: improve vector_add Triton kernel)",
        response_text="<design>tile x dimension</design><code>def kernel(): pass</code><predict>~1.3x</predict>",
        code_text="def kernel(): pass",
        design_text="tile x dimension",
        predict_text="~1.3x",
        metadata={"submitted_by_pid": os.getpid()},
    )

    result = pool.score_many([request])[0]

    # Show Ray actors AFTER scoring
    log("--- Ray actors (after scoring) ---")
    for actor_info in ray._private.state.actors().values():
        log(f"  ActorID={actor_info['ActorID'][:20]}...  State={actor_info['State']}  "
            f"Name={actor_info.get('Name', '(unnamed)')}  "
            f"PID={actor_info.get('Pid', 'N/A')}")
    log()

    log("=" * 60)
    log("SCORER RESULT")
    log("=" * 60)
    log(f"  status        : {result.status}")
    log(f"  feedback      : {result.feedback_text}")
    log(f"  design_reward : {result.scalar_rewards.get('design', 'N/A')}")
    log(f"  code_reward   : {result.scalar_rewards.get('code', 'N/A')}")
    log(f"  predict_reward: {result.scalar_rewards.get('predict', 'N/A')}")
    log(f"  metrics       : {result.metrics}")
    log()

    log("=" * 60)
    log("SCORER REQUEST (sent to actor)")
    log("=" * 60)
    log(f"  tree_uid    : {request.tree_uid}")
    log(f"  node_uid    : {request.node_uid}")
    log(f"  parent_uid  : {request.parent_uid}")
    log(f"  depth       : {request.depth}")
    log(f"  task_spec   : {request.task_spec}")
    log(f"  bench_spec  : {request.bench_spec}")
    log(f"  code_text   :")
    log(f"    {request.code_text}")
    log(f"  design_text : {request.design_text}")
    log(f"  predict_text: {request.predict_text}")
    log()

    # Check the JSONL log
    jsonl_path = SCORER_LOG_DIR / "kernel_score_requests.jsonl"
    if jsonl_path.exists():
        lines = jsonl_path.read_text(encoding="utf-8").strip().split("\n")
        log(f"JSONL log: {jsonl_path}")
        log(f"  records : {len(lines)}")
        log(f"  size    : {jsonl_path.stat().st_size} bytes")
    else:
        log(f"WARNING: JSONL log not found at {jsonl_path}")

    log(f"\nLog written to: {LOG_PATH}")

    ray.shutdown()


if __name__ == "__main__":
    main()
