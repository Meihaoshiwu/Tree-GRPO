#!/usr/bin/env python3
"""
Progress tracker for multi-step RL training.

Watches the training log and scorer logs, periodically writes:
  - progress.json:   step number, metrics, scorer stats, ETA
  - samples/step_N/:  model output code for inspection

Usage:
  python scripts/kernel_rl/track_progress.py \
      --run-dir /path/to/run_dir \
      --sample-interval 10 \
      --max-samples 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_args():
    parser = argparse.ArgumentParser(description="Track RL training progress")
    parser.add_argument("--run-dir", required=True, help="Training run directory")
    parser.add_argument("--sample-interval", type=int, default=10,
                        help="Extract code samples every N steps (0=never)")
    parser.add_argument("--max-samples", type=int, default=5,
                        help="Max code samples to extract per interval")
    parser.add_argument("--log-interval", type=int, default=1,
                        help="Write progress every N steps")
    return parser.parse_args()


# ── Metrics parsing ──────────────────────────────────────────────────

_STEP_HEADER = re.compile(r"\[########\] kernel epoch (\d+), step (\d+)")
_STEP_LINE = re.compile(
    r"step:(\d+)\s+.*?"
    r"actor/pg_loss:([-\d.]+)\s+.*?"
    r"actor/pg_loss_design:([-\d.]+)\s+.*?"
    r"actor/pg_loss_code:([-\d.]+)\s+.*?"
    r"actor/pg_loss_predict:([-\d.]+)\s+.*?"
    r"actor/entropy_loss:([-\d.]+)"
)
_JSON_METRICS = re.compile(r"(\d+) (\{.*actor/pg_loss.*?\})")


def parse_step_metrics(log_file: str) -> List[Dict[str, Any]]:
    """Parse per-step metrics from training log. Returns list of {step, ...} dicts."""
    steps = []
    if not os.path.exists(log_file):
        return steps

    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Try JSON-format first (veRL internal logger)
    for m in _JSON_METRICS.finditer(content):
        step_num = int(m.group(1))
        try:
            metrics = json.loads(m.group(2))
            steps.append({"step": step_num, **metrics})
        except json.JSONDecodeError:
            pass

    # Fallback to regex on console output
    if not steps:
        for m in _STEP_LINE.finditer(content):
            steps.append({
                "step": int(m.group(1)),
                "pg_loss": float(m.group(2)),
                "pg_loss_design": float(m.group(3)),
                "pg_loss_code": float(m.group(4)),
                "pg_loss_predict": float(m.group(5)),
                "entropy_loss": float(m.group(6)),
            })

    return steps


# ── Scorer stats ─────────────────────────────────────────────────────

def parse_scorer_logs(scorer_dir: str, since_step: int = 0) -> Dict[str, Any]:
    """Aggregate scorer results since a given step."""
    stats = {
        "total_scored": 0,
        "compiled": 0,
        "correct": 0,
        "speedup_sum": 0.0,
        "speedup_count": 0,
        "speedup_max": 0.0,
        "rewards_code": [],
        "rewards_design": [],
        "rewards_predict": [],
        "by_task": defaultdict(lambda: {"compiled": 0, "correct": 0, "total": 0}),
    }

    for log_path in sorted(glob(os.path.join(scorer_dir, "scorer_pid*.jsonl"))):
        try:
            with open(log_path, "r") as f:
                for line in f:
                    record = json.loads(line)
                    stats["total_scored"] += 1
                    m = record.get("metrics", {})
                    if m.get("compiled"):
                        stats["compiled"] += 1
                    if m.get("correctness"):
                        stats["correct"] += 1
                    sp = m.get("speedup", 0)
                    if sp > 0:
                        stats["speedup_sum"] += sp
                        stats["speedup_count"] += 1
                        stats["speedup_max"] = max(stats["speedup_max"], sp)
                    r = record.get("rewards", {})
                    if isinstance(r, dict):
                        for k in ("code", "design", "predict"):
                            if k in r:
                                stats[f"rewards_{k}"].append(r[k])

                    node_uid = record.get("node_uid", "")
                    task_name = node_uid.rsplit("_", 1)[0] if "_" in node_uid else "unknown"
                    stats["by_task"][task_name]["total"] += 1
                    if m.get("compiled"):
                        stats["by_task"][task_name]["compiled"] += 1
                    if m.get("correctness"):
                        stats["by_task"][task_name]["correct"] += 1
        except Exception:
            pass

    return stats


def summarize_scorer(stats: Dict[str, Any]) -> Dict[str, Any]:
    total = max(stats["total_scored"], 1)
    compiled = stats["compiled"]
    correct = stats["correct"]
    sp_count = max(stats["speedup_count"], 1)
    return {
        "compile_rate": round(compiled / total, 4),
        "correctness_rate": round(correct / compiled, 4) if compiled else 0.0,
        "avg_speedup": round(stats["speedup_sum"] / sp_count, 3) if sp_count else 0.0,
        "max_speedup": round(stats["speedup_max"], 3),
        "avg_code_reward": round(_mean(stats["rewards_code"]), 4),
        "avg_design_reward": round(_mean(stats["rewards_design"]), 4),
        "avg_predict_reward": round(_mean(stats["rewards_predict"]), 4),
        "total_scored": total - 1 + 1,  # actual count
    }


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


# ── Code sample extraction ───────────────────────────────────────────

def extract_samples(tree_dir: str, samples_dir: str, step: int, max_samples: int):
    """Extract model-generated code from tree logs for inspection."""
    out_dir = os.path.join(samples_dir, f"step_{step:04d}")
    os.makedirs(out_dir, exist_ok=True)

    count = 0
    for tree_path in sorted(glob(os.path.join(tree_dir, "tree_*"))):
        if count >= max_samples:
            break
        tree_name = os.path.basename(tree_path)
        for node_file in sorted(glob(os.path.join(tree_path, "*.txt"))):
            if count >= max_samples:
                break
            node_name = os.path.splitext(os.path.basename(node_file))[0]

            with open(node_file, "r") as f:
                content = f.read()

            # Extract code section
            code_match = re.search(r"<code>\n?(.*?)\n?</code>", content, re.DOTALL)
            if code_match:
                code_text = code_match.group(1).strip()
                out_path = os.path.join(out_dir, f"{tree_name}_{node_name}.py")
                with open(out_path, "w") as f:
                    f.write(code_text)
                count += 1


# ── Progress writer ──────────────────────────────────────────────────

def write_progress(run_dir: str, info: Dict[str, Any]):
    path = os.path.join(run_dir, "progress.json")
    with open(path, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    run_dir = os.path.abspath(args.run_dir)
    log_file = os.path.join(run_dir, "train.log")
    tree_dir = os.path.join(run_dir, "tree_logs")
    scorer_dir = os.path.join(run_dir, "scorer_logs")
    samples_dir = os.path.join(run_dir, "samples")

    start_time = datetime.now()
    last_step = 0

    print(f"[tracker] Watching: {run_dir}")
    print(f"[tracker]   train log:   {log_file}")
    print(f"[tracker]   tree logs:   {tree_dir}")
    print(f"[tracker]   scorer logs: {scorer_dir}")
    print(f"[tracker]   samples:     {samples_dir}")
    print(f"[tracker]   progress:    {os.path.join(run_dir, 'progress.json')}")
    print()

    while True:
        time.sleep(30)

        # Check if training log exists
        if not os.path.exists(log_file):
            continue

        # Parse steps
        all_steps = parse_step_metrics(log_file)
        if not all_steps:
            continue

        current_step = all_steps[-1]["step"]
        if current_step <= last_step:
            continue
        last_step = current_step

        # Scorer stats
        scorer_stats = parse_scorer_logs(scorer_dir)
        scorer_summary = summarize_scorer(scorer_stats)

        # Latest step metrics
        latest = all_steps[-1]

        # ETA
        elapsed = (datetime.now() - start_time).total_seconds()
        steps_done = current_step
        steps_total = 50  # from config
        if steps_done > 0:
            eta_seconds = elapsed / steps_done * (steps_total - steps_done)
            eta = str(timedelta(seconds=int(eta_seconds)))
        else:
            eta = "unknown"

        progress = {
            "ts": datetime.now().isoformat(),
            "step": current_step,
            "elapsed": str(timedelta(seconds=int(elapsed))),
            "eta": eta,
            "training": {
                "pg_loss": latest.get("pg_loss"),
                "pg_loss_design": latest.get("pg_loss_design"),
                "pg_loss_code": latest.get("pg_loss_code"),
                "pg_loss_predict": latest.get("pg_loss_predict"),
                "entropy_loss": latest.get("entropy_loss"),
            },
            "scorer": scorer_summary,
            "by_task": {
                task: {
                    "compile_rate": round(v["compiled"] / max(v["total"], 1), 3),
                    "correctness_rate": round(v["correct"] / max(v["compiled"], 1), 3) if v["compiled"] else 0.0,
                }
                for task, v in sorted(scorer_stats["by_task"].items())
            },
        }

        write_progress(run_dir, progress)

        # Console summary
        print(f"[step {current_step:3d}] "
              f"pg_loss={latest.get('pg_loss', '?'):.3f}  "
              f"code={latest.get('pg_loss_code', '?'):.3f}  "
              f"design={latest.get('pg_loss_design', '?'):.3f}  "
              f"compile={scorer_summary['compile_rate']:.1%}  "
              f"correct={scorer_summary['correctness_rate']:.1%}  "
              f"speedup_avg={scorer_summary['avg_speedup']:.2f}x  "
              f"ETA={eta}",
              flush=True)

        # Extract code samples
        if args.sample_interval > 0 and current_step % args.sample_interval == 0:
            extract_samples(tree_dir, samples_dir, current_step, args.max_samples)
            print(f"  [samples] extracted to {os.path.join(samples_dir, f'step_{current_step:04d}')}",
                  flush=True)


if __name__ == "__main__":
    main()
