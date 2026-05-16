from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from itertools import cycle
from typing import Any, Dict, Iterable, List, Optional

import ray

from .schema import KernelScoreResult


def _short(uid: str) -> str:
    return uid[:8] if len(uid) > 8 else uid


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class KernelScoreRequest:
    tree_uid: str
    node_uid: str
    parent_uid: str
    depth: int
    task_spec: Dict[str, Any] = field(default_factory=dict)
    bench_spec: Dict[str, Any] = field(default_factory=dict)
    prompt_text: str = ""
    response_text: str = ""
    code_text: str = ""
    design_text: str = ""
    predict_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reference-code builder
# ---------------------------------------------------------------------------

def _build_ref_code(task_spec: Dict[str, Any], bench_spec: Dict[str, Any]) -> str:
    """Build a ``simple``-format ref_code string from task/bench specs.

    Produces::

        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import triton
        import triton.language as tl

        def ref_fn(...): ...
        def gen_inputs(): return [<bench_spec.input_gen>]
    """
    ref_python = task_spec.get("reference_python", "") or task_spec.get("reference_code", "")
    input_gen = bench_spec.get("input_gen", "torch.randn(4096)")

    lines = [
        "import torch",
        "import torch.nn as nn",
        "import torch.nn.functional as F",
        "import triton",
        "import triton.language as tl",
        "import math",
        "",
    ]
    if ref_python.strip():
        lines.append(ref_python.strip())
        lines.append("")

    # Wrap the input_gen expression into a gen_inputs function
    lines.append("def gen_inputs():")
    lines.append(f"    return [{input_gen}]")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scorer actor
# ---------------------------------------------------------------------------

@ray.remote
class KernelScoringWorker:
    """Ray actor that evaluates generated Triton kernels.

    Each ``score()`` call runs the full compile → correctness → performance
    pipeline inside a subprocess so that compiler crashes never kill the actor.
    """

    def __init__(
        self,
        log_dir: str,
        mode: str = "eval",
        timeout_s: int = 300,
        device_id: int = 0,
    ):
        self.log_dir = os.path.abspath(os.path.expanduser(log_dir))
        self.mode = mode
        self.timeout_s = timeout_s
        self.device_id = device_id
        os.makedirs(self.log_dir, exist_ok=True)

        # One log file per worker (per run)
        self.log_path = os.path.join(
            self.log_dir,
            f"scorer_pid{os.getpid()}_{_now_tag()}.jsonl",
        )

        # Lazy-init the evaluator (imported on first use to avoid blocking actor init)
        self._evaluator = None

    @property
    def evaluator(self):
        if self._evaluator is None:
            from .scoring.evaluator import KernelEvaluator, EvalConfig

            self._evaluator = KernelEvaluator(
                EvalConfig(
                    compile_timeout_s=self.timeout_s,
                    verbose=False,
                )
            )
        return self._evaluator

    # ------------------------------------------------------------------
    def score(self, request: KernelScoreRequest) -> KernelScoreResult:
        """Evaluate one node's generated kernel code."""
        if self.mode == "log_only":
            result = KernelScoreResult(
                status="logged",
                feedback_text="[Phase 1] log_only mode — no benchmark executed.",
                scalar_rewards={"design": 0.0, "code": 0.0, "predict": 0.0},
                metrics={"mode": self.mode},
            )
            self._log_result(request, result)
            return result

        if self.mode == "eval":
            return self._score_eval(request)

        raise NotImplementedError(f"Unknown scorer mode: {self.mode}")

    # ------------------------------------------------------------------
    def _score_eval(self, request: KernelScoreRequest) -> KernelScoreResult:
        """Real evaluation: compile + correctness + performance."""
        code_text = request.code_text or ""
        if not code_text.strip():
            result = KernelScoreResult(
                status="empty_code",
                feedback_text="No code found in the response.",
                scalar_rewards={"design": 0.0, "code": 0.0, "predict": 0.0},
                metrics={"error": "empty_code"},
            )
            self._log_result(request, result)
            return result

        ref_code = _build_ref_code(request.task_spec, request.bench_spec)
        target_speedup = request.bench_spec.get("target_speedup", 2.0)

        try:
            eval_result = self.evaluator.evaluate(
                ref_code=ref_code,
                new_code=code_text,
                problem_format="simple",
                device_id=self.device_id,
            )
        except Exception as exc:
            result = KernelScoreResult(
                status="eval_error",
                feedback_text=f"Evaluator crashed: {exc}",
                scalar_rewards={"design": 0.0, "code": 0.0, "predict": 0.0},
                metrics={"eval_exception": str(exc)},
            )
            self._log_result(request, result)
            return result

        rewards = self.evaluator.compute_rewards(eval_result, target_speedup=target_speedup)
        feedback = self.evaluator.build_feedback(eval_result)

        metrics = {
            "mode": self.mode,
            "log_path": self.log_path,
            "compiled": eval_result.compiled,
            "correctness": eval_result.correctness,
            "speedup": eval_result.speedup,
            "runtime_ms": eval_result.runtime_ms,
            "ref_runtime_ms": eval_result.ref_runtime_ms,
            "num_passed_trials": eval_result.num_passed_trials,
            "num_correct_trials": eval_result.num_correct_trials,
            "error_type": eval_result.error_type,
            **eval_result.metadata,
        }

        result = KernelScoreResult(
            status=eval_result.status,
            feedback_text=feedback,
            scalar_rewards=rewards,
            metrics=metrics,
        )
        self._log_result(request, result)
        return result

    # ------------------------------------------------------------------
    def _log_result(
        self, request: KernelScoreRequest, result: KernelScoreResult
    ) -> None:
        """Write execution-only result — one JSON line per scored node."""
        record = {
            "ts": datetime.now().isoformat(),
            "worker_pid": os.getpid(),
            "node_uid": request.node_uid,
            "tree_uid": request.tree_uid,
            "parent_uid": request.parent_uid,
            "depth": request.depth,
            "status": result.status,
            "rewards": result.scalar_rewards,
            "feedback": result.feedback_text,
            "metrics": {k: v for k, v in result.metrics.items()
                        if k not in ("log_path", "mode")},
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    def _run_subprocess(
        self, command: List[str], cwd: Optional[str] = None
    ) -> Dict[str, Any]:
        started = time.time()
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                timeout=self.timeout_s,
                check=False,
                capture_output=True,
                text=True,
            )
            return {
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "elapsed_s": time.time() - started,
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "returncode": -1,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
                "elapsed_s": time.time() - started,
                "error_type": "timeout",
            }
        except Exception as exc:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": str(exc),
                "elapsed_s": time.time() - started,
                "error_type": type(exc).__name__,
            }


# ---------------------------------------------------------------------------
# Actor pool
# ---------------------------------------------------------------------------

class KernelScoringPool:
    """Round-robin pool of KernelScoringWorker Ray actors."""

    def __init__(self, actors: List[ray.actor.ActorHandle]):
        if not actors:
            raise ValueError("KernelScoringPool requires at least one actor.")
        self.actors = actors
        self._actor_cycle = cycle(self.actors)

    @classmethod
    def create(
        cls,
        num_workers: int,
        log_dir: str,
        mode: str = "eval",
        timeout_s: int = 300,
        device_id: int = 0,
    ) -> "KernelScoringPool":
        actors = [
            KernelScoringWorker.options(num_cpus=1, num_gpus=0).remote(
                log_dir=log_dir,
                mode=mode,
                timeout_s=timeout_s,
                device_id=device_id,
            )
            for _ in range(num_workers)
        ]
        return cls(actors=actors)

    def submit_many(self, requests: Iterable[KernelScoreRequest]) -> List[ray.ObjectRef]:
        refs: List[ray.ObjectRef] = []
        for request in requests:
            actor = next(self._actor_cycle)
            refs.append(actor.score.remote(request))
        return refs

    def gather(self, refs: List[ray.ObjectRef]) -> List[KernelScoreResult]:
        return ray.get(refs)

    def score_many(self, requests: Iterable[KernelScoreRequest]) -> List[KernelScoreResult]:
        refs = self.submit_many(requests)
        return self.gather(refs)
