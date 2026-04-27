from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from itertools import cycle
from typing import Any, Dict, Iterable, List, Optional

import ray

from .schema import KernelScoreResult


@dataclass
class KernelScoreRequest:
    """
    One scorer request sent from rollout to a separate scoring actor.

    The request keeps both tree metadata and code payload so that future scoring
    implementations can be fully decoupled from the rollout process.
    """

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


@ray.remote
class KernelScoringWorker:
    """
    Dedicated scorer actor managed by Ray.

    Phase 1 behavior is intentionally minimal: it logs requests and returns a
    standardized placeholder result. The subprocess helper is already included
    so that later phases can evaluate kernels without letting compiler crashes
    kill the Ray actor itself.
    """

    def __init__(self, log_dir: str, mode: str = "log_only", timeout_s: int = 300):
        self.log_dir = os.path.abspath(os.path.expanduser(log_dir))
        self.mode = mode
        self.timeout_s = timeout_s
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, "kernel_score_requests.jsonl")

    def score(self, request: KernelScoreRequest) -> KernelScoreResult:
        """Score or log one node, depending on the current phase mode."""
        request_dict = asdict(request)
        request_dict["received_at"] = time.time()
        self._append_jsonl(self.log_path, request_dict)

        if self.mode == "log_only":
            return KernelScoreResult(
                status="logged",
                feedback_text="Phase1 scorer only logged this node. No benchmark has been executed yet.",
                scalar_rewards={
                    "design": 0.0,
                    "code": 0.0,
                    "predict": 0.0,
                },
                metrics={
                    "mode": self.mode,
                    "log_path": self.log_path,
                },
            )

        raise NotImplementedError(f"Unknown scorer mode: {self.mode}")

    def _append_jsonl(self, path: str, payload: Dict[str, Any]) -> None:
        """Append one JSON record so rollout and scoring stay inspectable."""
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _run_subprocess(self, command: List[str], cwd: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a benchmark subprocess in a crash-isolated way.

        This helper is not wired into phase-1 scoring yet, but the actor keeps
        it here so that later phases can collect compile failures, timeouts,
        runtime errors, and stderr without crashing the actor process itself.
        """
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


class KernelScoringPool:
    """
    Thin async-friendly scorer pool wrapper around multiple Ray actors.

    Phase 1 uses a synchronous ``score_many`` implementation for simplicity,
    but requests are already routed to separate actors so rollout and scoring
    are decoupled at the process boundary.
    """

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
        mode: str = "log_only",
        timeout_s: int = 300,
    ) -> "KernelScoringPool":
        actors = [
            KernelScoringWorker.options(num_cpus=1).remote(
                log_dir=log_dir,
                mode=mode,
                timeout_s=timeout_s,
            )
            for _ in range(num_workers)
        ]
        return cls(actors=actors)

    def submit_many(self, requests: Iterable[KernelScoreRequest]) -> List[ray.ObjectRef]:
        """Submit a batch of requests in round-robin order."""
        refs: List[ray.ObjectRef] = []
        for request in requests:
            actor = next(self._actor_cycle)
            refs.append(actor.score.remote(request))
        return refs

    def gather(self, refs: List[ray.ObjectRef]) -> List[KernelScoreResult]:
        """Synchronously collect scorer results."""
        return ray.get(refs)

    def score_many(self, requests: Iterable[KernelScoreRequest]) -> List[KernelScoreResult]:
        """Convenience wrapper used by rollout code."""
        refs = self.submit_many(requests)
        return self.gather(refs)
