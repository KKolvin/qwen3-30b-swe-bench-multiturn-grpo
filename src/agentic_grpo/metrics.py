"""Per-trajectory metrics + a minimal timer to fill them.

Scope is deliberately narrow: we only collect here what has **no** server-side
ground truth. Everything the inference server already measures — prefill/TTFT,
inter-token/decode latency, end-to-end latency, queue time, throughput, and the
rollout *drain* phase (when the GPU stops being saturated) — is scraped straight
from SGLang's ``/metrics`` in :mod:`agentic_grpo.server_monitor`, not guessed
here. So this module keeps:

* token usage (server-reported via the response ``usage``),
* tool/env execution time (docker — the server never sees it),
* turn / tool-call / edit counts (agent-loop bookkeeping),
* the SWE-bench harness grading outcome.

Per-trajectory generation time is *derived* (``total_trajectory_time -
total_tool_call_time``) rather than separately timed, since the server already
owns the authoritative generation-latency breakdown.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator


@dataclass
class TrajectoryMetrics:
    # --- Token Statistics (server-reported via response usage where noted) ---
    prompt_tokens: int = 0        # peak prompt/context size fed to the model (last, largest turn)
    completion_tokens: int = 0    # generated tokens summed across turns (server usage)
    cached_prompt_tokens: int = 0  # prompt tokens served from prefix cache, summed (server usage)
    total_trajectory_tokens: int = 0
    tool_obs_tokens: list[int] = field(default_factory=list)  # exact tokens per tool observation
    response_tokens: int = 0  # tokens the policy is actually trained on (assistant turns)

    # --- Latency (seconds; only what the server can't tell us) ---
    total_trajectory_time: float = 0.0  # wall time start->terminal message
    total_tool_call_time: float = 0.0   # time spent inside env.execute(...) (docker)

    # --- Action / Event Counts ---
    tool_call_count: int = 0
    edit_count: int = 0       # specific parsing of 'edit' tool calls
    num_turns: int = 0
    truncated: bool = False   # context_length breached -> trajectory cut short

    # --- Artifacts ---
    pytest_output_length: int = 0  # size of the harness test log

    # --- Outcome (from upstream SWE-bench harness report.json) ---
    resolved: bool = False
    reward: float = 0.0
    patch_applied: bool = False   # patch_successfully_applied
    f2p_passed: int = 0           # FAIL_TO_PASS tests that passed
    f2p_total: int = 0
    p2p_passed: int = 0           # PASS_TO_PASS tests that stayed passing
    p2p_total: int = 0
    instance_id: str = ""
    exit_status: str = ""

    def generation_time(self) -> float:
        """Per-trajectory generation wall time, derived (traj - tool).

        The server owns the authoritative prefill/decode split; this coarse
        client-side figure just complements it per trajectory.
        """
        return max(self.total_trajectory_time - self.total_tool_call_time, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def aggregate(batch: list["TrajectoryMetrics"]) -> dict[str, float]:
        """Reduce a group/batch of trajectories to scalar logging metrics.

        Latency here is only the two things with no server source (whole-episode
        wall time and tool/env time) plus derived generation time. The
        prefill/decode/queue/drain breakdown comes from the server monitor.
        """
        if not batch:
            return {}
        n = len(batch)

        def mean(key: str) -> float:
            return sum(getattr(m, key) for m in batch) / n

        flat_obs = [t for m in batch for t in m.tool_obs_tokens]

        def rate(passed: str, total: str) -> float:
            tot = sum(getattr(m, total) for m in batch)
            return (sum(getattr(m, passed) for m in batch) / tot) if tot > 0 else 0.0

        return {
            "traj/resolve_rate": sum(1 for m in batch if m.resolved) / n,
            "traj/mean_reward": mean("reward"),
            "reward/patch_applied_rate": sum(1 for m in batch if m.patch_applied) / n,
            "reward/f2p_pass_rate": rate("f2p_passed", "f2p_total"),
            "reward/p2p_pass_rate": rate("p2p_passed", "p2p_total"),
            "traj/mean_turns": mean("num_turns"),
            "traj/mean_tool_calls": mean("tool_call_count"),
            "traj/mean_edits": mean("edit_count"),
            "traj/truncation_rate": sum(1 for m in batch if m.truncated) / n,
            "tokens/mean_prompt": mean("prompt_tokens"),
            "tokens/mean_completion": mean("completion_tokens"),
            "tokens/mean_cached_prompt": mean("cached_prompt_tokens"),
            "tokens/mean_total": mean("total_trajectory_tokens"),
            "tokens/mean_response": mean("response_tokens"),
            "tokens/mean_obs": (sum(flat_obs) / len(flat_obs)) if flat_obs else 0.0,
            "tokens/max_obs": max(flat_obs) if flat_obs else 0.0,
            "latency/mean_trajectory_s": mean("total_trajectory_time"),
            "latency/max_trajectory_s": max(m.total_trajectory_time for m in batch),
            "latency/mean_tool_s": mean("total_tool_call_time"),
            "latency/mean_generation_s": sum(m.generation_time() for m in batch) / n,
        }


class TrajectoryTimer:
    """Time the two spans the server can't see: the whole episode and env calls.

    Usage::

        timer = TrajectoryTimer(metrics)
        with timer.trajectory():
            for turn in ...:
                ...                        # generation latency -> server /metrics
                with timer.tool_call():    # env.execute (docker)
                    ...
    """

    def __init__(self, metrics: TrajectoryMetrics):
        self.m = metrics

    @contextmanager
    def trajectory(self) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.m.total_trajectory_time = time.perf_counter() - start

    @contextmanager
    def tool_call(self) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.m.total_tool_call_time += time.perf_counter() - start
