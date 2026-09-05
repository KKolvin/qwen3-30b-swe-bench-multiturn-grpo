"""Per-trajectory metrics + a minimal timer to fill them.

Scope is deliberately narrow: we only collect here what has **no** server-side
ground truth. Aggregate server behaviour — throughput, KV usage, and the rollout
*drain* phase (when the GPU stops being saturated) — is scraped straight from
SGLang's ``/metrics`` in :mod:`agentic_grpo.server_monitor`, not guessed here. So
this module keeps:

* token usage (server-reported via the response ``usage``),
* tool/env execution time (docker — the server never sees it),
* turn / tool-call / edit counts (agent-loop bookkeeping),
* the SWE-bench harness grading outcome,
* the **per-turn timeline** (:class:`TurnTiming`) that lines an episode up
  against wall clock.

``/metrics`` histograms are aggregate, so they cannot say *which* trajectory
straggled. :class:`TurnTiming` closes that gap: each generate call carries the
server's own per-request timestamps (queue admission, prefill finish / decode
start, decode finish) alongside the client-side call boundaries, so a trajectory
can be replayed on a timeline. See :mod:`agentic_grpo.sglang_timing` for how
those server timestamps are plumbed out of SGLang.

All timestamps are UNIX wall clock (``time.time()``) so they are comparable
across processes — the agent loop runs in N ``AgentLoopWorker`` actors while the
timestamps originate in the rollout-server actor. ``time.perf_counter()`` is
per-process and would be meaningless across that boundary; it is still used for
*durations* within one process.
"""

from __future__ import annotations

import logging
import os
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

logger = logging.getLogger("agentic_grpo.metrics")


@dataclass
class TurnTiming:
    """Timeline of one assistant turn: one generate call plus its tool calls.

    Two clocks are recorded for the generate call and it matters which you read:

    * ``gen_call_start`` / ``gen_call_end`` — client side, in the agent-loop
      worker. Includes the Ray round trip and the load balancer, so it brackets
      but overstates the server's work.
    * ``request_received`` / ``decode_start`` / ``decode_finished`` — the
      server's own timestamps for the same request. ``decode_start`` is the
      instant the first sampled token reached SGLang's tokenizer manager, i.e.
      **prefill finished and decoding began**. These are ``0.0`` when the server
      did not report them (see ``timing_source`` on the trajectory).
    """

    turn: int = 0

    # --- generate call, client side (AgentLoopWorker process) ---
    gen_call_start: float = 0.0
    gen_call_end: float = 0.0

    # --- generate call, server side (SGLang tokenizer manager) ---
    request_received: float = 0.0    # request admitted, before scheduling
    request_scheduled: float = 0.0   # handed to the scheduler (queue wait ends)
    decode_start: float = 0.0        # prefill done / first token sampled
    decode_finished: float = 0.0     # last token sampled
    response_sent: float = 0.0       # response handed back to the client
    completion_tokens: int = 0       # tokens the server says it generated
    cached_tokens: int = 0           # prompt tokens served from the prefix cache
    prompt_tokens: int = 0           # this turn's full context size (grows each turn)

    # --- tool execution (docker; the server never sees this) ---
    tool_start: float = 0.0
    tool_end: float = 0.0
    num_tool_calls: int = 0

    def queue_s(self) -> float:
        """Time between admission and prefill start (queue + prefill wait)."""
        if self.request_received > 0.0 and self.decode_start > 0.0:
            return max(self.decode_start - self.request_received, 0.0)
        return 0.0

    def prefill_s(self) -> float:
        """Scheduler-to-first-token: prefill proper, queueing excluded."""
        if self.request_scheduled > 0.0 and self.decode_start > 0.0:
            return max(self.decode_start - self.request_scheduled, 0.0)
        return 0.0

    def decode_s(self) -> float:
        if self.decode_start > 0.0 and self.decode_finished > 0.0:
            return max(self.decode_finished - self.decode_start, 0.0)
        return 0.0

    def gen_s(self) -> float:
        """Client-observed generate latency (Ray round trip included)."""
        if self.gen_call_start > 0.0 and self.gen_call_end > 0.0:
            return max(self.gen_call_end - self.gen_call_start, 0.0)
        return 0.0

    def tool_s(self) -> float:
        if self.tool_start > 0.0 and self.tool_end > 0.0:
            return max(self.tool_end - self.tool_start, 0.0)
        return 0.0

    def cache_hit_rate(self) -> float:
        """Fraction of this turn's context served from the prefix cache.

        Multi-turn rollout re-sends the whole conversation every turn, so this
        being high is what makes the scaffold affordable; a low value means the
        server is re-prefilling the transcript from scratch each turn.
        """
        if self.prompt_tokens > 0:
            return min(self.cached_tokens / self.prompt_tokens, 1.0)
        return 0.0

    def has_server_timing(self) -> bool:
        return self.decode_start > 0.0 and self.decode_finished > 0.0


@dataclass
class TrajectoryMetrics:
    # --- Token Statistics (server-reported via response usage where noted) ---
    prompt_tokens: int = 0        # INITIAL prompt only (system + problem statement).
                                  # NOT the peak context: the conversation grows every turn and
                                  # that growth lands in response_ids. For per-turn context size
                                  # see TurnTiming.prompt_tokens.
    completion_tokens: int = 0    # generated tokens summed across turns (server usage)
    cached_prompt_tokens: int = 0  # prompt tokens served from prefix cache, summed (server usage)
    total_trajectory_tokens: int = 0
    tool_obs_tokens: list[int] = field(default_factory=list)  # exact tokens per tool observation
    response_tokens: int = 0  # tokens the policy is actually trained on (assistant turns)

    # --- Latency (seconds; only what the server can't tell us) ---
    total_trajectory_time: float = 0.0  # wall time start->terminal message
    total_tool_call_time: float = 0.0   # time spent inside env.execute(...) (docker)
    admission_wait_s: float = 0.0       # queued for a live-container permit, before t_start

    # --- Timeline (UNIX wall clock, time.time(); 0.0 == never reached) --------
    # Absolute instants, not durations, so a trajectory can be laid over the
    # step's rollout window and over the srv/* drain curve. "Rollout end" is the
    # end of the *episode* (container released); harness grading is a separate
    # span because it runs its own docker container and would otherwise swamp
    # the generation timeline.
    t_start: float = 0.0            # episode admitted to the live set; work begins
    t_first_decode: float = 0.0     # prefill of turn 1 finished / decoding began
    t_last_gen_end: float = 0.0     # last generate call returned
    t_end: float = 0.0              # episode over: loop exited, container released
    t_score_start: float = 0.0      # SWE-bench harness invoked
    t_score_end: float = 0.0        # harness returned

    # Per-turn timeline. Kept out of the batch payload shipped to the trainer
    # (see ``to_dict(include_turns=False)``) — 80 turns x a dozen floats per
    # trajectory x 2048 trajectories is a lot of Ray-serialised non-tensor data
    # for something only the dump consumes.
    turns: list[TurnTiming] = field(default_factory=list)
    # "server" when SGLang reported per-request timestamps, "client" when only
    # the generate-call boundaries were observable. A run silently degrading to
    # "client" is the difference between a real prefill/decode split and a
    # round-trip measurement, so it is recorded rather than inferred.
    timing_source: str = "none"

    # Server-side spans summed over the episode's turns. Derived from ``turns``
    # by :meth:`summarize_turns`, and kept as plain scalars so they survive the
    # trip to the trainer without the per-turn detail.
    server_queue_time: float = 0.0    # admission -> decode start, summed
    server_prefill_time: float = 0.0  # scheduled -> decode start, summed
    server_decode_time: float = 0.0   # decode start -> decode finish, summed

    # --- Action / Event Counts ---
    tool_call_count: int = 0
    edit_count: int = 0       # str_replace_based_edit_tool calls that changed a file
    edit_errors: int = 0      # edit-tool calls refused (no/ambiguous match, bad args, missing file)
    view_count: int = 0       # edit-tool `view` calls (reads that did not go through cat)
    unknown_tool_calls: int = 0  # calls naming a tool that does not exist (e.g. the command as name)
    test_runs: int = 0        # bash calls that ran pytest/tox/unittest (see agent_loop._TEST_CMD_RE)
    num_turns: int = 0
    truncated: bool = False   # context_length breached -> trajectory cut short
    format_errors: int = 0        # turns whose tool call we could neither parse nor salvage
    salvaged_tool_calls: int = 0  # calls hermes rejected but we recovered (see _salvage_tool_calls)

    # --- Artifacts ---
    pytest_output_length: int = 0  # size of the harness test log

    # --- Outcome (from upstream SWE-bench harness report.json) ---
    resolved: bool = False
    reward: float = 0.0
    # Why a reward is 0. Without these, "harness is broken" and "agent produced a
    # wrong patch" are the same number -- which is exactly how a missing swebench
    # install went unnoticed across two full runs.
    empty_patch: bool = False     # agent never submitted a diff
    # Submission came from the git-diff fallback rather than the submit marker
    # (see agent_loop._recover_patch). Splits "the agent did nothing" from "the
    # agent did the work and forgot to submit", which empty_patch alone conflated.
    patch_recovered: bool = False
    eval_error: str = ""          # harness raised (missing dep, docker, timeout)
    eval_cached: bool = False     # verdict reused from the patch-keyed cache
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

    def summarize_turns(self) -> None:
        """Fold ``turns`` into the scalar timeline fields. Idempotent.

        Call once after the episode ends and before the metrics leave the worker:
        ``turns`` is dropped from the trainer payload, so anything the aggregate
        needs has to be reduced to a scalar here.
        """
        if not self.turns:
            return
        served = [t for t in self.turns if t.has_server_timing()]
        self.timing_source = "server" if served else "client"
        self.server_queue_time = sum(t.queue_s() for t in served)
        self.server_prefill_time = sum(t.prefill_s() for t in served)
        self.server_decode_time = sum(t.decode_s() for t in served)
        if served:
            self.t_first_decode = served[0].decode_start
            # Prefix-cache hits, which only the server can account for. Every turn
            # after the first re-sends the whole conversation, so this is normally
            # most of the prompt and is the reason multi-turn rollout is affordable
            # at all. Skipped entirely when the server is silent, so the metric
            # stays 0 rather than reading as "no cache hits".
            self.cached_prompt_tokens = sum(t.cached_tokens for t in served)
        gen_ends = [t.gen_call_end for t in self.turns if t.gen_call_end > 0.0]
        if gen_ends:
            self.t_last_gen_end = max(gen_ends)

    @property
    def _edit_attempts(self) -> int:
        return self.edit_count + self.edit_errors

    def to_dict(self, include_turns: bool = True) -> dict[str, Any]:
        out = asdict(self)
        if not include_turns:
            out.pop("turns", None)
        return out

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

        # How episodes ended, as a fraction of the batch. With a binary reward, an
        # all-zero batch is otherwise opaque: this separates "agent ran out of
        # turns" (TurnLimit) from "gave up" (NoToolCall), "couldn't emit valid
        # calls" (FormatErrorLimit) and infra failure (Crashed:*).
        exits = Counter(m.exit_status or "Unknown" for m in batch)

        return {
            **{f"traj/exit/{status}": count / n for status, count in exits.items()},
            # If eval_error_rate is not ~0 the reward signal is not measuring the
            # agent at all -- treat any nonzero value as a broken run, not a hard task.
            "reward/eval_error_rate": sum(1 for m in batch if m.eval_error) / n,
            "reward/empty_patch_rate": sum(1 for m in batch if m.empty_patch) / n,
            # Fraction of graded patches served from cache instead of a fresh
            # container. Legitimately nonzero (GRPO groups do emit duplicate
            # patches), but a rate near 1.0 means the cache key has stopped
            # discriminating -- the exact failure that made 3870 submitted
            # patches share 23 harness reports in run 20260820-023256.
            "reward/cache_hit_rate": (
                sum(1 for m in batch if m.eval_cached)
                / max(1, sum(1 for m in batch if not m.empty_patch))
            ),
            # Episodes the git-diff fallback rescued, and how they graded. A
            # recovered patch is a real submission the submit marker missed, so
            # a resolve rate near zero here would mean the fallback is only
            # adding noise and should go back off.
            "reward/patch_recovered_rate": sum(1 for m in batch if m.patch_recovered) / n,
            "reward/recovered_resolve_rate": (
                sum(1 for m in batch if m.patch_recovered and m.resolved)
                / max(1, sum(1 for m in batch if m.patch_recovered))
            ),
            "traj/resolve_rate": sum(1 for m in batch if m.resolved) / n,
            "traj/mean_reward": mean("reward"),
            "reward/patch_applied_rate": sum(1 for m in batch if m.patch_applied) / n,
            "reward/f2p_pass_rate": rate("f2p_passed", "f2p_total"),
            "reward/p2p_pass_rate": rate("p2p_passed", "p2p_total"),
            "traj/mean_turns": mean("num_turns"),
            "traj/mean_tool_calls": mean("tool_call_count"),
            "traj/mean_edits": mean("edit_count"),
            # Edit-tool adoption and reliability. edit_error_rate is per *edit
            # attempt*: a high value means the model is not copying old_str
            # exactly (whitespace) or is editing without viewing first.
            "traj/mean_edit_errors": mean("edit_errors"),
            "traj/edit_error_rate": rate("edit_errors", "_edit_attempts"),
            "traj/mean_views": mean("view_count"),
            "traj/edit_tool_use_rate": sum(1 for m in batch if m.edit_count or m.view_count or m.edit_errors) / n,
            "traj/mean_unknown_tool_calls": mean("unknown_tool_calls"),
            # Whether the freed turn budget goes into verification. 0.57% of bash
            # calls / ~2% of episodes ran any test on run 20260903-002235.
            "traj/mean_test_runs": mean("test_runs"),
            "traj/test_run_rate": sum(1 for m in batch if m.test_runs) / n,
            "traj/truncation_rate": sum(1 for m in batch if m.truncated) / n,
            # Tool-call health: unparseable calls used to end the episode outright,
            # so these two decide how much of the batch carries real signal.
            "traj/mean_format_errors": mean("format_errors"),
            "traj/format_error_rate": sum(1 for m in batch if m.format_errors) / n,
            "traj/mean_salvaged_calls": mean("salvaged_tool_calls"),
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
            "latency/mean_admission_wait_s": mean("admission_wait_s"),
            "latency/max_admission_wait_s": max(m.admission_wait_s for m in batch),
            "latency/mean_generation_s": sum(m.generation_time() for m in batch) / n,
            **_timeline_aggregate(batch),
        }


def guard_infra_failures(agg: dict[str, float], step: int | None = None) -> None:
    """Kill the run when a batch is mostly *infrastructure* failure, not policy failure.

    Both failure classes land on reward 0.0 and are indistinguishable there, so a
    broken run looks exactly like a hard task and trains for hours on noise. Run
    20260811-011747 did precisely that: at 64 live containers per worker the
    rootless daemon returned ``exit status 125`` for most ``docker run`` calls,
    87% of every batch became ``Crashed:CalledProcessError``, mean reward fell
    from 0.093 to 0.017, and five steps plus a 342GB checkpoint were spent before
    anyone looked. ``reward/eval_error_rate`` stayed 0.0 throughout -- the harness
    was healthy, so nothing in the reward path could notice.

    Two signals, both of which mean "the reward is not measuring the agent":

    * ``traj/exit/Crashed:*`` -- the episode died before producing a trajectory
      (container start exhausted its retries, or the loop raised).
    * ``reward/eval_error_rate`` -- the harness could not grade a submitted patch.

    A healthy run sits at ~0.0 on both, so the thresholds are loose on purpose:
    this catches collapse, not the occasional flake. Raising is the point --
    verl's fit() has no other way to say "stop, this data is worthless", and the
    alternative (a warning in a log nobody is tailing) is what already failed.

    Env:
      ``AGENTIC_MAX_CRASH_RATE`` abort above this crashed fraction (default 0.25)
      ``AGENTIC_CRASH_GUARD=0``  disable the abort entirely (still warns)
    """
    crashed = sum(v for k, v in agg.items() if k.startswith("traj/exit/Crashed"))
    eval_err = agg.get("reward/eval_error_rate", 0.0)
    worst = max(crashed, eval_err)
    if worst < 0.02:
        return

    try:
        limit = float(os.environ.get("AGENTIC_MAX_CRASH_RATE", "") or 0.25)
    except ValueError:
        limit = 0.25
    where = f"step {step}" if step is not None else "this step"
    detail = (
        f"{crashed:.1%} of the batch crashed before producing a trajectory "
        f"(traj/exit/Crashed:*), {eval_err:.1%} failed harness grading "
        f"(reward/eval_error_rate)"
    )
    if worst < limit or os.environ.get("AGENTIC_CRASH_GUARD", "").lower() in {"0", "off", "false", "no"}:
        logger.warning("infra failures at %s: %s -- reward signal is degraded", where, detail)
        return
    raise RuntimeError(
        f"Aborting: {detail} at {where}. These are infrastructure failures, not the "
        f"agent failing the task, and they are indistinguishable from real reward-0 "
        f"samples once training starts. Check docker (`exit status 125` means the "
        f"daemon is saturated -- lower AGENTIC_MAX_LIVE_CONTAINERS) and the harness. "
        f"Set AGENTIC_MAX_CRASH_RATE higher or AGENTIC_CRASH_GUARD=0 to override."
    )


def _timeline_aggregate(batch: list["TrajectoryMetrics"]) -> dict[str, float]:
    """Scalar timeline metrics for a batch of trajectories.

    The wall-clock span (``timeline/rollout_span_s``) is the *real* cost of the
    rollout phase: the sum of per-trajectory times overcounts massively because
    thousands of episodes run concurrently, and the mean hides stragglers. The
    span plus ``timeline/tail_s`` (how long after the median episode finished the
    last one was still running) is what says whether a step is bound by a long
    tail — which is the shape docker-slot starvation produces.
    """
    n = len(batch)
    out: dict[str, float] = {}

    served = [m for m in batch if m.timing_source == "server"]
    out["timeline/server_timing_rate"] = len(served) / n
    if served:
        k = len(served)
        out["latency/mean_server_queue_s"] = sum(m.server_queue_time for m in served) / k
        out["latency/mean_server_prefill_s"] = sum(m.server_prefill_time for m in served) / k
        out["latency/mean_server_decode_s"] = sum(m.server_decode_time for m in served) / k

    # Container start + first prefill: everything between admission and "the
    # model started producing tokens". The admission queue is excluded (it lives
    # in ``admission_wait_s``), so a large value here really is docker or prefill.
    ttfd = [m.t_first_decode - m.t_start for m in batch if m.t_start > 0.0 and m.t_first_decode > 0.0]
    if ttfd:
        out["latency/mean_time_to_first_decode_s"] = sum(ttfd) / len(ttfd)
        out["latency/max_time_to_first_decode_s"] = max(ttfd)

    scores = [m.t_score_end - m.t_score_start for m in batch if m.t_score_start > 0.0 and m.t_score_end > 0.0]
    if scores:
        out["latency/mean_score_s"] = sum(scores) / len(scores)

    starts = [m.t_start for m in batch if m.t_start > 0.0]
    ends = [m.t_end for m in batch if m.t_end > 0.0]
    if starts and ends:
        out["timeline/rollout_span_s"] = max(max(ends) - min(starts), 0.0)
        ordered = sorted(ends)
        median_end = ordered[len(ordered) // 2]
        out["timeline/tail_s"] = max(ordered[-1] - median_end, 0.0)
    return out


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
        # Both clocks: perf_counter for the duration (monotonic), wall clock for
        # the timeline. The standalone loop has no per-turn server timing, so it
        # fills only the episode endpoints.
        self.m.t_start = time.time()
        try:
            yield
        finally:
            self.m.total_trajectory_time = time.perf_counter() - start
            self.m.t_end = time.time()

    @contextmanager
    def tool_call(self) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.m.total_tool_call_time += time.perf_counter() - start
