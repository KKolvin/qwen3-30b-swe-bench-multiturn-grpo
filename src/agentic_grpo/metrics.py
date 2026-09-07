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
from collections import Counter, defaultdict
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
    test_runs: int = 0        # bash calls that ran pytest/tox/unittest (see episode.TEST_CMD_RE)
    policy_denials: int = 0   # bash calls refused by bash_tool.POLICIES (install/network, servers, whole-repo lint)
    tool_timeouts: int = 0    # bash calls killed at the environment timeout
    repeated_calls: int = 0   # calls identical to an earlier one with an unchanged result (episode.Repeats)
    # rollout.calculate_log_probs was on but this episode's output carried no
    # usable server logprobs (crash before generating, or a length mismatch), so
    # its rollout_log_probs are zeros. Should be ~0; nonzero means the
    # importance correction is being fed padding for those episodes.
    missing_logprobs: bool = False
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
        durations = [d for d in (_episode_duration(m) for m in batch) if d > 0.0]
        p50, p99 = _pct(durations, 0.50), _pct(durations, 0.99)

        def rate(passed: str, total: str) -> float:
            tot = sum(getattr(m, total) for m in batch)
            return (sum(getattr(m, passed) for m in batch) / tot) if tot > 0 else 0.0

        # How episodes ended, as a fraction of the batch. With a binary reward, an
        # all-zero batch is otherwise opaque: this separates "agent ran out of
        # turns" (TurnLimit) from "gave up" (NoToolCall), "couldn't emit valid
        # calls" (FormatErrorLimit) and infra failure (Crashed:*).
        exits = Counter(m.exit_status or "Unknown" for m in batch)

        # How much of the batch produced no gradient at all. GRPO normalizes
        # reward WITHIN each group of ``rollout.n`` samples of one prompt, so a
        # group whose samples all scored the same contributes exactly zero
        # advantage: those episodes ran containers, decoded tokens and were
        # graded, and taught the policy nothing. Nothing else logged says this --
        # ``critic/advantages/{max,min}`` only reach 0 when EVERY group is
        # uniform, and they read +-2.47 on a run where 68% of groups were flat
        # (20260906-013757: 175/256 uniform at step 1, of which 38 were already
        # solved). It is the number that argues for filtering the dataset,
        # changing ``rollout.n``, or shaping a denser reward, which is why it
        # earns two passes over the batch.
        #
        # Groups of one (validation, ``rollout.n=1``) are uniform by definition
        # and would read 1.0, so the rows are omitted rather than reported wrong.
        groups: dict[str, list[float]] = defaultdict(list)
        for m in batch:
            if m.instance_id:
                groups[m.instance_id].append(m.reward)
        grouped = [v for v in groups.values() if len(v) > 1]
        group_rates: dict[str, float] = {}
        if grouped:
            g = len(grouped)
            group_rates = {
                "reward/zero_advantage_group_rate": sum(1 for v in grouped if max(v) == min(v)) / g,
                # The half of those that are uniform because the policy already
                # solves the instance: wasted rollout rather than a hard task,
                # and the one the dataset can be filtered on.
                "reward/solved_group_rate": sum(1 for v in grouped if min(v) >= 1.0) / g,
            }

        # Naming rules for this dict (see METRICS.md "Reading the metric names"):
        #   * ``reward/*`` is what the SWE-bench harness produced -- it exists only
        #     because tests were run. ``traj/*`` is what is readable off the episode
        #     itself. So ``resolve_rate`` is a harness verdict and lives under
        #     reward/, while ``empty_patch_rate`` needs no grading and does not.
        #   * ``any_<x>_rate`` is the share of EPISODES with at least one x. A bare
        #     ``<x>_rate`` is per event (per edit attempt, per test) or per episode
        #     for something boolean per episode anyway. The pair that used to read
        #     as parallel and is not: traj/edit_error_rate (per attempt) vs
        #     traj/any_format_error_rate (per episode).
        return {
            **{f"traj/exit/{status}": count / n for status, count in exits.items()},
            # --- reward/: verdicts from the grading harness --------------------
            "reward/resolve_rate": sum(1 for m in batch if m.resolved) / n,
            # If eval_error_rate is not ~0 the reward signal is not measuring the
            # agent at all -- treat any nonzero value as a broken run, not a hard task.
            "reward/eval_error_rate": sum(1 for m in batch if m.eval_error) / n,
            **group_rates,
            "reward/f2p_pass_rate": rate("f2p_passed", "f2p_total"),
            "reward/p2p_pass_rate": rate("p2p_passed", "p2p_total"),
            # Fraction of graded patches served from cache instead of a fresh
            # container. Legitimately nonzero (GRPO groups do emit duplicate
            # patches), but a rate near 1.0 means the cache key has stopped
            # discriminating -- the exact failure that made 3870 submitted
            # patches share 23 harness reports in run 20260820-023256. Named for
            # the EVAL cache: srv/prefix_cache_hit_rate* is SGLang's KV cache and
            # is an unrelated number.
            "reward/eval_cache_hit_rate": (
                sum(1 for m in batch if m.eval_cached)
                / max(1, sum(1 for m in batch if not m.empty_patch))
            ),
            # How the git-diff-rescued episodes graded -- denominator is those
            # episodes, not the batch. Near zero would mean the fallback is only
            # feeding the harness noise and should go back off.
            "reward/recovered_resolve_rate": (
                sum(1 for m in batch if m.patch_recovered and m.resolved)
                / max(1, sum(1 for m in batch if m.patch_recovered))
            ),
            # --- traj/: readable off the episode, no grading needed ------------
            "traj/empty_patch_rate": sum(1 for m in batch if m.empty_patch) / n,
            # Episodes the git-diff fallback rescued: the agent did the work and
            # never submitted. Pair with reward/recovered_resolve_rate.
            "traj/patch_recovered_rate": sum(1 for m in batch if m.patch_recovered) / n,
            "traj/mean_turns": mean("num_turns"),
            "traj/mean_tool_calls": mean("tool_call_count"),
            "traj/mean_edits": mean("edit_count"),
            # The one bare _rate under traj/ with a per-EVENT denominator (edit
            # attempts): a high value means the model is not copying old_str
            # exactly (whitespace) or is editing without viewing first. Adoption
            # (any_edit_tool_rate) was 0.98 and is no longer a question.
            "traj/edit_error_rate": rate("edit_errors", "_edit_attempts"),
            # Whether the freed turn budget goes into verification. 0.57% of bash
            # calls / ~2% of episodes ran any test on run 20260903-002235.
            "traj/mean_test_runs": mean("test_runs"),
            "traj/any_test_run_rate": sum(1 for m in batch if m.test_runs) / n,
            # Structurally useless commands the harness refused before running
            # them (7.0% + 3.6% + 1.1% of tool time on run 20260903-002235), and
            # the timeouts that survive the policy layer.
            "traj/mean_policy_denials": mean("policy_denials"),
            "traj/mean_tool_timeouts": mean("tool_timeouts"),
            # Looping: identical calls with an unchanged result, collapsed by
            # episode.Repeats. Pair with traj/exit/RepetitionLimit.
            "traj/mean_repeated_calls": mean("repeated_calls"),
            "traj/any_repeat_rate": sum(1 for m in batch if m.repeated_calls) / n,
            # How much of the batch carries a usable action at all. The per-turn
            # count and the salvage count were both measured and dropped (0.22
            # and 0.098 a step); the two exit statuses say what a format error
            # actually cost.
            "traj/any_format_error_rate": sum(1 for m in batch if m.format_errors) / n,
            # Generated tokens only -- verl's response_length/* counts the tool
            # observations too, and the ratio of the two (3.3k of 12.7k on run
            # 20260906-013757) is how much of the trained sequence is the agent's
            # own output.
            "tokens/mean_completion": mean("completion_tokens"),
            "tokens/mean_observation": (sum(flat_obs) / len(flat_obs)) if flat_obs else 0.0,
            # An assertion, not a measurement: its correct value is the constant
            # 0.0. guard_infra_failures checks it and strips it, so it never
            # reaches W&B as a flat line (see METRICS.md 13).
            "_check/missing_logprobs_rate": sum(1 for m in batch if m.missing_logprobs) / n,
            # Episode wall time. The MEAN is inflated by the tail -- 272s against
            # a 200s median on run 20260906-013757 -- so the median is what
            # "a typical episode costs" and the mean is what throughput
            # arithmetic wants (work = n x mean).
            "latency/mean_trajectory_s": (sum(durations) / len(durations)) if durations else 0.0,
            "latency/median_trajectory_s": p50,
            "latency/p99_trajectory_s": p99,
            # p99/median is the straggler ratio (4.8-5.1x on that run, against a
            # null of 1.0) -- one division away, so it is not its own row.
            # Tool and grading time are NOT here: we hand both to verl in
            # AgentLoopMetrics, which logs them as
            # timing_s/agent_loop/{tool_calls,compute_score}/{min,max,mean}.
            **_timeline_aggregate(batch),
        }


def _check_instrumentation(agg: dict[str, float], step: int | None = None) -> None:
    """Assert the ``_check/*`` tripwires, then remove them from the metric dict.

    Each one has a single correct value, and each one fails silently: nothing
    else in the metric dict notices, and the failure looks like a property of
    the workload rather than of the measurement.
    """
    where = f"step {step}" if step is not None else "this step"

    coverage = agg.pop("_check/server_timing_coverage", None)
    if coverage is not None and coverage < 0.99:
        logger.warning(
            "only %.1f%% of the batch carried SGLang per-request timestamps at %s: "
            "latency/mean_server_* now describe a subset, and the prefill/decode split "
            "degrades to client-side call boundaries. Check engine_kwargs.sglang."
            "enable_metrics and AGENTIC_SGLANG_REQUEST_TIMING (see METRICS.md 5a).",
            coverage * 100.0, where,
        )

    # Divisibility, not equality: the worker count lives in verl's config and is
    # not visible here, but the observed slots must be ``cap x workers``. A batch
    # smaller than one worker's cap (validation, 20 instances against a cap of
    # 33) never fills the slots, so it can say nothing and is skipped.
    slots = agg.pop("_check/slots_observed", None)
    cap = os.environ.get("AGENTIC_MAX_LIVE_CONTAINERS", "")
    if slots and cap.isdigit() and int(cap) > 0 and slots >= int(cap) and float(slots) % int(cap):
        logger.warning(
            "slots/observed=%d at %s is not a multiple of AGENTIC_MAX_LIVE_CONTAINERS=%s: "
            "the env var probably never reached the AgentLoopWorkers through Ray's "
            "runtime_env and they fell back to the default. Every other metric looks "
            "normal; the rollout is just several times slower.",
            int(slots), where, cap,
        )

    missing = agg.pop("_check/missing_logprobs_rate", None)
    if missing:
        logger.warning(
            "%.1f%% of episodes at %s returned no usable rollout logprobs, so their "
            "rollout_log_probs are zeros and the importance correction is being fed "
            "padding for them (rollout.calculate_log_probs is on).",
            missing * 100.0, where,
        )


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

    It also owns the three *instrumentation* tripwires, which used to be logged
    rows: whether the run is measuring what it thinks it is measuring. Their
    correct value is a constant -- 1.0 coverage, the configured slot count, zero
    episodes with missing logprobs -- so charting them produces a flat line
    nobody reads, and a flat line is the wrong container for an assertion. They
    arrive under ``_check/*`` and are stripped here so they never reach W&B.
    These warn rather than raise: unlike a crashed batch they degrade the
    *analysis*, not the training data.

    Env:
      ``AGENTIC_MAX_CRASH_RATE`` abort above this crashed fraction (default 0.25)
      ``AGENTIC_CRASH_GUARD=0``  disable the abort entirely (still warns)
    """
    _check_instrumentation(agg, step)
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


def _tail_bubble(batch: list["TrajectoryMetrics"], t0: float, t1: float) -> dict[str, float]:
    """What the ragged end of a rollout costs the step barrier.

    The barrier waits for the last episode, so the rollout runs longer than the
    work requires by exactly the slot-seconds that went idle while it waited:
    ``span - sum(durations) / slots`` is that in wall clock -- the seconds a
    perfect packing of the same episodes into the same container slots would
    have saved. No threshold to choose and no server timing to depend on, only
    ``t_start`` and ``t_end``, which every episode has.

    That this is the *tail* rather than general jitter is measured, not assumed:
    on run 20260906-013757 all of it lies after the last instant every slot was
    full. Taking the drain window from that instant accounts for 244/263/258s of
    idle slot time on the three steps -- the total imbalance, to the second. No
    packing loss can accrue while the admission queue is still full, so the whole
    of it is the drain.

    Scale on that run: 244-263s a step, 10-11% of the 2355s rollout but only ~5%
    of the 4933s step -- an order of magnitude under the ~2340s the engine spends
    parked while the trainer runs (``timing_s/step - timing_s/gen``). Worth
    knowing before reading too much into it.

    What the tail is *made of* is :func:`_tail_profile`; this function only
    prices it. The two are separate on purpose: the cost is a scheduling
    quantity, the composition is a statement about the policy's behaviour, and
    on this workload they have different answers.
    """
    live = [(m.t_start, m.t_end) for m in batch if m.t_start > 0.0 and m.t_end > m.t_start]
    span = t1 - t0
    if not live or span <= 0.0:
        return {}

    # Slot count from the concurrency curve itself: the peak number of episodes
    # alive at once IS the semaphore, observed rather than read off a config that
    # may never have reached the workers (264 on that run, matching
    # ``slots/observed``).
    running = slots = 0
    for _, delta in sorted([(s, 1) for s, _ in live] + [(e, -1) for _, e in live]):
        running += delta
        slots = max(slots, running)
    if slots <= 0:
        return {}

    busy = sum(end - start for start, end in live)
    waste = max(span - busy / slots, 0.0)
    out = {"bubble/tail_waste_s": waste}
    # The wasted window is the span past the instant a perfect packing would have
    # finished. Passing it, rather than a percentile of end times, is what keeps
    # the profile tied to the cost it explains.
    out.update(_tail_profile(batch, t1 - waste, t0, t1))
    return out


def _tail_profile(
    batch: list["TrajectoryMetrics"], waste_start: float, t0: float, t1: float
) -> dict[str, float]:
    """What the wasted seconds were spent running.

    ``bubble/tail_waste_s`` prices the window ``[waste_start, t1]``, where
    ``waste_start`` is the instant a perfect packing would have finished. This
    says what was *in* it: every episode is weighted by the seconds it occupies
    that window, so an episode running through all of it counts for all of it and
    one ending a second in counts for a second. No cohort boundary and no
    percentile -- the window comes from the cost metric.

    The control is a window of the **same length** centred mid-rollout, weighted
    the same way, so the ratio's null is exactly 1.0. That matters more than it
    sounds: any tail cohort selected by "still running" oversamples long
    episodes, and long episodes generate more tokens, so a raw contrast against
    the batch starts around 1.7 before anything about the tail is special (§5d
    has the measurement). Dividing by a window that shares the selection rule
    removes it.

    Measured on run 20260906-013757: **1.36 / 1.56 / 1.20**. The wasted seconds
    are spent on episodes that generated 20-56% more than a normal window's
    occupants -- real, and much smaller than the tip suggests. The last 21
    episodes to finish score 3.0-3.8x against a 1.7 null, but they only hold the
    barrier for the final 77-104s of the 244-263s: most of the waste is ordinary
    episodes finishing at their ordinary length, and only its thinnest part is
    the policy looping until it runs out of context.

    Reading: **1.0 means the tail is pure scheduling** -- the wasted window was
    running the same kind of episode as the middle of the rollout, and the only
    fix is throughput. The further above 1.0, the more the barrier is being held
    by degenerate generation, which should move with ``traj/mean_repeated_calls``
    and ``traj/exit/ContextLimit``.
    """
    window = t1 - waste_start
    if window <= 0.0 or t1 <= t0:
        return {}
    mid = (t0 + t1) / 2.0

    def weighted_tokens(a: float, b: float) -> float:
        """Mean response tokens of the episodes occupying ``[a, b]``, by seconds held."""
        num = den = 0.0
        for m in batch:
            if m.t_start <= 0.0 or m.t_end <= m.t_start:
                continue
            held = min(m.t_end, b) - max(m.t_start, a)
            if held > 0.0:
                num += m.response_tokens * held
                den += held
        return (num / den) if den > 0.0 else 0.0

    control = weighted_tokens(mid - window / 2.0, mid + window / 2.0)
    if control <= 0.0:
        return {}
    return {"tail/response_tokens_ratio": weighted_tokens(waste_start, t1) / control}


def _episode_duration(m: "TrajectoryMetrics") -> float:
    """Episode wall time, preferring the timestamps.

    ``t_end - t_start`` and ``total_trajectory_time`` are the same quantity --
    verified equal to 1e-6s over all 6143 episodes of run 20260906-013757, mean
    269.4s against the 269.0-272.1s that run logged. The timestamps win so this
    and ``timeline/rollout_span_s`` are always read off one clock; the field is
    the fallback for records that carry no timeline.
    """
    if m.t_start > 0.0 and m.t_end > 0.0:
        return max(m.t_end - m.t_start, 0.0)
    return m.total_trajectory_time


def _pct(values: list[float], q: float) -> float:
    """Nearest-rank percentile. 0.0 for an empty list.

    Deliberately not a max: a max over ~2048 episodes tracks whichever single
    container hung (which is why ``latency/max_trajectory_s`` was dropped), while
    p99 is the 20th-slowest and moves only when the tail really moves.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * q), len(ordered) - 1)]


def _timeline_aggregate(batch: list["TrajectoryMetrics"]) -> dict[str, float]:
    """Scalar timeline metrics for a batch of trajectories.

    The wall-clock span (``timeline/rollout_span_s``) is the *real* cost of the
    rollout phase: the sum of per-trajectory times overcounts massively because
    thousands of episodes run concurrently, and the mean hides stragglers.

    The span is mostly THROUGHPUT, not straggling: on run 20260906-013757, 2048
    episodes x 269s of work over ~264 slots is a ~2110s floor against a ~2350s
    span, so ~89% of it is just the time to push the batch through the slots. The
    ~10% that is not is ``bubble/tail_waste_s``. Do not try to read a straggler
    signal out of episode *end* times: they are spread by admission scheduling,
    which is what made the first ``timeline/straggler_ratio`` sit at 0.5 on every
    step (see METRICS.md). The straggler signal lives in the duration spread --
    ``p99_trajectory_s / median_trajectory_s``, 4.8-5.1x against a null of 1.0,
    which is why neither of those two rows is a ratio.
    """
    n = len(batch)
    out: dict[str, float] = {}

    served = [m for m in batch if m.timing_source == "server"]
    # An assertion (correct value 1.0), checked and stripped by
    # guard_infra_failures rather than charted -- see METRICS.md 13.
    out["_check/server_timing_coverage"] = len(served) / n
    if served:
        k = len(served)
        # Prefill and decode, both per episode summed over turns and averaged
        # over the trajectories that HAVE server timing (not the batch --
        # otherwise adding a client-only rollout would look like decoding got
        # faster). The queue span (admission -> scheduled) is not a row: it
        # measured 9ms against 4.0s of prefill and 258s of decode, i.e. the
        # rollout never waits for the scheduler. It is still on every ``generate``
        # timeline event if that ever changes.
        out["latency/mean_server_prefill_s"] = sum(m.server_prefill_time for m in served) / k
        out["latency/mean_server_decode_s"] = sum(m.server_decode_time for m in served) / k

    # Container start + first prefill: everything between admission and "the
    # model started producing tokens". The admission queue is excluded (it lives
    # in ``admission_wait_s``), so a large value here really is docker or prefill.
    ttfd = [m.t_first_decode - m.t_start for m in batch if m.t_start > 0.0 and m.t_first_decode > 0.0]
    if ttfd:
        out["latency/mean_time_to_first_decode_s"] = sum(ttfd) / len(ttfd)

    starts = [m.t_start for m in batch if m.t_start > 0.0]
    ends = [m.t_end for m in batch if m.t_end > 0.0]
    span = 0.0
    if starts and ends:
        span = max(max(ends) - min(starts), 0.0)
        out["timeline/rollout_span_s"] = span
        # Decode tokens per second over the rollout, cluster-wide. Server-reported
        # and exact -- no FLOPs model, no device roofline. Its value is that it is
        # directly comparable to the ~3.6k tok/s SGLang plateaus at when saturated
        # (run 20260906-013757), so the gap says how far from saturation the
        # rollout ran. See METRICS.md 10 for why the MFU form of this is not here.
        decode_tokens = sum((m.completion_tokens or m.response_tokens) for m in batch)
        if span > 0.0 and decode_tokens:
            out["perf/rollout_decode_tok_s"] = decode_tokens / span

        # What the ragged end of the rollout costs, and what the barrier spent it
        # waiting for. The gap BETWEEN rollouts is deliberately not here: it is
        # timing_s/step - timing_s/gen, which verl already logs.
        out.update(_tail_bubble(batch, min(starts), max(ends)))

    # Live-container slots. These replace mean/max ``admission_wait_s``, which
    # carried no information: the queue is arithmetic, not a property of the
    # batch. With N episodes over S slots each slot runs N/S episodes back to
    # back, so the mean wait is just ``(N/S - 1)/2 * mean_trajectory_s`` -- on
    # run 20260906-013757 that formula predicted the measured 855/899/916s to
    # within 8%. What is *not* derivable is whether the slots exist and stay
    # busy, which is what these two say.
    #
    # Exactly the first wave acquires the semaphore uncontended, so this is the
    # observed slot count and it must be ``AGENTIC_MAX_LIVE_CONTAINERS *
    # rollout.agent.num_workers`` (264 on that run, every step). If it silently
    # drops to the default 8*workers the env var never reached the workers
    # through Ray's ``runtime_env`` -- a failure that otherwise only shows up as
    # "this step got 4x slower" with every other metric unchanged. That is an
    # assertion about the launch, not a per-step measurement, so
    # guard_infra_failures checks it and strips it.
    #
    # Slot UTILIZATION is deliberately not a row: it is
    # ``1 - bubble/tail_waste_s / timeline/rollout_span_s`` by construction, and
    # on run 20260906-013757 the identity held to the second on all three steps
    # (244/263/258s of waste against 0.896/0.888/0.889 utilization over a
    # 2355/2349/2326s span). Seconds are the useful unit: they compare directly
    # against timing_s/step.
    out["_check/slots_observed"] = float(sum(1 for m in batch if m.admission_wait_s < 1.0))
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
