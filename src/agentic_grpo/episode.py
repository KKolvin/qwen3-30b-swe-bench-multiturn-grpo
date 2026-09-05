"""One rollout's state and its record.

:class:`Episode` is what the agent loop threads through its phases. It owns the
token trajectory, the per-turn timings, the metrics object, the optional
timeline recorder and the optional dump — and exposes **one method per event**
(``generated``, ``tool_called``, ``format_error`` ...) so the control flow in
:mod:`agentic_grpo.agent_loop` reads as control flow, not as bookkeeping with
a loop somewhere inside it.

Nothing here talks to verl, docker or the model server; it is all plain data
and clocks, which is what makes it unit-testable without either.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from contextlib import nullcontext
from typing import Any

from agentic_grpo import tools
from agentic_grpo.bash_tool import BASH_TOOL_NAME
from agentic_grpo.editor_tool import EDIT_TOOL_NAME
from agentic_grpo.metrics import TrajectoryMetrics, TurnTiming
from agentic_grpo.sglang_timing import SGLANG_TIMING_KEY
from agentic_grpo.timeline import trajectory_timeline

logger = logging.getLogger("agentic_grpo.episode")


# ---------------------------------------------------------------------------
# tokens
# ---------------------------------------------------------------------------
class Trajectory:
    """Accumulate the flat token sequence + response mask for one episode.

    Mirrors verl's ``tool_agent_loop`` layout: ``prompt_ids`` is the full running
    sequence (initial prompt + every generated turn + every tool observation),
    and ``response_mask`` covers only the post-prompt region — ``1`` for tokens
    the policy generated (trained on), ``0`` for tool-observation tokens.
    """

    def __init__(self, prompt_ids: list[int]):
        self._all: list[int] = list(prompt_ids)
        self._prompt_len = len(prompt_ids)
        self._mask: list[int] = []

    def current_ids(self) -> list[int]:
        """Full sequence so far — the prompt for the next ``generate`` call."""
        return self._all

    def add_generated(self, ids: list[int]) -> None:
        self._all.extend(ids)
        self._mask.extend([1] * len(ids))

    def add_tool(self, ids: list[int]) -> None:
        self._all.extend(ids)
        self._mask.extend([0] * len(ids))

    def response_len(self) -> int:
        return len(self._mask)

    def finalize(self, response_length: int, *, pad_token_id: int) -> tuple[list[int], list[int], list[int]]:
        """Return ``(prompt_ids, response_ids, response_mask)`` for AgentLoopOutput.

        ``response_ids``/``response_mask`` are right-clipped to ``response_length``.
        If the episode produced no response tokens at all (e.g. the container
        failed before the first generation), emit a single padding token so verl's
        ``_pad_token_ids`` never sees an empty list — an empty response is what
        crashed the previous HTTP-based path.
        """
        prompt = self._all[: self._prompt_len]
        response = self._all[self._prompt_len :]
        mask = self._mask
        if not response:
            return prompt, [pad_token_id], [1]
        return prompt, response[:response_length], mask[:response_length]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def truncate(text: str, max_len: int, side: str = "middle") -> str:
    if max_len <= 0 or len(text) <= max_len:
        return text
    if side == "left":
        return "(truncated)..." + text[-max_len:]
    if side == "right":
        return text[:max_len] + "...(truncated)"
    half = max_len // 2
    return text[:half] + "...(truncated)..." + text[-half:]


def turn_timing(turn: int, call_start: float, call_end: float, out: Any) -> TurnTiming:
    """Build a :class:`TurnTiming` for one generate call.

    ``call_start``/``call_end`` are wall clock (``time.time()``) around the Ray
    round trip. The server's own timestamps ride on ``extra_fields`` when
    :mod:`agentic_grpo.sglang_timing` is active — see that module for why the
    prefill/decode boundary is unobtainable from this side. Absent them, the turn
    carries client timing only and the trajectory reports
    ``timing_source == "client"``.
    """
    t = TurnTiming(turn=turn, gen_call_start=call_start, gen_call_end=call_end)
    srv = (getattr(out, "extra_fields", None) or {}).get(SGLANG_TIMING_KEY)
    if isinstance(srv, dict):
        t.request_received = float(srv.get("request_received", 0.0) or 0.0)
        t.request_scheduled = float(srv.get("request_scheduled", 0.0) or 0.0)
        # SGLang's prefill_finished_ts: the instant the first token was sampled,
        # i.e. prefill done and decoding under way.
        t.decode_start = float(srv.get("prefill_finished", 0.0) or 0.0)
        t.decode_finished = float(srv.get("decode_finished", 0.0) or 0.0)
        t.response_sent = float(srv.get("response_sent", 0.0) or 0.0)
        t.completion_tokens = int(srv.get("completion_tokens", 0) or 0)
        t.cached_tokens = int(srv.get("cached_tokens", 0) or 0)
        t.prompt_tokens = int(srv.get("prompt_tokens", 0) or 0)
    return t


# The commands that count as "ran the tests": the behaviour the edit tool is
# meant to free turn budget for. 0.57% of bash calls on run 20260903-002235.
TEST_CMD_RE = re.compile(
    r"(?<![\w.-])(?:pytest|py\.test|tox|python[23]?\s+-m\s+(?:pytest|unittest)|runtests\.py|manage\.py\s+test)(?![\w.-])"
)


def count_call(metrics: TrajectoryMetrics, name: str, args: dict, obs: dict) -> None:
    """Per-call behaviour counters (see ``TrajectoryMetrics.aggregate``)."""
    if name == EDIT_TOOL_NAME:
        if obs.get("edit"):
            metrics.edit_count += 1
        elif obs.get("returncode") == 0:
            metrics.view_count += 1
        else:
            metrics.edit_errors += 1
    elif name == BASH_TOOL_NAME:
        if TEST_CMD_RE.search(args.get("command", "") or ""):
            metrics.test_runs += 1
    else:
        metrics.unknown_tool_calls += 1


# ---------------------------------------------------------------------------
# the dump (opt-in per-episode JSONL)
# ---------------------------------------------------------------------------
def dump_enabled() -> bool:
    return bool(os.environ.get("AGENTIC_TRAJECTORY_DUMP_DIR", ""))


def dump_trajectory(
    instance_id: str,
    exit_status: str,
    reward: float,
    actions: list[dict],
    metrics: TrajectoryMetrics | None = None,
) -> None:
    """Append one JSON line describing an episode, if dumping is enabled.

    Opt-in via ``AGENTIC_TRAJECTORY_DUMP_DIR``. A binary reward tells you *that* a
    rollout scored 0, never *why* — this records the actual commands so you can
    see whether the agent worked productively and ran out of turns, or never
    attempted the submit marker at all.

    The full :class:`TrajectoryMetrics` goes in too, including the per-turn
    timeline that is deliberately stripped from the batch payload sent to the
    trainer (:meth:`TrajectoryMetrics.to_dict`). This file is the only place the
    per-turn detail survives, so it carries everything: the aggregate W&B numbers
    can say a step had a long tail, but only these records say which instance,
    which turn, and whether the time went to prefill, decode or docker.

    Best-effort: a dump failure must never disturb a rollout.
    """
    dump_dir = os.environ.get("AGENTIC_TRAJECTORY_DUMP_DIR", "")
    if not dump_dir:
        return
    try:
        os.makedirs(dump_dir, exist_ok=True)
        record = {
            "instance_id": instance_id,
            "exit_status": exit_status,
            "reward": reward,
            "num_actions": len(actions),
            "actions": actions,
        }
        if metrics is not None:
            record["metrics"] = metrics.to_dict()
        # One file per process; each line is a complete episode.
        path = os.path.join(dump_dir, f"trajectories-{os.getpid()}.jsonl")
        with open(path, "a") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:  # pragma: no cover - diagnostics must not break training
        logger.warning("trajectory dump failed for %s", instance_id, exc_info=True)


# ---------------------------------------------------------------------------
# the episode
# ---------------------------------------------------------------------------
class Episode:
    """State of one rollout plus every instrument that records it.

    Two clocks throughout: durations come from ``perf_counter`` (monotonic), but
    the instants stored on ``metrics`` are ``time.time()`` so the timeline can be
    laid over the server's own timestamps, which come from another process.
    """

    def __init__(self, instance_id: str, prompt_ids: list[int]):
        self.instance_id = instance_id
        self.metrics = TrajectoryMetrics(instance_id=instance_id)
        self.traj = Trajectory(prompt_ids)
        self.turns: list[TurnTiming] = []  # per-turn timeline, always collected
        self.actions: list[dict] = []  # only populated when dumping is enabled
        # Event stream for the run timeline (None unless AGENTIC_TIMELINE_DIR is
        # set). Generate events are derived from `turns` at finish(); what is
        # recorded here is everything that has no other record: the two waits
        # before the first token, each individual tool call, and cleanup.
        self.tl = trajectory_timeline(instance_id)

        self.gen_s = 0.0
        self.tool_s = 0.0
        self.assistant_turns = 0
        self.consecutive_format_errors = 0
        self.submission = ""
        self.exit_status = "IncompleteRollout"
        self._turn: TurnTiming | None = None
        # Defaults only, so an episode that dies before admission still yields a
        # usable envelope; both are reset by :meth:`admitted`.
        self._t0 = time.perf_counter()
        self.metrics.t_start = time.time()

    # --- lifecycle -----------------------------------------------------------
    def admitted(self, t_slot: float) -> None:
        """The episode got its container permit: real work starts now.

        Admission, not work: with 256 episodes per worker and `cap` permits,
        everything before this instant is a queue position -- a property of the
        batch, not of the trajectory (admitted first waits 0s, admitted last
        waits most of the step; measured 1167s mean against 466s of real work).
        The clocks restart here so no duration counts the queue;
        ``admission_wait`` keeps it.
        """
        self._t0 = time.perf_counter()
        self.metrics.t_start = time.time()
        self.metrics.admission_wait_s = self.metrics.t_start - t_slot
        if self.tl is not None:
            self.tl.mark("admission_wait", t_slot, self.metrics.t_start)

    def span(self, name: str):
        """Time a block on the timeline (no-op when the timeline is off)."""
        return self.tl.span(name) if self.tl is not None else nullcontext()

    def can_continue(self, max_turns: int, response_length: int) -> bool:
        return self.assistant_turns < max_turns and self.traj.response_len() < response_length

    def context_full(self, response_length: int) -> bool:
        return self.traj.response_len() >= response_length

    def stop(self, status: str, *, truncated: bool = False) -> bool:
        """Record why the episode ended. Returns False so callers can ``return ep.stop(...)``."""
        self.exit_status = status
        if truncated:
            self.metrics.truncated = True
        return False

    def crashed(self, exc: BaseException) -> None:
        self.exit_status = f"Crashed:{type(exc).__name__}"

    def close(self) -> None:
        """The episode is over and the container released; grading is timed separately."""
        self.metrics.total_trajectory_time = time.perf_counter() - self._t0
        self.metrics.total_tool_call_time = self.tool_s
        self.metrics.t_end = time.time()
        self.metrics.turns = self.turns
        self.metrics.summarize_turns()

    # --- per-turn events ---------------------------------------------------------
    def generated(self, out: Any, call_start: float, call_end: float, elapsed: float) -> None:
        self.gen_s += elapsed
        self.traj.add_generated(out.token_ids)
        self.assistant_turns += 1
        self._turn = turn_timing(self.assistant_turns, call_start, call_end, out)
        self.turns.append(self._turn)

    def salvaged(self, n: int) -> None:
        self.metrics.salvaged_tool_calls += n

    def format_error(self, attempted: bool, raw: str) -> None:
        """A turn with no usable action; ``attempted`` = there was a broken <tool_call>."""
        self.consecutive_format_errors += 1
        self.metrics.format_errors += 1
        if self.tl is not None and self._turn is not None:
            self.tl.mark("format_error", self._turn.gen_call_end, turn=self.assistant_turns, attempted=attempted)
        if dump_enabled():
            self.actions.append(
                {
                    "turn": self.assistant_turns,
                    "tool": "<format_error>" if attempted else "<no_tool_call>",
                    "command": None,
                    "raw_response": truncate(raw, 600),
                }
            )

    def tool_phase(self, n_calls: int) -> None:
        if self._turn is not None:
            self._turn.tool_start = time.time()
            self._turn.num_tool_calls = n_calls

    def tool_called(self, name: str, args: dict, obs: dict, sub: str | None, call_t: float, elapsed: float) -> None:
        self.metrics.tool_call_count += 1
        self.tool_s += elapsed
        if self._turn is not None:
            self._turn.tool_end = time.time()
        count_call(self.metrics, name, args, obs)
        command = tools.summarize(name, args)
        if self.tl is not None:
            # Per CALL, unlike TurnTiming.tool_start/tool_end, which collapse a
            # turn's calls into one span.
            self.tl.tool_call(
                self.assistant_turns, name, command, call_t,
                returncode=obs.get("returncode"),
                submitted=sub is not None or None,
            )
        if dump_enabled():
            self.actions.append(
                {
                    "turn": self.assistant_turns,
                    "tool": name,
                    "command": truncate(command, 600),
                    "returncode": obs.get("returncode"),
                    "output": truncate(obs.get("output", "") or "", 400),
                    "submitted": sub is not None,
                }
            )

    def add_observation(self, ids: list[int]) -> None:
        self.metrics.tool_obs_tokens.append(len(ids))
        self.traj.add_tool(ids)

    # --- grading -------------------------------------------------------------------
    def scoring(self, t_eval_slot: float) -> None:
        self.metrics.t_score_start = time.time()
        if self.tl is not None:
            # Queueing for a grading slot is not grading. Kept as its own span so
            # `score` stays comparable to the pre-fix timelines.
            self.tl.mark("eval_wait", t_eval_slot, self.metrics.t_score_start)

    def scored(self) -> None:
        self.metrics.t_score_end = time.time()

    # --- output ----------------------------------------------------------------------
    def finalize(self, response_length: int, *, pad_token_id: int) -> tuple[list[int], list[int], list[int]]:
        """Clip the trajectory and fill the token/turn fields of ``metrics``."""
        prompt_ids, response_ids, response_mask = self.traj.finalize(response_length, pad_token_id=pad_token_id)
        m = self.metrics
        m.num_turns = self.assistant_turns
        m.exit_status = self.exit_status
        m.prompt_tokens = len(prompt_ids)
        m.completion_tokens = sum(response_mask)
        m.response_tokens = sum(response_mask)
        m.total_trajectory_tokens = len(prompt_ids) + len(response_ids)
        return prompt_ids, response_ids, response_mask

    def flush(self, reward: float) -> None:
        """Write the dump line and the timeline, once metrics are final."""
        dump_trajectory(self.instance_id, self.exit_status, reward, self.actions, self.metrics)
        if self.tl is not None:
            # One write per episode: finish() adds the generate events (from
            # `turns`), the grading span and the episode envelope on top of what
            # was buffered during the run.
            self.tl.finish(self.metrics)
