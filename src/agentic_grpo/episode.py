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

import hashlib
import json
import logging
import os
import re
import time
from contextlib import nullcontext
from typing import Any

from agentic_grpo import bash_tool, tools
from agentic_grpo.bash_tool import BASH_TOOL_NAME
from agentic_grpo.config import int_env
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
        # Rollout-engine logprob per response token (0.0 on tool tokens, which
        # the response mask excludes anyway). Only meaningful when the server
        # returned them; see ``logprobs``.
        self._logprobs: list[float] = []
        self._has_logprobs = False

    def current_ids(self) -> list[int]:
        """Full sequence so far — the prompt for the next ``generate`` call."""
        return self._all

    def add_generated(self, ids: list[int], logprobs: list[float] | None = None) -> None:
        self._all.extend(ids)
        self._mask.extend([1] * len(ids))
        if logprobs is not None and len(logprobs) == len(ids):
            self._logprobs.extend(float(x) for x in logprobs)
            self._has_logprobs = True
        else:
            self._logprobs.extend([0.0] * len(ids))

    def add_tool(self, ids: list[int]) -> None:
        self._all.extend(ids)
        self._mask.extend([0] * len(ids))
        self._logprobs.extend([0.0] * len(ids))

    def logprobs(self, response_length: int) -> list[float] | None:
        """Rollout logprobs aligned with ``finalize``'s ``response_ids``, or None if the server sent none.

        verl pads these to ``response_length`` and ships them as
        ``rollout_log_probs``, which ``algorithm.rollout_correction`` compares
        with the actor's recomputed logprobs to weight (or reject) tokens whose
        sampling distribution drifted from the training one. That drift is what
        makes MoE training unstable: SGLang and FSDP route the experts
        differently for the same weights. The degenerate no-response case
        mirrors ``finalize``: one 0.0 for the single pad token.
        """
        if not self._has_logprobs:
            return None
        if not self._logprobs:
            return [0.0]
        return self._logprobs[:response_length]

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
def _omitted(text: str) -> str:
    """``"N lines (M chars)"`` for the part of an observation that was cut."""
    lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
    return f"{lines} line{'' if lines == 1 else 's'} ({len(text)} chars)"


def _cut_back_to_line(head: str) -> str:
    """Drop the partial last line of ``head`` when it has whole lines to keep."""
    nl = head.rfind("\n")
    return head[: nl + 1] if nl > 0 else head


def _cut_forward_to_line(tail: str) -> str:
    """Drop the partial first line of ``tail`` when it has whole lines to keep."""
    nl = tail.find("\n")
    return tail[nl + 1 :] if 0 <= nl < len(tail) - 1 else tail


def truncate(text: str, max_len: int, side: str = "middle") -> str:
    """Cap an observation at ``max_len`` chars, saying how much was cut.

    ``side`` follows verl's ``tool_response_truncate_side`` convention and names
    the side that is cut: ``"right"`` keeps the head, ``"left"`` keeps the
    tail, ``"middle"`` keeps both ends. Cuts land on line
    boundaries when there are lines to cut on, and the marker carries the
    omitted line count. The bare ``...(truncated)...`` it replaces gave the
    model no idea whether it lost a line or a thousand, and run
    20260903-002235 showed the effect: TurnLimit episodes re-ran the same
    ``cat``/``grep`` hoping to see the rest.
    """
    if max_len <= 0 or len(text) <= max_len:
        return text
    if side == "left":
        tail = _cut_forward_to_line(text[-max_len:])
        return f"(first {_omitted(text[: len(text) - len(tail)])} omitted; only the end is shown)\n{tail}"
    if side == "right":
        head = _cut_back_to_line(text[:max_len])
        omitted = _omitted(text[len(head) :])
        return (
            f"{head.rstrip(chr(10))}\n...({omitted} omitted). Narrow the command "
            "(head, grep -m, a smaller view_range) to see a specific part."
        )
    half = max_len // 2
    head = _cut_back_to_line(text[:half])
    tail = _cut_forward_to_line(text[-half:])
    omitted = _omitted(text[len(head) : len(text) - len(tail)])
    return f"{head.rstrip(chr(10))}\n...({omitted} omitted from the middle)...\n{tail}"


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


# Re-exported: the test-command pattern lives with the bash tool, which also
# uses it to pick the longer timeout.
TEST_CMD_RE = bash_tool.TEST_CMD_RE


# ---------------------------------------------------------------------------
# repetition guard
# ---------------------------------------------------------------------------
class Repeats:
    """Identical tool calls within one episode, and what to do about them.

    TurnLimit episodes on run 20260903-002235 were 16.5% of the batch and were
    not working: one had 80 actions with 7 distinct commands, the same
    ``find``/``grep``/``cat`` cycle over and over ([[turn-cap-not-the-bottleneck]]),
    and 43 of 130 timed-out commands were re-issued verbatim. More turns buys
    more looping; the fix is to make the loop visible to the model and to stop
    paying for it.

    Three rules, all keyed on the exact ``(tool, arguments)``:

    * A call whose result is identical to the last result of that same call is
      not shown again. The observation says which turn already showed it and
      that nothing has changed. That is also a context saving: a repeated
      ``view`` or ``cat`` no longer costs its tokens twice.
    * A call that timed out before is refused outright: the same command will
      hold a container slot for another 60s and time out again.
    * After ``limit`` such repeats in one episode the episode ends
      (``RepetitionLimit``). Its edits are still recovered by the git-diff
      fallback and graded, so a looper that already fixed the bug keeps its
      reward; what it loses is the tail of generate calls that were never
      going to change anything. ``AGENTIC_MAX_REPEATED_CALLS`` (default 6; 0
      disables the stop, keeping the collapse and the timeout refusal).

    A call whose output *differs* from last time resets its baseline: re-running
    the tests after an edit is progress, not repetition.
    """

    def __init__(self, limit: int):
        self.limit = limit
        self._seen: dict[str, dict] = {}
        self.identical = 0

    @staticmethod
    def key(name: str, args: dict) -> str:
        return json.dumps([name, args], sort_keys=True, default=str)

    @staticmethod
    def _digest(obs: dict) -> str:
        parts = [obs.get("returncode"), obs.get("output"), obs.get("exception_info"), (obs.get("extra") or {}).get("exception_info")]
        return hashlib.sha1(json.dumps(parts, default=str).encode("utf-8", "replace")).hexdigest()

    def refusal(self, name: str, args: dict) -> dict | None:
        """The observation to return WITHOUT running the call, or None to run it."""
        prior = self._seen.get(self.key(name, args))
        if prior is None or not prior["timeout"]:
            return None
        prior["same"] += 1
        self.identical += 1
        return {
            "returncode": -1,
            "output": (
                f"Not run: this exact command already timed out at turn {prior['turn']} "
                f"(repeat #{prior['same']}). Re-running it unchanged would time out again. "
                "Use a different command or a non-interactive form."
            ),
            "repeat": prior["same"],
        }

    def observe(self, name: str, args: dict, obs: dict, turn: int) -> dict:
        """Record this call's result; return the observation to show the model."""
        k = self.key(name, args)
        digest = self._digest(obs)
        prior = self._seen.get(k)
        if prior is None or prior["digest"] != digest:
            self._seen[k] = {"turn": turn, "digest": digest, "same": 0, "timeout": bool(obs.get("timeout"))}
            return obs
        prior["same"] += 1
        self.identical += 1
        return {
            "returncode": obs.get("returncode"),
            "output": (
                f"Identical result to the same call at turn {prior['turn']} (repeat #{prior['same']}); "
                "the output is not shown again. Nothing has changed since, so repeating it will not help: "
                "take a different action, or call `submit` if your fix is complete."
            ),
            "repeat": prior["same"],
        }

    def exhausted(self) -> bool:
        return self.limit > 0 and self.identical >= self.limit


def count_call(metrics: TrajectoryMetrics, name: str, args: dict, obs: dict) -> None:
    """Per-call behaviour counters (see ``TrajectoryMetrics.aggregate``)."""
    if obs.get("repeat"):
        metrics.repeated_calls += 1
    if name == EDIT_TOOL_NAME:
        if obs.get("edit"):
            metrics.edit_count += 1
        elif obs.get("returncode") == 0:
            metrics.view_count += 1
        else:
            metrics.edit_errors += 1
    elif name == BASH_TOOL_NAME:
        if obs.get("policy"):
            metrics.policy_denials += 1
        elif bash_tool.is_test_command(args.get("command", "") or ""):
            metrics.test_runs += 1
        if obs.get("timeout"):
            metrics.tool_timeouts += 1
    elif tools.lookup(name) is None:
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
        self.repeats = Repeats(int_env("AGENTIC_MAX_REPEATED_CALLS", 6))
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
        self.traj.add_generated(out.token_ids, getattr(out, "log_probs", None))
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
            record = {
                "turn": self.assistant_turns,
                "tool": name,
                "command": truncate(command, 600),
                "returncode": obs.get("returncode"),
                "output": truncate(obs.get("output", "") or "", 400),
                "submitted": sub is not None,
            }
            if obs.get("repeat"):
                record["repeat"] = obs["repeat"]
            self.actions.append(record)

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
