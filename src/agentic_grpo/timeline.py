"""Run-wide event timeline: absolute timestamps for rollout *and* training.

W&B gets per-step scalars and :class:`~agentic_grpo.metrics.TrajectoryMetrics`
gets per-episode summaries; neither can say what the machine was doing at a
given instant. This records that: one flat JSON event per thing that happened,
each with its own start (and end) wall-clock instant.

Two producers, one directory:

* ``cat="traj"`` — from :mod:`agentic_grpo.agent_loop` in the worker actors:
  admission wait, container start, every generate call (carrying SGLang's
  own prefill/decode timestamps when available), every tool call, cleanup, and
  harness grading.
* ``cat="train"`` — from the trainer actor via :func:`patch_trainer_timeline`,
  which wraps the ``marked_timer`` verl already brackets every phase with
  (``gen``, ``reward``, ``adv``, ``update_actor``, ``update_weights``,
  ``save_checkpoint``, ``testing``, ``step``). verl keeps only their durations
  in ``timing_s/*``; we keep the instants, so a trajectory can be placed inside
  the phase that produced it.

Everything is ``time.time()`` — the only clock comparable across the trainer
process, the worker actors and SGLang.

Each process appends to ``timeline-<pid>.jsonl`` in ``AGENTIC_TIMELINE_DIR``
(unset disables every entry point here). ``scripts/build_timeline.py`` merges
the shards into one ``timeline.json``. Writes are buffered and best-effort: a
failure disables the writer rather than disturbing a run.

Env: ``AGENTIC_TIMELINE_DIR``, ``AGENTIC_TIMELINE_FLUSH`` (buffered events,
default 400), ``AGENTIC_TIMELINE_MAX_CMD`` (command chars kept, default 200).
"""

from __future__ import annotations

import atexit
import functools
import json
import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator
from uuid import uuid4

logger = logging.getLogger("agentic_grpo.timeline")

DIR_ENV = "AGENTIC_TIMELINE_DIR"


def enabled() -> bool:
    return bool(os.environ.get(DIR_ENV, ""))


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, ""))
    except (TypeError, ValueError):
        return default


class TimelineWriter:
    """Buffered append-only JSONL writer, one file per process."""

    def __init__(self, directory: str, flush_every: int = 400):
        self.directory = directory
        self.path = os.path.join(directory, f"timeline-{os.getpid()}.jsonl")
        self.flush_every = max(1, flush_every)
        self._buf: list[str] = []
        self._lock = threading.Lock()
        self._broken = False

    def emit(self, events: list[dict[str, Any]]) -> None:
        if self._broken or not events:
            return
        with self._lock:
            self._buf.extend(json.dumps(e, ensure_ascii=False, default=str) for e in events)
            if len(self._buf) >= self.flush_every:
                self._flush_locked()

    def flush(self) -> None:
        if not self._broken:
            with self._lock:
                self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._buf:
            return
        try:
            os.makedirs(self.directory, exist_ok=True)
            with open(self.path, "a") as fh:
                fh.write("\n".join(self._buf) + "\n")
        except Exception:  # noqa: BLE001 - diagnostics must never break training
            self._broken = True
            logger.warning("timeline: writing %s failed; disabling", self.path, exc_info=True)
        finally:
            self._buf.clear()


_WRITER: TimelineWriter | None = None
_WRITER_INIT = False


def get_writer() -> TimelineWriter | None:
    """Process-local writer, or None when the timeline is off."""
    global _WRITER, _WRITER_INIT
    if not _WRITER_INIT:
        _WRITER_INIT = True
        directory = os.environ.get(DIR_ENV, "")
        if directory:
            _WRITER = TimelineWriter(directory, flush_every=_int_env("AGENTIC_TIMELINE_FLUSH", 400))
            atexit.register(_WRITER.flush)
    return _WRITER


def _reset_for_tests() -> None:
    """Drop the cached writer so a test can repoint DIR_ENV."""
    global _WRITER, _WRITER_INIT
    if _WRITER is not None:
        _WRITER.flush()
    _WRITER, _WRITER_INIT = None, False


def make_event(name: str, t: float, t_end: float | None = None, *, cat: str, **attrs: Any) -> dict[str, Any]:
    """One flat event. ``t_end is None`` -> instantaneous. ``None`` attrs are dropped."""
    ev: dict[str, Any] = {"cat": cat, "name": name, "t": round(t, 6), "pid": os.getpid()}
    if t_end:
        ev["t_end"] = round(t_end, 6)
        ev["dur"] = round(max(t_end - t, 0.0), 6)
    ev.update({k: v for k, v in attrs.items() if v is not None})
    return ev


def emit(name: str, t: float, t_end: float | None = None, *, cat: str = "train", **attrs: Any) -> None:
    w = get_writer()
    if w is not None:
        w.emit([make_event(name, t, t_end, cat=cat, **attrs)])


# ---------------------------------------------------------------------------
# rollout side
# ---------------------------------------------------------------------------
class TrajectoryTimeline:
    """One episode's events, buffered in memory and written at :meth:`finish`.

    Deferring the write keeps the rollout hot path free of I/O and of a lock
    shared with the worker's other concurrent episodes; the cost is ~20 KiB held
    for the life of an episode that is already holding a docker container.

    ``traj`` is unique per *rollout*: GRPO samples the same instance
    ``rollout.n`` times per step and again every epoch, so ``instance_id`` alone
    cannot key a trajectory on a timeline.
    """

    def __init__(self, instance_id: str):
        self.instance_id = instance_id
        self.traj = f"{instance_id}#{uuid4().hex[:8]}"
        self.events: list[dict[str, Any]] = []
        self.max_cmd = _int_env("AGENTIC_TIMELINE_MAX_CMD", 200)

    def mark(self, name: str, t: float, t_end: float | None = None, **attrs: Any) -> None:
        self.events.append(
            make_event(name, t, t_end, cat="traj", traj=self.traj, instance_id=self.instance_id, **attrs)
        )

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[None]:
        """Time a block, recording it even when it raises — a container start
        that dies after 90s is exactly what a timeline should show."""
        start = time.time()
        failed = None
        try:
            yield
        except BaseException as exc:  # noqa: BLE001 - re-raised immediately
            failed = type(exc).__name__
            raise
        finally:
            self.mark(name, start, time.time(), failed=failed, **attrs)

    def tool_call(self, turn: int, tool: str, command: str, start: float, **attrs: Any) -> None:
        cmd = command if len(command) <= self.max_cmd else command[: self.max_cmd] + "...(truncated)"
        self.mark("tool_call", start, time.time(), turn=turn, tool=tool, command=cmd, **attrs)

    def _add_turns(self, turns: list[Any]) -> None:
        """Expand TurnTiming into generate events.

        Derived from the turn list rather than recorded inline so the server's
        per-request timestamps reach the timeline through the one path
        :mod:`agentic_grpo.sglang_timing` already feeds, and cannot drift from
        what the metrics report. The server instants ride as attributes instead
        of three more events: they describe a single request.
        """
        for t in turns:
            attrs: dict[str, Any] = {
                "turn": t.turn,
                "completion_tokens": t.completion_tokens or None,
                "prompt_tokens": t.prompt_tokens or None,
                "cached_tokens": t.cached_tokens or None,
                "num_tool_calls": t.num_tool_calls or None,
            }
            if t.has_server_timing():
                attrs.update(
                    request_received=t.request_received or None,
                    request_scheduled=t.request_scheduled or None,
                    decode_start=t.decode_start,
                    decode_finished=t.decode_finished,
                    response_sent=t.response_sent or None,
                    queue_s=round(t.queue_s(), 6),
                    prefill_s=round(t.prefill_s(), 6),
                    decode_s=round(t.decode_s(), 6),
                )
            self.mark("generate", t.gen_call_start, t.gen_call_end or None, **attrs)

    def finish(self, metrics: Any) -> None:
        """Emit the episode envelope plus everything buffered, in one write.

        ``score`` is a separate span from ``episode``: harness grading runs its
        own container *after* the container the rollout held is released.
        """
        w = get_writer()
        if w is None:
            return
        self._add_turns(getattr(metrics, "turns", None) or [])
        if metrics.t_score_start:
            self.mark(
                "score", metrics.t_score_start, metrics.t_score_end or None,
                reward=metrics.reward, resolved=metrics.resolved,
                empty_patch=metrics.empty_patch or None, eval_error=metrics.eval_error or None,
            )
        self.mark(
            "episode", metrics.t_start, metrics.t_end or None,
            exit_status=metrics.exit_status, num_turns=metrics.num_turns,
            reward=metrics.reward, resolved=metrics.resolved,
            truncated=metrics.truncated or None, tool_calls=metrics.tool_call_count,
            format_errors=metrics.format_errors or None, timing_source=metrics.timing_source,
            patch_recovered=getattr(metrics, "patch_recovered", False) or None,
            prompt_tokens=metrics.prompt_tokens, response_tokens=metrics.response_tokens,
        )
        self.events.sort(key=lambda e: e["t"])
        w.emit(self.events)
        self.events.clear()


def trajectory_timeline(instance_id: str) -> TrajectoryTimeline | None:
    """A recorder for one episode, or None when the timeline is off."""
    return TrajectoryTimeline(instance_id) if enabled() else None


# ---------------------------------------------------------------------------
# training side
# ---------------------------------------------------------------------------
def _caller_step(depth: int = 2) -> int | None:
    """The trainer's ``global_steps``, read off the calling frame.

    ``marked_timer`` is used as a bare context manager inside
    ``RayPPOTrainer.fit``, so the step is not in its arguments — but it is on the
    ``self`` of the frame that called it. A few microseconds a dozen times per
    step; the alternative is patching ``fit`` itself. None when undeterminable,
    in which case the merge attributes the event by wall clock.
    """
    try:
        frame: Any = sys._getframe(depth)
    except ValueError:  # pragma: no cover - shallower stack than expected
        return None
    for _ in range(6):
        if frame is None:
            break
        step = getattr(frame.f_locals.get("self"), "global_steps", None)
        if isinstance(step, int):
            return step
        frame = frame.f_back
    return None


class _TimedPhase:
    """verl's timer context manager, plus the absolute instants."""

    def __init__(self, name: str, inner: Any, step: int | None):
        self.name, self.inner, self.step = name, inner, step
        self.start = 0.0

    def __enter__(self):
        self.start = time.time()
        return self.inner.__enter__()

    def __exit__(self, exc_type, exc, tb):
        try:
            return self.inner.__exit__(exc_type, exc, tb)
        finally:
            emit(self.name, self.start, time.time(), cat="train", step=self.step,
                 failed=exc_type.__name__ if exc_type else None)
            # Flush every training event instead of waiting for the buffer. The
            # trainer emits ~12 events per step -- at a 2.5h step the buffer
            # would not fill for days, and Ray SIGKILLs actors at job end, so
            # atexit is not a reliable backstop. A few hundred bytes appended
            # a dozen times per step costs nothing; losing the training half of
            # the timeline costs the whole cross-referencing.
            w = get_writer()
            if w is not None:
                w.flush()


_TRAINER_PATCHED = False


def patch_trainer_timeline() -> None:
    """Record an event for every training phase verl already brackets.

    Rebinds the ``marked_timer`` *name in* ``verl.trainer.ppo.ray_trainer``: its
    call sites resolve it as a module global at call time, so this catches
    ``gen`` / ``reward`` / ``old_log_prob`` / ``adv`` / ``update_actor`` /
    ``update_critic`` / ``save_checkpoint`` / ``update_weights`` / ``testing`` /
    ``step`` without touching verl. Other modules keep the original.

    ``_validate`` is wrapped separately: ``val_before_train`` runs it outside any
    ``marked_timer``, and on this workload that first validation is ~an hour.

    Must run in the process that executes ``fit()`` — the ``TaskRunner`` actor,
    which is what ``scripts/verl_entry.py`` arranges.
    """
    global _TRAINER_PATCHED
    if _TRAINER_PATCHED or not enabled():
        return
    try:
        import verl.trainer.ppo.ray_trainer as ray_trainer
    except Exception:  # noqa: BLE001 - no verl (standalone/tests): nothing to patch
        return

    original_timer = ray_trainer.marked_timer

    def marked_timer(name, timing_raw, *args, **kwargs):
        # Resolve the step HERE, at call time, while the caller's frame is
        # directly above. A @contextmanager wrapper would defer this body to
        # __enter__, by which point contextlib's frames sit in between.
        return _TimedPhase(name, original_timer(name, timing_raw, *args, **kwargs), _caller_step())

    ray_trainer.marked_timer = marked_timer

    trainer_cls = getattr(ray_trainer, "RayPPOTrainer", None)
    validate = getattr(trainer_cls, "_validate", None)
    if validate is not None:

        @functools.wraps(validate)
        def _validate(self, *args, **kwargs):
            start = time.time()
            try:
                return validate(self, *args, **kwargs)
            finally:
                emit("validate", start, time.time(), cat="train", step=getattr(self, "global_steps", None))
                w = get_writer()
                if w is not None:
                    w.flush()

        trainer_cls._validate = _validate

    _TRAINER_PATCHED = True
    logger.warning("timeline: recording training phases -> %s", os.environ.get(DIR_ENV))
