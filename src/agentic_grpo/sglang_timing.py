"""Per-request SGLang timestamps, plumbed out to the agent loop.

Why this exists
---------------
The agent loop can time its own ``server_manager.generate(...)`` call, but that
measures a Ray round trip through the load balancer, not the model. It cannot see
where prefill ended and decoding began, because verl asks SGLang for the whole
completion in one shot (``generate_request(...).__anext__()`` — no streaming), so
no first token ever crosses the client boundary.

SGLang itself knows exactly. Its scheduler stamps ``prefill_finished_ts`` with
``time.time()`` the moment a request's first token is sampled
(``scheduler_output_processor_mixin``), and the tokenizer manager publishes that
— plus admission, scheduling and decode-finish timestamps — on the response's
``meta_info``. verl reads three fields off ``meta_info`` (``finish_reason``,
logprobs, output ids) and drops the rest, so the timing never reaches us.

This module recovers it, in two parts:

1. :class:`TimedSGLangHttpServer` — a subclass of verl's rollout-server actor
   that copies the timing subset of ``meta_info`` into
   ``TokenOutput.extra_fields[SGLANG_TIMING_KEY]``. Fields verl already returns
   are untouched, so this is additive: with the patch absent, or the server
   silent, the agent loop simply records client-side timestamps instead.
2. :func:`patch_verl_rollout_timing` — swaps that subclass in as the replica's
   actor class. It runs in the *trainer* process (``SGLangReplica.__init__`` is
   called from ``RayPPOTrainer.init_workers``), which is where the class is
   chosen. Ray then ships the class to the server actor by pickling it **by
   value** (see the capture comment below) -- which is why no state this module
   keeps at module scope is shared with the actor's copy of it.

Requires ``enable_metrics`` on the SGLang server
------------------------------------------------
``meta_info``'s timing block is gated on ``TokenizerManager.enable_metrics``, so
``configs/grpo_swebench.yaml`` sets ``engine_kwargs.sglang.enable_metrics: true``.
That is NOT the same switch as ``rollout.prometheus.enable``: the latter installs
Prometheus middleware on the server's HTTP request path and forces the per-batch
stats loop on (``disable_log_stats: false``), which is why it is off. This flag
only makes the tokenizer manager attach timestamps it already holds to responses
it is already sending — no endpoint, no middleware, no stats loop. Without it,
``meta_info`` carries no timestamps and trajectories fall back to
``timing_source == "client"``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("agentic_grpo.sglang_timing")

# Key under which the timing dict rides on TokenOutput.extra_fields.
SGLANG_TIMING_KEY = "sglang_timing"

# meta_info field -> our name. Everything here is a UNIX wall-clock instant
# (time.time()) except the two explicit durations, so it is comparable across
# processes -- which is the whole point, since the agent loop lives in a
# different actor from the server.
# Deliberately only what TurnTiming consumes. meta_info also carries durations
# (queue_time, prefill_launch_latency) that overlap the spans these timestamps
# already imply; shipping a second, subtly different queue measure would just
# raise the question of which one is "the" queue time.
_TIMESTAMP_FIELDS = {
    "request_received_ts": "request_received",       # admitted by the tokenizer manager
    "request_sent_to_scheduler_ts": "request_scheduled",
    "prefill_finished_ts": "prefill_finished",       # first token sampled -> decode starts
    "decode_finished_ts": "decode_finished",         # last token sampled
    "response_sent_to_client_ts": "response_sent",
}
_COUNT_FIELDS = {
    "completion_tokens": "completion_tokens",
    # The server owns prefix-cache accounting, so these two are the only place the
    # client can learn it. Both are needed: cached_tokens alone cannot be turned
    # into a hit RATE, because the agent loop never sees the per-turn context size
    # (TrajectoryMetrics.prompt_tokens is the INITIAL prompt, not the growing one).
    "cached_tokens": "cached_tokens",
    "prompt_tokens": "prompt_tokens",
}


def extract_timing(meta_info: dict[str, Any]) -> dict[str, Any]:
    """Pull the timing subset out of an SGLang ``meta_info``.

    Returns ``{}`` when the server reported no timestamps at all (``enable_metrics``
    off), which is the signal for the caller to fall back to client-side timing.
    """
    out: dict[str, Any] = {}
    for src, dst in _TIMESTAMP_FIELDS.items():
        v = meta_info.get(src)
        if isinstance(v, (int, float)) and v > 0:
            out[dst] = float(v)
    for src, dst in _COUNT_FIELDS.items():
        v = meta_info.get(src)
        if isinstance(v, int):
            out[dst] = v
    # Timestamps are what this is for; counts alone are not worth a payload.
    if not any(k in out for k in _TIMESTAMP_FIELDS.values()):
        return {}
    return out


# ---------------------------------------------------------------------------
# meta_info capture inside the server actor
# ---------------------------------------------------------------------------
# The rid -> timing map lives on the SERVER INSTANCE, not in a module global.
#
# This is not a style choice, it is the whole correctness of the mechanism. Ray
# pickles an actor class **by value**: `ray.remote(cls)` first runs
# `_inject_tracing_into_class`, which replaces every method with a wrapper, and
# the resulting class no longer resolves back to `getattr(module, qualname)`, so
# cloudpickle serialises it by value. The actor then holds a *second* copy of this
# module's namespace: the copy `generate` closes over, distinct from the real
# `sys.modules` entry that `_install_meta_capture` writes through.
#
# With a module-global map that split is silent and total -- measured on run
# 20260806-040804, the recorder filled the module's dict to 3824 entries while the
# dict `generate` popped from stayed permanently at 0, so every episode fell back
# to client timing. Hanging the map off `self` removes the question: both halves
# are handed the same instance, whatever happened to the class.
_CAPTURE_ATTR = "_agentic_timing_capture"
_WRAPPED_ATTR = "_agentic_timing_wrapped"

# A request that raises between capture and drain would leak an entry. Cap the
# map so a persistent failure mode cannot grow it without bound -- this is a
# diagnostics side channel and must never be the thing that OOMs a rollout
# server. The bound is far above any real in-flight count (max_num_seqs=256).
_MAX_CAPTURED = 4096


def _install_meta_capture(server: Any, tokenizer_manager: Any) -> dict[str, dict[str, Any]]:
    """Tee ``meta_info`` off ``generate_request``; return the map to drain.

    The parent ``generate`` reads only ``output["output_ids"]`` and a few
    ``meta_info`` keys, then discards the dict. Wrapping the async generator lets
    us keep a copy per request id while passing every yielded value through
    untouched.

    The map is created on ``server`` and the wrapper closes over that exact dict,
    so the writer here and the reader in ``generate`` cannot drift apart (see the
    comment above).
    """
    captured = getattr(server, _CAPTURE_ATTR, None)
    if captured is None:
        captured = {}
        setattr(server, _CAPTURE_ATTR, captured)
    if getattr(tokenizer_manager, _WRAPPED_ATTR, False):
        return captured

    original = tokenizer_manager.generate_request

    def remember(rid: str, meta_info: dict[str, Any]) -> None:
        timing = extract_timing(meta_info)
        if not timing:
            return
        if len(captured) >= _MAX_CAPTURED:
            captured.clear()
            logger.warning(
                "sglang_timing: capture map hit %d entries and was cleared; timing is "
                "being recorded but never drained.", _MAX_CAPTURED
            )
        captured[rid] = timing

    def generate_request(obj, request=None, *args, **kwargs):
        inner = original(obj, request, *args, **kwargs)
        rid = getattr(obj, "rid", None)

        async def _tee():
            async for output in inner:
                if isinstance(rid, str) and isinstance(output, dict):
                    meta = output.get("meta_info")
                    if isinstance(meta, dict):
                        remember(rid, meta)
                yield output

        return _tee()

    tokenizer_manager.generate_request = generate_request
    setattr(tokenizer_manager, _WRAPPED_ATTR, True)
    logger.warning("sglang_timing: meta_info capture installed on tokenizer manager")
    return captured


_TIMED_CLS: type | None = None


def _build_timed_server_class() -> type:
    """Define (once) the actor subclass. verl is imported lazily, not at import.

    The class is built on demand rather than at module scope so this module stays
    importable without verl/sglang (tests, the standalone loop). ``__getattr__``
    below re-exports it under a stable module-level name so the class has a real
    importable identity; Ray still pickles actor classes by value (it wraps every
    method first, which defeats the by-reference lookup), so do NOT rely on this
    module's globals being shared with the actor — the per-request map is kept on
    the server instance for exactly that reason.
    """
    global _TIMED_CLS
    if _TIMED_CLS is not None:
        return _TIMED_CLS

    from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer

    class TimedSGLangHttpServer(SGLangHttpServer):  # type: ignore[misc, valid-type]
        """verl's SGLang rollout server, plus per-request timing on the output."""

        async def generate(self, prompt_ids, sampling_params, request_id, **kwargs):  # type: ignore[override]
            # tokenizer_manager only exists on node_rank 0, and only after
            # launch_server; installing lazily avoids depending on either.
            tm = getattr(self, "tokenizer_manager", None)
            if tm is not None:
                try:
                    _install_meta_capture(self, tm)
                except Exception:  # noqa: BLE001 - never break generation for metrics
                    logger.warning("sglang_timing: capture install failed", exc_info=True)

            try:
                output = await super().generate(prompt_ids, sampling_params, request_id, **kwargs)
            finally:
                # Drain from the map on THIS INSTANCE -- never a module global; a
                # by-value class copy would give this method its own empty one.
                # Unconditional: on an abort or exception the entry is dead weight,
                # and leaving it behind is the only way this leaks.
                captured = getattr(self, _CAPTURE_ATTR, None)
                timing = captured.pop(request_id, None) if captured else None

            if timing and output is not None:
                try:
                    output.extra_fields[SGLANG_TIMING_KEY] = timing
                except Exception:  # noqa: BLE001 - e.g. a future non-pydantic output type
                    logger.warning("sglang_timing: could not attach timing", exc_info=True)
            return output

    # Erase the `<locals>` qualname so cloudpickle can look the class back up.
    TimedSGLangHttpServer.__qualname__ = "TimedSGLangHttpServer"
    _TIMED_CLS = TimedSGLangHttpServer
    return TimedSGLangHttpServer


def __getattr__(name: str) -> Any:
    """Resolve ``agentic_grpo.sglang_timing.TimedSGLangHttpServer`` on demand.

    PEP 562 module ``__getattr__``. Both ends of the pickle go through here: the
    trainer when cloudpickle verifies the class is importable, and the server
    actor when Ray resolves the reference — where this is what triggers the
    subclass (and this module's capture hooks) being defined at all.
    """
    if name == "TimedSGLangHttpServer":
        return _build_timed_server_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_ROLLOUT_TIMING_PATCHED = False


def patch_verl_rollout_timing() -> bool:
    """Make SGLang replicas use the timing-aware server actor.

    Call from the process that constructs the replicas (the trainer / TaskRunner
    actor). Returns True if the patch is in place. Best-effort by design: a
    failure costs per-request timestamps, so the agent loop degrades to
    client-side timing rather than the run dying for a diagnostic.

    Disable with ``AGENTIC_SGLANG_REQUEST_TIMING=0``.
    """
    global _ROLLOUT_TIMING_PATCHED
    if _ROLLOUT_TIMING_PATCHED:
        return True
    if os.environ.get("AGENTIC_SGLANG_REQUEST_TIMING", "1") == "0":
        logger.info("sglang_timing: disabled by AGENTIC_SGLANG_REQUEST_TIMING=0")
        return False

    try:
        import ray

        from verl.workers.rollout.sglang_rollout import async_sglang_server as mod

        timed_cls = _build_timed_server_class()
        original_init = mod.SGLangReplica.__init__

        def __init__(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # The parent sets server_class = ray.remote(SGLangHttpServer); swap in
            # the subclass. Ray pickles actor classes by module reference, so the
            # server process imports agentic_grpo.sglang_timing itself -- which is
            # how code reaches a process that never imports our package otherwise.
            self.server_class = ray.remote(timed_cls)

        mod.SGLangReplica.__init__ = __init__
        _ROLLOUT_TIMING_PATCHED = True
        logger.info("sglang_timing: SGLangReplica will use TimedSGLangHttpServer")
        return True
    except Exception:  # noqa: BLE001 - verl/sglang layout drift must not be fatal
        logger.warning(
            "sglang_timing: could not patch SGLangReplica; per-turn timing will be "
            "client-side only (timing_source='client').",
            exc_info=True,
        )
        return False
