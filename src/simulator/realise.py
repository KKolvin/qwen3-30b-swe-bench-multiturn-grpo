"""Turn a workload that says "I don't know" into one the engine can run.

A converter facing an unfamiliar harness leaves fields unmeasured, and
:mod:`simulator.ir` lets it say so by hanging an :class:`~simulator.ir.Unknown`
on the field instead of inventing a number. Nothing downstream of here can cope
with that: the engine wants a duration, not a distribution. This module is the
one step between them. It draws every unknown under a seed and hands the engine
an ordinary :class:`~simulator.ir.Workload`.

Putting the draw here, rather than inside the engine, is what keeps
"the engine must not branch on provenance" (`SIMULATOR.md` §6) true in a workload
that is half estimated. The engine cannot treat an estimated duration differently
from a measured one, because by the time it runs they are the same thing: a
float. What tells them apart is :attr:`~simulator.ir.Request.prov`, which travels
with the workload and into the report, and which the engine never reads.

Two callers:

- the **interval** of `SIMULATOR.md` §11 realises the same workload N times under
  N seeds and takes p10--p90 of the predicted e2e;
- the **sensitivity table** of §6 pins every knob at ``0.5`` and walks one of them
  from ``0.1`` to ``0.9``, which is what :func:`realise`'s ``at`` is for.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Iterator, Mapping

from simulator.ir import IRError, Prefix, Request, Task, Workload

#: Fields counted in tokens. Everything else an :class:`~simulator.ir.Unknown`
#: can land on is seconds, and stays a float.
_INT_FIELDS = frozenset({"prompt_tokens", "shared_prefix_tokens", "completion_tokens",
                         "max_new_tokens", "tokens"})


def knobs(w: Workload) -> dict[str, int]:
    """Knob name to the number of fields it governs, for the sensitivity table.

    The count is the second thing the table should say: a knob covering 95,403
    tool durations and a knob covering one training phase both move the e2e, and
    the reader needs to know which kind of thing they are being asked to go and
    measure.
    """
    out: dict[str, int] = {}
    for carrier in _carriers(w):
        for u in carrier.unknown.values():
            out[u.knob] = out.get(u.knob, 0) + 1
    return out


def realise(w: Workload, *, seed: int, at: Mapping[str, float] | None = None) -> Workload:
    """Draw every unknown field and return a workload with none left.

    ``at`` pins named knobs instead of drawing them: ``{"tool": 0.9}`` takes the
    p90 of what the tool knob's fields declared as their
    :attr:`~simulator.ir.Unknown.scale` and multiplies every tool duration by it,
    leaving each task to draw its own value underneath. Only a knob some field
    declared a scale for can be pinned; pinning one that nobody did would quietly
    do nothing, so it raises instead.

    Holding the seed fixed across a sweep is deliberate and is the reason a pinned
    knob still consumes its draw: with the same seed every per-node draw is
    identical from one point of the sweep to the next, so what the sweep moves is
    the knob and nothing else.

    The result is reproducible: same workload, same seed, same ``at``, same
    numbers. The seed and the drawn quantiles land in
    :attr:`~simulator.ir.Workload.meta` so a prediction can say how it was drawn.

    This does not re-run :func:`~simulator.ir.validate`. A distribution can
    certainly produce a workload that violates the IR's arithmetic -- a drawn
    ``prompt_tokens`` below an undrawn ``shared_prefix_tokens``, say -- so the
    caller should validate the result rather than assume the draw was kind.
    """
    pinned = dict(at or {})
    scalable = {u.knob for c in _carriers(w) for u in c.unknown.values() if u.scale is not None}
    unsweepable = sorted(set(pinned) - scalable)
    if unsweepable:
        raise IRError(
            f"cannot pin {unsweepable}: no field declares a scale under "
            f"{'that knob' if len(unsweepable) == 1 else 'those knobs'}, so pinning it would "
            f"change nothing"
        )
    rng = random.Random(seed)
    drawn: dict[str, float] = {}
    for knob in sorted(knobs(w)):
        # Drawn even when pinned, and then thrown away: pinning a knob must not
        # shorten the random stream, or every per-node draw downstream would move
        # too and a sweep would be measuring itself.
        q = rng.random()
        drawn[knob] = pinned[knob] if knob in pinned else q

    sessions = tuple(
        replace(
            s,
            requests=tuple(_resolve(r, drawn, rng) for r in s.requests),
            tasks=tuple(_resolve(t, drawn, rng) for t in s.tasks),
        )
        for s in w.sessions
    )
    prefixes = tuple(_resolve(p, drawn, rng) for p in w.prefixes)
    meta = {**w.meta, "realised": {"seed": seed, "quantiles": drawn, "pinned": pinned}}
    return replace(w, sessions=sessions, prefixes=prefixes, meta=meta)


def _carriers(w: Workload) -> Iterator[Request | Task | Prefix]:
    for s in w.sessions:
        yield from s.nodes()
    yield from w.prefixes


def _resolve(obj: Request | Task | Prefix, drawn: dict[str, float],
             rng: random.Random) -> Request | Task | Prefix:
    """One node, with its unknowns replaced by numbers and the table cleared.

    Every field draws its own value, so a workload's tool times keep their spread
    and the queue has something to be built out of. The knob's draw then scales
    all of its fields by one factor, which says "this distribution may sit in the
    wrong place" rather than "every tool call in the run is simultaneously a p90
    one".
    """
    if not obj.unknown:
        return obj
    values: dict[str, object] = {}
    # Sorted, so the per-node draws do not depend on dict insertion order.
    for name in sorted(obj.unknown):
        u = obj.unknown[name]
        v = u.dist.quantile(rng.random())
        if u.scale is not None:
            v *= u.scale.quantile(drawn[u.knob])
        values[name] = int(round(v)) if name in _INT_FIELDS else float(v)
    return replace(obj, unknown={}, **values)
