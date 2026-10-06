"""Priors measured on one of our runs, for converting a trace that recorded less.

The degradation test of `SIMULATOR.md` §15 strips durations out of a trace and
asks the adapter to estimate them. The distributions it estimates from have to
come from somewhere, and the honest somewhere is another run of a harness that
did record them: convert that run with its own adapter, then read the traced task
durations off the resulting workload. :func:`~simulator.adapters.base.check`
refuses the result if that other run turns out to be the one being converted.

The same three steps build the use-time default: ``scripts/simulator_dev/build_default_priors.py``
measures our run with :func:`priors_from_workload`, stretches a scale across our
other runs with :func:`scale_from_runs`, and freezes it with :func:`priors_to_json`.
Reading it back is use time, so that half lives in :mod:`simulator.adapters.priors`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Any, Mapping

from simulator.adapters.base import KNOB_BY_KIND, AdapterError, Prior, Priors
from simulator.ir import Dist, Empirical, Uniform, Workload


def priors_from_workload(w: Workload, *, scale: Dist | None = None,
                         knob_by_kind: Mapping[str, str] = KNOB_BY_KIND) -> Priors:
    """One :class:`Prior` per task kind, from the durations ``w`` actually measured.

    Only traced or derived durations count, and only ones that are not a timeout
    cap: an estimate built on estimates would launder provenance, and a killed
    tool's cap is not how long tools take. ``scale`` is the §6 "how wrong might
    this whole shape be" distribution; it is the caller's claim to make, since it
    comes from comparing runs, which one workload cannot do. ``None`` claims the
    shape is right.
    """
    values: dict[str, list[float]] = defaultdict(list)
    for s in w.sessions:
        for t in s.tasks:
            knob = knob_by_kind.get(t.kind)
            if knob is None or t.duration_is_cap:
                continue
            if t.prov.get("duration_s") in ("traced", "derived"):
                values[knob].append(t.duration_s)
    return Priors.of(*(Prior(knob=k, dist=Empirical(tuple(v)), source=w.source, scale=scale)
                       for k, v in sorted(values.items())))


def scale_from_runs(base: Priors, others: Mapping[str, Priors]) -> tuple[Priors, dict[str, dict[str, float]]]:
    """Give each of ``base``'s priors a scale: how far other runs' means sit from its own.

    The scale is ``Uniform(lo, hi)`` over the ratios of each other run's mean to
    ``base``'s, widened to include 1.0, or ``None`` when no other run measured the
    knob. Means, not medians, because what the simulator sums is durations. Two
    runs give a range, not a confidence interval: it is the spread we have seen,
    and a trace from a different harness may sit outside it. Also returns the
    ratios, ``{knob: {source: ratio}}``, for whoever reports them.
    """
    out, ratios = [], {}
    for knob, p in sorted(base.by_knob.items()):
        m0 = _mean(p.dist)
        r = {o.source: _mean(o.dist) / m0
             for o in (op.by_knob.get(knob) for op in others.values()) if o is not None and m0 > 0}
        lo, hi = min([1.0, *r.values()]), max([1.0, *r.values()])
        scale = Uniform(lo, hi) if hi > lo else None
        out.append(replace(p, scale=scale, scale_sources=tuple(r) if scale else ()))
        ratios[knob] = r
    return Priors.of(*out), ratios


def priors_to_json(priors: Priors, *, meta: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """A :class:`Priors` of :class:`Empirical` dists and :class:`Uniform` scales, as JSON.

    The inverse is :func:`simulator.adapters.priors.priors_from_json`.
    """
    knobs = {}
    for knob, p in sorted(priors.by_knob.items()):
        if not isinstance(p.dist, Empirical):
            raise AdapterError(f"prior {knob!r}: only measured (Empirical) distributions are stored")
        if p.scale is not None and not isinstance(p.scale, Uniform):
            raise AdapterError(f"prior {knob!r}: only Uniform scales are stored")
        # Every value, not a quantile summary: quantiles evenly spaced in q
        # interpolate straight across the tail, and between tool_duration's p99.9
        # (12 s) and its max (300 s) that alone lifts the mean a draw sees by a third.
        knobs[knob] = {
            "source": p.source,
            "n": len(p.dist.values),
            "mean": _mean(p.dist),
            "values": [float(f"{v:.5g}") for v in p.dist.values],
            "scale": [p.scale.lo, p.scale.hi] if p.scale else None,
            "scale_sources": list(p.scale_sources),
        }
    return {"meta": dict(meta or {}), "priors": knobs}


def _mean(d: Dist) -> float:
    xs = d.values  # type: ignore[attr-defined]
    return sum(xs) / len(xs) if xs else float("nan")
