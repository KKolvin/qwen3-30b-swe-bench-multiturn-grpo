"""Priors measured on one workload, for converting a trace that recorded less.

The degradation test of `SIMULATOR.md` §15 strips durations out of a trace and
asks the adapter to estimate them. The distributions it estimates from have to
come from somewhere, and the honest somewhere is another run of a harness that
did record them: convert that run with its own adapter, then read the traced task
durations off the resulting workload. :func:`check` in :mod:`simulator.adapters.base`
refuses the result if that other run turns out to be the one being converted.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping

from simulator.adapters.base import KNOB_BY_KIND, Prior, Priors
from simulator.ir import Dist, Empirical, Workload


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
