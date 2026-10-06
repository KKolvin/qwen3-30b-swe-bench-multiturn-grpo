"""Trace adapters: one per format, all behind the contract in :mod:`simulator.adapters.base`.

This package is what use time reads a trace with (`SIMULATOR.md` §1), and the
only place in ``simulator`` that may know what a trace file looks like, apart
from the adapter for our own timelines, which only development has and so lives
in :mod:`simulator.dev` (§13). It may import :mod:`simulator.ir` and
:mod:`simulator.observed`, and nothing else from the simulator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from simulator.adapters.agentic_messages import AgenticMessagesAdapter
from simulator.adapters.base import (KNOB_BY_KIND, Adapter, AdapterError, Conversion, Prior,
                                     Priors, check, detect)
from simulator.adapters.priors import default_priors, load_priors, resolve_priors

__all__ = ["ADAPTERS", "Adapter", "AdapterError", "AgenticMessagesAdapter", "Conversion",
           "KNOB_BY_KIND", "Prior", "Priors", "check", "default_priors", "detect", "load", "load_priors",
           "resolve_priors"]

#: Every adapter :func:`load` tries by default: the formats a trace can arrive in
#: at use time. Our own timeline is not one of them; development passes
#: ``simulator.dev.agentic_timeline.AgenticTimelineAdapter(step=...)`` explicitly.
ADAPTERS: tuple[Adapter, ...] = (AgenticMessagesAdapter(),)


def load(path: Path, priors: Priors, adapters: Sequence[Adapter] | None = None) -> Conversion:
    """Read a trace with whichever adapter claims it, and hold the result to the contract."""
    adapter = detect(path, ADAPTERS if adapters is None else adapters)
    conv = adapter.convert(path, priors)
    check(conv, priors)
    return conv
