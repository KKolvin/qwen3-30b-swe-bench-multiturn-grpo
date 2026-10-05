"""Trace adapters: one per format, all behind the contract in :mod:`simulator.adapters.base`.

This package is the only place in ``simulator`` that may know what a trace file
looks like (`SIMULATOR.md` §13). It may import :mod:`simulator.ir` and
:mod:`simulator.observed`, and nothing else from the simulator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from simulator.adapters.agentic_messages import AgenticMessagesAdapter
from simulator.adapters.agentic_timeline import AgenticTimelineAdapter
from simulator.adapters.base import (KNOB_BY_KIND, Adapter, AdapterError, Conversion, Prior,
                                     Priors, check, detect)
from simulator.adapters.priors import priors_from_workload

__all__ = ["ADAPTERS", "Adapter", "AdapterError", "AgenticMessagesAdapter", "AgenticTimelineAdapter", "Conversion",
           "KNOB_BY_KIND", "Prior", "Priors", "check", "detect", "load",
           "priors_from_workload"]

#: Every adapter :func:`load` tries. A run directory with several training steps
#: needs ``AgenticTimelineAdapter(step=...)`` passed explicitly instead.
ADAPTERS: tuple[Adapter, ...] = (AgenticTimelineAdapter(), AgenticMessagesAdapter())


def load(path: Path, priors: Priors, adapters: Sequence[Adapter] | None = None) -> Conversion:
    """Read a trace with whichever adapter claims it, and hold the result to the contract."""
    adapter = detect(path, ADAPTERS if adapters is None else adapters)
    conv = adapter.convert(path, priors)
    check(conv, priors)
    return conv
