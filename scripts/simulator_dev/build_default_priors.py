#!/usr/bin/env python
"""Freeze our run's task durations into the simulator's default priors.

A config bundle that does not state a knob gets this file's distribution for it
(`SIMULATOR.md` §5, :func:`simulator.adapters.priors.resolve_priors`). The shape
comes from one step of one run (``--prior-run``); the scale on each knob is the
spread of the other runs' means around it (``--scale-runs``), widened to include
1.0, exactly as the degradation experiment builds its priors.

Every run named here becomes unusable as a trace to convert under the defaults:
``check`` refuses a prior whose shape or scale was measured on the run being
converted. Converting one of these runs means stating its priors explicitly.

    python scripts/simulator_dev/build_default_priors.py --prior-run 20260912-072938 \\
        --scale-runs 20261005-113503 20261005-143426 20261006-030010
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from simulator.adapters import check  # noqa: E402
from simulator.adapters.base import Priors  # noqa: E402
from simulator.adapters.priors import DEFAULT_PRIORS_PATH  # noqa: E402
from simulator.dev.agentic_timeline import AgenticTimelineAdapter  # noqa: E402
from simulator.dev.priors import priors_from_workload, priors_to_json, scale_from_runs  # noqa: E402

USER = os.environ.get("USER", "nobody")


def measured(timelines: Path, run: str, step: int) -> Priors:
    conv = AgenticTimelineAdapter(step=step).convert(timelines / run, Priors())
    check(conv, Priors())
    return priors_from_workload(conv.workload)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prior-run", required=True)
    ap.add_argument("--scale-runs", nargs="+", required=True)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--timelines", type=Path, default=Path(f"/data0/shared/{USER}/agentic-timelines"))
    ap.add_argument("--out", type=Path, default=DEFAULT_PRIORS_PATH)
    args = ap.parse_args()

    base = measured(args.timelines, args.prior_run, args.step)
    others = {run: measured(args.timelines, run, args.step) for run in args.scale_runs}
    priors, ratios = scale_from_runs(base, others)
    meta = {
        "what": "default priors for knobs a config bundle does not state (SIMULATOR.md §5)",
        "prior_run": args.prior_run, "scale_runs": args.scale_runs, "step": args.step,
        "built": datetime.date.today().isoformat(), "built_by": "scripts/simulator_dev/build_default_priors.py",
        "mean_ratio": {k: {s: round(v, 3) for s, v in r.items()} for k, r in ratios.items()},
    }
    doc = priors_to_json(priors, meta=meta)
    args.out.write_text(json.dumps(doc, separators=(",", ":")) + "\n")

    print(f"{'knob':16} {'n':>6} {'mean':>8} {'p50':>8} {'p90':>8}  scale")
    for knob, p in sorted(priors.by_knob.items()):
        d = doc["priors"][knob]
        sc = f"[{p.scale.lo:.3f}, {p.scale.hi:.3f}]" if p.scale else "none"
        print(f"{knob:16} {d['n']:6d} {d['mean']:8.3f} {p.dist.quantile(0.5):8.3f} "
              f"{p.dist.quantile(0.9):8.3f}  {sc}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
