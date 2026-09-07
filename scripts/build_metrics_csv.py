#!/usr/bin/env python
"""Extract ``step:N - key:value - ...`` lines from a run's console log into metrics.csv.

There is no metrics writer in this repo -- W&B/console only ever see one stdout
line per step (see agent_loop._patch_verl_data_metrics). This just parses that
line back out of run.log into a metric x step table, which previously had to be
done by hand (see METRICS.md).

    python scripts/build_metrics_csv.py --log analysis/<experiment>/run.log

Run by scripts/run_grpo.sh on exit, so every run gets one. Safe to re-run: it
only reads the log.
"""

from __future__ import annotations

import argparse
import csv
import os
import re

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_STEP_RE = re.compile(r"step:(\d+)\s*-\s*(.*)")
_NUM_RE = re.compile(r"^-?\d+\.?\d*(?:[eE][-+]?\d+)?")


def parse_log(path: str) -> tuple[list[int], dict[str, dict[int, str]]]:
    """Returns (sorted step numbers, {metric: {step: value}})."""
    values: dict[str, dict[int, str]] = {}
    steps: set[int] = set()
    with open(path, errors="replace") as fh:
        for line in fh:
            line = _ANSI_RE.sub("", line)
            m = _STEP_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            steps.add(step)
            for token in m.group(2).split(" - "):
                key, sep, rest = token.partition(":")
                if not sep:
                    continue
                num = _NUM_RE.match(rest)
                if not num:
                    continue
                values.setdefault(key, {})[step] = num.group(0)
    return sorted(steps), values


def write_csv(out_path: str, steps: list[int], values: dict[str, dict[int, str]]) -> None:
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric"] + [f"step_{s}" for s in steps])
        for metric in sorted(values):
            writer.writerow([metric] + [values[metric].get(s, "") for s in steps])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True, help="path to run.log")
    ap.add_argument("--out", default=None, help="output csv (default: metrics.csv next to --log)")
    args = ap.parse_args()

    out_path = args.out or os.path.join(os.path.dirname(args.log), "metrics.csv")
    steps, values = parse_log(args.log)
    if not steps:
        print(f"build_metrics_csv: no 'step:N - ...' lines found in {args.log}, skipping")
        return
    write_csv(out_path, steps, values)
    print(f"build_metrics_csv: wrote {out_path} ({len(values)} metrics x {len(steps)} steps)")


if __name__ == "__main__":
    main()
