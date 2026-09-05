#!/usr/bin/env python
"""Merge the per-process timeline shards of a run into one ``timeline.json``.

:mod:`agentic_grpo.timeline` writes ``timeline-<pid>.jsonl`` from every process
that produces events (the trainer actor plus each ``AgentLoopWorker``). This
merges them, sorts by wall clock, and attributes each trajectory to the training
step whose rollout window contains it -- the workers have no step number of
their own, but ``gen`` / ``validate`` spans from the trainer bracket them
exactly, since a step's rollout does not overlap the next one's.

    python scripts/build_timeline.py --dir <shard dir> [--out timeline.json]

Run by ``scripts/run_grpo.sh`` on exit, so an interrupted run still gets a file.
Safe to re-run: it only reads the shards.
"""

from __future__ import annotations

import argparse
import bisect
import glob
import gzip
import json
import os
import time
from collections import Counter

# Trainer spans that contain trajectories: training rollout, and validation.
_ROLLOUT_PHASES = {"gen": "train", "validate": "val", "testing": "val"}


def load_shards(directory: str) -> tuple[list[dict], list[str]]:
    """Read every shard. A truncated last line (killed run) is skipped, not fatal."""
    events, shards = [], sorted(glob.glob(os.path.join(directory, "timeline-*.jsonl")))
    for path in shards:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    events.sort(key=lambda e: e.get("t", 0.0))
    return events, [os.path.basename(p) for p in shards]


def attribute_steps(events: list[dict]) -> list[dict]:
    """Stamp ``step`` / ``phase`` onto trajectory events; return the step index.

    Attribution is per *trajectory*, not per event: one lookup decides the whole
    episode, so a rollout straddling a phase boundary cannot be split in half.
    """
    windows = sorted(
        (e["t"], e.get("t_end", e["t"]), e.get("step"), _ROLLOUT_PHASES[e["name"]])
        for e in events
        if e.get("cat") == "train" and e.get("name") in _ROLLOUT_PHASES
    )
    starts = [w[0] for w in windows]

    def locate(t: float):
        i = bisect.bisect_right(starts, t) - 1
        # Walk back over enclosing spans (`testing` wraps `validate`) for the
        # innermost one that actually contains t.
        while i >= 0:
            if t <= windows[i][1]:
                return windows[i][2], windows[i][3]
            i -= 1
        return None, None

    by_traj: dict[str, tuple] = {}
    for e in events:
        if e.get("cat") == "traj" and e.get("name") == "episode":
            by_traj[e["traj"]] = locate(e["t"])
    for e in events:
        if e.get("cat") != "traj":
            continue
        step, phase = by_traj.get(e.get("traj", ""), (None, None))
        if step is not None:
            e["step"] = step
        if phase is not None:
            e["phase"] = phase

    steps: dict[int, dict] = {}
    for e in events:
        if e.get("cat") == "train" and isinstance(e.get("step"), int):
            steps.setdefault(e["step"], {"step": e["step"], "phases": {}})["phases"][e["name"]] = {
                "t": e["t"], "t_end": e.get("t_end"), "dur": e.get("dur"),
            }
    for traj_step, _ in by_traj.values():
        if traj_step is not None:
            s = steps.setdefault(traj_step, {"step": traj_step, "phases": {}})
            s["trajectories"] = s.get("trajectories", 0) + 1
    return [steps[k] for k in sorted(steps)]


# Episode-envelope fields that describe the trajectory as a whole; lifted onto
# the trajectory object by --by-trajectory rather than left on a child event.
_ENVELOPE = (
    "instance_id", "step", "phase", "pid", "t", "t_end", "dur", "exit_status", "num_turns",
    "reward", "resolved", "truncated", "tool_calls", "format_errors", "timing_source",
    "prompt_tokens", "response_tokens",
)


def find_waves(events: list[dict], gap: float = 60.0) -> list[dict]:
    """Group episodes into rollout waves by a gap in their start times.

    A step launches its whole batch at once and the next step cannot start until
    the last episode returns, so a gap between consecutive episode *starts*
    separates one rollout phase from the next. This is the fallback for
    ``--wave`` when the trainer's ``gen`` spans are unavailable (they are the
    real answer -- see :func:`attribute_steps` -- but a run killed before its
    buffer flushed, or a shard collected on its own, has none).
    """
    eps = sorted((e for e in events if e.get("name") == "episode"), key=lambda e: e["t"])
    waves: list[list[dict]] = [[]]
    for prev, cur in zip([None] + eps, eps):
        if prev is not None and cur["t"] - prev["t"] >= gap:
            waves.append([])
        waves[-1].append(cur)
    return [
        {
            "wave": i,
            "num_trajectories": len(w),
            "t": w[0]["t"],
            "t_end": max(e.get("t_end", e["t"]) for e in w),
            "step": w[0].get("step"),
            "trajectories": {e["traj"] for e in w},
        }
        for i, w in enumerate(waves)
        if w
    ]


def group_by_trajectory(events: list[dict]) -> list[dict]:
    """One object per trajectory: the episode's own fields, plus its events.

    ``cat``/``traj``/``instance_id``/``pid`` are dropped from the child events --
    they are constant within a trajectory and repeating them across ~90 events x
    2048 trajectories is most of the file size.
    """
    grouped: dict[str, dict] = {}
    for e in events:
        if e.get("cat") != "traj":
            continue
        obj = grouped.setdefault(e["traj"], {"traj": e["traj"], "events": []})
        if e["name"] == "episode":
            obj.update({k: e[k] for k in _ENVELOPE if k in e})
        else:
            obj["events"].append({k: v for k, v in e.items()
                                  if k not in ("cat", "traj", "instance_id", "pid")})
    for obj in grouped.values():
        obj["events"].sort(key=lambda e: e["t"])
        obj["num_events"] = len(obj["events"])
    return sorted(grouped.values(), key=lambda o: o.get("t", 0.0))


def to_chrome_trace(events: list[dict], header: dict) -> dict:
    """Chrome Trace Event Format — loads in chrome://tracing and ui.perfetto.dev.

    Layout: one *process* per producer (the trainer, and each AgentLoopWorker),
    one *track* per trajectory inside its worker. A trajectory's events nest
    inside its ``episode`` span, so each track reads as a flame chart -- the slot
    wait, the container start, then generate/tool_call alternating per turn --
    while the trainer's process shows ``step`` with ``gen`` / ``update_actor`` /
    ``update_weights`` nested inside it, on the same time axis.

    Timestamps are microseconds relative to the first event (chrome renders
    absolute epoch µs as unreadably large numbers); the epoch origin is kept in
    ``otherData.t0_epoch`` so absolute time is recoverable.
    """
    t0 = min(e["t"] for e in events)
    us = lambda t: round((t - t0) * 1e6, 3)  # noqa: E731

    trace, tids, track_name, train_pids = [], {}, {}, set()
    for e in events:
        pid = e.get("pid", 0)
        if e.get("cat") == "traj":
            tid = tids.setdefault(e["traj"], len(tids) + 1)
            track_name.setdefault((pid, tid), e.get("instance_id") or e["traj"])
        else:
            tid, _ = 0, train_pids.add(pid)
        ev = {
            "name": e["name"],
            "cat": e.get("cat", ""),
            "pid": pid,
            "tid": tid,
            "ts": us(e["t"]),
            "args": {k: v for k, v in e.items()
                     if k not in ("cat", "name", "t", "t_end", "dur", "pid", "traj")},
        }
        if "t_end" in e:
            ev["ph"], ev["dur"] = "X", round(e.get("dur", 0.0) * 1e6, 3)
        else:
            ev["ph"], ev["s"] = "i", "t"  # instantaneous (format_error)
        trace.append(ev)

    def meta(name, pid, tid, value):
        return {"ph": "M", "name": name, "pid": pid, "tid": tid, "args": {"name": value}}

    for pid in {e["pid"] for e in trace}:
        label = "trainer" if pid in train_pids else f"AgentLoopWorker {pid}"
        trace.append(meta("process_name", pid, 0, label))
    for (pid, tid), name in track_name.items():
        trace.append(meta("thread_name", pid, tid, name))
    for pid in train_pids:
        trace.append(meta("thread_name", pid, 0, "training phases"))

    trace.sort(key=lambda e: (e.get("ts", -1), e["ph"] == "M"))
    return {
        "traceEvents": trace,
        "displayTimeUnit": "ms",
        "otherData": {**header, "t0_epoch": t0},
    }


def build(directory: str, experiment: str = "") -> dict:
    events, shards = load_shards(directory)
    steps = attribute_steps(events)
    times = [e["t"] for e in events]
    return {
        "run": {
            "experiment": experiment or os.path.basename(os.path.normpath(directory)),
            "built_at": time.time(),
            "t_start": min(times) if times else 0.0,
            "t_end": max(e.get("t_end", e["t"]) for e in events) if events else 0.0,
            "num_events": len(events),
            "num_trajectories": sum(1 for e in events if e.get("name") == "episode"),
            "shards": shards,
            "event_counts": dict(Counter(f"{e.get('cat')}/{e.get('name')}" for e in events).most_common()),
        },
        "steps": steps,
        "events": events,
    }


def _restat(doc: dict) -> None:
    """Recompute the header from ``doc["events"]`` after a filter narrowed them."""
    events = doc["events"]
    doc["run"].update(
        t_start=min((e["t"] for e in events), default=0.0),
        t_end=max((e.get("t_end", e["t"]) for e in events), default=0.0),
        num_events=len(events),
        num_trajectories=sum(1 for e in events if e.get("name") == "episode"),
        event_counts=dict(Counter(f"{e.get('cat')}/{e.get('name')}" for e in events).most_common()),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=os.environ.get("AGENTIC_TIMELINE_DIR", ""),
                    help="shard directory (default: $AGENTIC_TIMELINE_DIR)")
    ap.add_argument("--out", default="", help="output file (default: <dir>/timeline.json)")
    ap.add_argument("--experiment", default="", help="run name recorded in the header")
    ap.add_argument("--gzip", action="store_true", help="write <out>.gz instead")
    ap.add_argument("--summary-only", action="store_true",
                    help="drop the events array (header + per-step index only)")
    ap.add_argument("--format", choices=("flat", "trajectory", "chrome"), default="flat",
                    help="flat: one sorted event stream (default). trajectory: one object per "
                         "trajectory with its events nested. chrome: Chrome Trace Event Format, "
                         "for chrome://tracing and ui.perfetto.dev")
    ap.add_argument("--wave", type=int, default=None,
                    help="keep only the Nth rollout wave (0-based; wave 0 is usually val_before_train). "
                         "The wave table is printed either way.")
    ap.add_argument("--limit", type=int, default=0,
                    help="keep only the first N trajectories -- for viewers that choke on the full step")
    args = ap.parse_args()

    if not args.dir or not os.path.isdir(args.dir):
        raise SystemExit(f"no timeline shard directory: {args.dir!r}")
    doc = build(args.dir, args.experiment)

    waves = find_waves(doc["events"])
    for w in waves:
        print(f"wave {w['wave']}: {w['num_trajectories']:5d} trajectories, "
              f"{w['t_end'] - w['t']:7.0f}s, step={w['step']}")
    if args.wave is not None:
        if not 0 <= args.wave < len(waves):
            raise SystemExit(f"--wave {args.wave} out of range (0..{len(waves) - 1})")
        keep = waves[args.wave]["trajectories"]
        doc["events"] = [e for e in doc["events"] if e.get("traj") in keep]
        doc["run"]["wave"] = args.wave
        doc["run"]["num_trajectories"] = len(keep)
    if args.limit:
        keep = {e["traj"] for e in doc["events"] if e.get("name") == "episode"}
        keep = set(sorted(keep)[: args.limit])
        doc["events"] = [e for e in doc["events"] if e.get("traj") in keep]
        doc["run"]["num_trajectories"] = len(keep)
    for w in waves:  # not JSON-serialisable, and redundant with the events
        w.pop("trajectories")
    doc["waves"] = waves
    if args.wave is not None or args.limit:
        # build() stated the whole run; after filtering the header has to describe
        # what was actually written, or a 2048-trajectory slice claims the run's
        # event count and duration.
        _restat(doc)

    if args.format == "trajectory":
        doc["trajectories"] = group_by_trajectory(doc.pop("events"))
        doc["run"]["num_events"] = sum(t["num_events"] for t in doc["trajectories"])
    elif args.format == "chrome":
        doc = to_chrome_trace(doc["events"], {**doc["run"], "waves": waves})
    if args.summary_only:
        doc.pop("events", None)
        doc.pop("trajectories", None)

    out = args.out or os.path.join(args.dir, "timeline.json")
    if args.gzip:
        out += ".gz"
        with gzip.open(out, "wt") as fh:
            json.dump(doc, fh)
    else:
        with open(out, "w") as fh:
            json.dump(doc, fh)
    run = doc.get("run") or doc["otherData"]
    print(
        f"timeline[{args.format}]: {run['num_events']} events, {run['num_trajectories']} "
        f"trajectories, {run['t_end'] - run['t_start']:.0f}s -> {out} "
        f"({os.path.getsize(out) / 1e6:.1f} MB)"
    )


if __name__ == "__main__":
    main()
