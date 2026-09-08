#!/usr/bin/env python
"""Render a run's timeline shards as an HTML report.

    python scripts/plot_timeline.py --dir analysis/<experiment>/timeline

Companion to :mod:`scripts.profile_rollout`, which prints the same reductions as
text. This one draws them: where the step's wall clock goes, how rollout
occupancy decays over the span, what throughput the server actually gets at each
in-flight band, and how long episodes take. One page per run, self-contained
(no network), written next to the shards as ``timeline.html``.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import glob
import json
import os
import statistics

# In-flight request bands -- the load regimes the server runs in, same cuts as
# profile_rollout so the table and the chart can be read together.
BANDS = [(0, 0), (1, 8), (9, 32), (33, 64), (65, 128), (129, 200), (201, 10**9)]

# Trainer spans that contain trajectories.
ROLLOUT_PHASES = {"gen": "train", "validate": "val"}

SERIES_POINTS = 700  # occupancy samples per step; enough for a 3000s span at ~4s


def band_of(c: int) -> tuple[int, int]:
    for lo, hi in BANDS:
        if lo <= c <= hi:
            return (lo, hi)
    return BANDS[-1]


def band_label(b: tuple[int, int]) -> str:
    lo, hi = b
    return str(lo) if lo == hi else (f"{lo}-{hi}" if hi < 10**9 else f"{lo}+")


def load(directory: str):
    by_traj: dict[str, list[dict]] = collections.defaultdict(list)
    train: list[dict] = []
    for path in sorted(glob.glob(os.path.join(directory, "timeline-*.jsonl"))):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:  # truncated last line of a killed run
                    continue
                if e.get("cat") == "train":
                    train.append(e)
                elif e.get("traj"):
                    by_traj[e["traj"]].append(e)
    return by_traj, sorted(train, key=lambda e: e["t"])


def concurrency_peak(intervals) -> int:
    running = peak = 0
    for _, delta in sorted([(a, 1) for a, _ in intervals] + [(b, -1) for _, b in intervals]):
        running += delta
        peak = max(peak, running)
    return peak


# Trace segment kinds. 0 is the gap between spans -- drawn as nothing, but it has
# to be encoded so a row stays a contiguous run of durations.
KINDS = ["idle", "admission", "container", "generate", "tool", "score", "cleanup", "recover"]
KIND_OF = {
    "admission_wait": 1, "container_start": 2, "generate": 3, "tool_call": 4,
    "eval_wait": 5, "score": 5, "cleanup": 6, "patch_recover": 7,
}


def trace_row(events, env, start: float, instances: list, tools: list) -> dict:
    """One trajectory as a contiguous run of ``[kind, ms, kind, ms, ...]``.

    Contiguous is what makes it small: only durations are stored, so a segment's
    position is the sum of everything before it, and a gap between two spans is
    an explicit ``idle`` segment rather than a second timestamp per span.
    """
    spans = sorted(
        (e["t"], e.get("t_end", e["t"]), KIND_OF[e["name"]], e)
        for e in events
        if e["name"] in KIND_OF and e.get("t_end")
    )
    flat, tool_codes, cur = [], [], env["t"]

    def push(kind: int, secs: float) -> None:
        ms = int(round(secs * 1000))
        if ms <= 0:
            return
        if flat and flat[-2] == kind:  # merge a run of the same kind
            flat[-1] += ms
        else:
            flat.extend((kind, ms))

    for t0, t1, kind, e in spans:
        if t1 <= cur:
            continue
        if t0 > cur:
            push(0, t0 - cur)
        push(kind, t1 - max(t0, cur))
        if kind == 4:
            name = e.get("tool", "?")
            if name not in tools:
                tools.append(name)
            tool_codes.append(tools.index(name))
        cur = t1
    if env["t_end"] > cur:
        push(0, env["t_end"] - cur)

    inst = env.get("instance_id", "?")
    if inst not in instances:
        instances.append(inst)
    return {
        "s": int(round((env["t"] - start) * 1000)),
        "d": flat,
        "i": instances.index(inst),
        "x": env.get("exit_status", "?"),
        "r": env.get("reward", 0.0) or 0.0,
        "n": env.get("num_turns", 0),
        "tt": tool_codes,
    }


def step_payload(by_traj, gen, phases, instances: list, tool_names: list) -> dict | None:
    """Every reduction one step's charts need, in one JSON-able dict."""
    t0, t1 = gen["t"], gen["t"] + gen.get("dur", 0.0)
    episodes, requests, tools, traced = [], [], [], []
    for events in by_traj.values():
        env = next((e for e in events if e["name"] == "episode" and e.get("t_end")), None)
        if env is None or not (t0 - 1 <= env["t"] <= t1 + 1):
            continue
        episodes.append(env)
        traced.append(events)
        for e in events:
            if e["name"] == "generate" and e.get("decode_start") and e.get("decode_finished"):
                requests.append(e)
            elif e["name"] == "tool_call":
                tools.append(e)
    if not episodes or not requests:
        return None

    start = min(e["t"] for e in episodes)
    end = max(e["t_end"] for e in episodes)
    span = end - start

    # In-flight concurrency: +1 when a request starts occupying the GPU (it is
    # scheduled), -1 when its last token is sampled.
    busy = [(e.get("request_scheduled") or e["decode_start"], e["decode_finished"]) for e in requests]
    times, conc, running = [], [], 0
    for t, delta in sorted([(a, 1) for a, _ in busy] + [(b, -1) for _, b in busy]):
        running += delta
        times.append(t)
        conc.append(running)

    wall: collections.Counter = collections.Counter()
    for i in range(len(times) - 1):
        dt = times[i + 1] - times[i]
        if dt > 0:
            wall[band_of(conc[i])] += dt
    tokens: collections.Counter = collections.Counter()
    decode: collections.Counter = collections.Counter()
    count: collections.Counter = collections.Counter()
    for e in requests:
        i = bisect.bisect_right(times, e["decode_start"]) - 1
        b = band_of(conc[i] if i >= 0 else 0)
        tokens[b] += e.get("completion_tokens", 0)
        decode[b] += e["decode_finished"] - e["decode_start"]
        count[b] += 1
    total_tok = sum(tokens.values()) or 1

    bands = [
        {
            "label": band_label(b),
            "wall": round(wall[b], 1),
            "pct_span": wall[b] / span,
            "requests": count[b],
            "tokens": tokens[b],
            "pct_tok": tokens[b] / total_tok,
            "ms_per_tok": (decode[b] / tokens[b] * 1000) if tokens[b] else 0.0,
            "tok_s": (tokens[b] / wall[b]) if wall[b] else 0.0,
        }
        for b in BANDS
        if wall[b] or count[b]
    ]

    # Sampled series: concurrent episodes, in-flight requests, decode tok/s.
    dt = max(span / SERIES_POINTS, 0.5)
    nb = int(span / dt) + 1
    ep_grid, req_grid, tok_grid = [0.0] * nb, [0.0] * nb, [0.0] * nb

    def occupancy(intervals, grid: list[float]) -> None:
        """Instantaneous concurrency, sampled at each bucket's midpoint."""
        edges = sorted([(a, 1) for a, _ in intervals] + [(b, -1) for _, b in intervals])
        ts, running, cum = [e[0] for e in edges], 0, []
        for _, d in edges:
            running += d
            cum.append(running)
        for i in range(nb):
            j = bisect.bisect_right(ts, start + (i + 0.5) * dt) - 1
            grid[i] = float(cum[j]) if j >= 0 else 0.0

    def spread(t_a: float, t_b: float, grid: list[float], amount: float) -> None:
        """An amount split evenly over the buckets its interval covers."""
        a = max(0, int((t_a - start) / dt))
        b = min(nb - 1, int((t_b - start) / dt))
        for i in range(a, b + 1):
            grid[i] += amount / (b - a + 1)

    occupancy([(e["t"], e["t_end"]) for e in episodes], ep_grid)
    occupancy(busy, req_grid)
    for e in requests:
        spread(e["decode_start"], e["decode_finished"], tok_grid, e.get("completion_tokens", 0))

    durations = sorted(e["t_end"] - e["t"] for e in episodes)
    n = len(durations)
    slots = max(concurrency_peak([(e["t"], e["t_end"]) for e in episodes]), 1)
    ideal = sum(durations) / slots  # a perfect packing of the same work
    tool_s = sum(e.get("dur", 0.0) for e in tools)
    dec_s = sum(e["decode_finished"] - e["decode_start"] for e in requests)
    q_s = sum(e.get("queue_s", 0.0) for e in requests)
    pre_s = sum(e.get("prefill_s", 0.0) for e in requests)
    ep_wall = sum(durations)
    out_tok = sum(e.get("completion_tokens", 0) for e in requests)
    prompt_tok = sum(e.get("prompt_tokens", 0) for e in requests)
    cached_tok = sum(e.get("cached_tokens", 0) for e in requests)

    trace = sorted(
        (trace_row(ev, next(e for e in ev if e["name"] == "episode" and e.get("t_end")), start,
                   instances, tool_names) for ev in traced),
        key=lambda r: r["s"],
    )

    exits = collections.Counter(e.get("exit_status", "?") for e in episodes)
    tool_mix: dict[str, dict] = {}
    for e in tools:
        row = tool_mix.setdefault(e.get("tool", "?"), {"tool": e.get("tool", "?"), "calls": 0, "s": 0.0})
        row["calls"] += 1
        row["s"] += e.get("dur", 0.0)

    def pct(p: float) -> float:
        return durations[min(n - 1, int(n * p))]

    return {
        "step": gen.get("step"),
        "phase": ROLLOUT_PHASES[gen["name"]],
        "span": span,
        "t_offset": start - t0,  # rollout start relative to the gen span
        "gen_dur": gen.get("dur", 0.0),
        "phases": phases,
        "episodes": n,
        "requests": len(requests),
        "out_tok": out_tok,
        "prompt_tok": prompt_tok,
        "cached_tok": cached_tok,
        "peak_inflight": max(conc),
        "slots": slots,
        "ideal_s": ideal,
        "tail_waste_s": span - ideal,
        "bands": bands,
        "dt": dt,
        "series": {
            "episodes": [round(v, 1) for v in ep_grid],
            "inflight": [round(v, 1) for v in req_grid],
            "tok_s": [round(v / dt, 1) for v in tok_grid],
        },
        "durations": {
            "p50": pct(0.5),
            "p90": pct(0.9),
            "p99": pct(0.99),
            "max": durations[-1],
            "mean": statistics.mean(durations),
            # the ordered duration curve, thinned to a drawable number of points
            "curve": [round(durations[min(n - 1, int(n * i / 300))], 1) for i in range(301)],
        },
        "ep_wall": {"decode": dec_s, "tool": tool_s, "other": ep_wall - dec_s - tool_s},
        "req_time": {"decode": dec_s, "prefill": pre_s, "queue": q_s},
        "rates": {
            "req_s": len(requests) / span,
            "ep_min": n / (span / 60),
            "tok_s": out_tok / span,
            "tok_per_req": out_tok / len(requests),
            "mean_inflight": dec_s / span,
            "turns": statistics.mean([e.get("num_turns", 0) for e in episodes]),
            "reward": statistics.mean([e.get("reward", 0.0) or 0.0 for e in episodes]),
            "resolved": sum(1 for e in episodes if e.get("resolved")) / n,
        },
        "trace": trace,
        "exits": sorted(exits.items(), key=lambda kv: -kv[1]),
        "tools": sorted(tool_mix.values(), key=lambda r: -r["calls"]),
    }


def run_dir(directory: str) -> str:
    """The run's own ``analysis/<experiment>`` directory: the parent of the shard dir.

    Not ``realpath``: the shard dir is usually a symlink onto a data volume, and
    the report belongs next to ``run.log`` / ``metrics.csv`` / ``profile.txt``.
    """
    return os.path.dirname(os.path.abspath(directory.rstrip("/")))


def experiment_name(directory: str) -> str:
    """The run id. Shards usually live in ``<data volume>/<experiment>``, reached
    through the ``analysis/<experiment>/timeline`` symlink -- so resolve the link
    and take that directory's own name, falling back to its parent when the
    shards sit in a directory literally called ``timeline``."""
    real = os.path.realpath(directory.rstrip("/"))
    name = os.path.basename(real)
    return os.path.basename(os.path.dirname(real)) if name == "timeline" else name


def build(directory: str, experiment: str | None = None) -> dict:
    by_traj, train = load(directory)
    gens = [e for e in train if e["name"] in ROLLOUT_PHASES]
    if not gens:
        raise SystemExit(f"no trainer rollout spans in {directory}; nothing to attribute episodes to")
    by_step: dict = collections.defaultdict(dict)
    for e in train:
        if e.get("t_end"):
            by_step[e.get("step")][e["name"]] = {"dur": e.get("dur", 0.0), "t": e["t"], "t_end": e["t_end"]}
    steps, instances, tool_names = [], [], []
    for gen in gens:
        payload = step_payload(by_traj, gen, by_step.get(gen.get("step"), {}), instances, tool_names)
        if payload:
            steps.append(payload)
    return {
        "experiment": experiment or experiment_name(directory),
        "kinds": KINDS,
        "instances": instances,
        "tool_names": tool_names,
        "t_start": min(e["t"] for e in train),
        "t_end": max(e.get("t_end") or e["t"] for e in train),
        "steps": steps,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="timeline shard directory (analysis/<run>/timeline)")
    ap.add_argument("--experiment", default=None, help="run id shown in the title (default: inferred)")
    ap.add_argument("--out", default=None,
                    help="output HTML (default: timeline.html in the run dir, i.e. <dir>/..)")
    args = ap.parse_args()
    data = build(args.dir, args.experiment)
    tmpl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "timeline_report.html")
    with open(tmpl) as fh:
        html = fh.read()
    html = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    out = args.out or os.path.join(run_dir(args.dir), "timeline.html")
    with open(out, "w") as fh:
        fh.write(html)
    print(f"wrote {out} ({os.path.getsize(out)/1e3:.0f} kB, {len(data['steps'])} steps)")


if __name__ == "__main__":
    main()
