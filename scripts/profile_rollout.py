#!/usr/bin/env python
"""Baseline profile of a rollout phase, from the timeline shards of a run.

Answers the question the per-step metrics deliberately do not: *what shape is
this workload*. One table per training step, built from the ``generate`` events'
server timestamps (in-flight request concurrency) and the ``episode`` envelopes
(wall clock, waves).

    python scripts/profile_rollout.py --dir analysis/<experiment>/timeline

The in-flight bands are the load regimes the inference server actually runs in.
Read the table as "the rollout spends X% of its wall clock and produces Y% of
its tokens at this concurrency, at Z ms per token per request". Measured on run
20260906-013757 (three steps, batch 2048 over 264 container slots), and stable
across all three:

* **86% of the wall clock and 91% of the tokens are produced at 201+ in-flight.**
  The rollout is a plateau, not a ramp: the batch is ~7.8 episodes deep per slot,
  so admissions keep the server saturated until the queue drains.
* **Per-request decode latency is 5x better at low concurrency** (16 ms/token at
  1-8 in-flight vs 81 ms at 201+) **and cluster throughput is 50x worse**
  (63 tok/s vs 3,100). That is the batching trade-off, and the 16.3 ms is the bar
  any single-request-optimised backend has to beat -- it is what SGLang already
  delivers when the server is nearly empty.
* **The low-concurrency region is thin**: 1-8 in-flight is 2-4% of the span and
  0.1% of the tokens.
* A request is 97% decode (prefill 1.5%, queueing 1.5% -- the batch is
  admission-limited, not server-limited). An episode is ~95% decode, ~3% tool.
* Throughput, on one TP=8 replica over 8 GPUs: **2.8-2.9k decoded tok/s**, 18.6
  requests/s, 52 episodes/min, ~150 output tokens per request. Prefill is
  invisible by comparison only because of the prefix cache: 0.52 **G** prompt
  tokens are re-sent per step (every turn resends the transcript) and 95.1% of
  them are cache hits, leaving ~11k tok/s of real prefill.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import glob
import json
import os
import statistics

# In-flight request bands. Chosen to separate the regimes, not the quantiles:
# empty, single-digit (where a bs=1 backend would live), the ramp, and the
# saturated plateau.
BANDS = [(0, 0), (1, 8), (9, 32), (33, 64), (65, 128), (129, 200), (201, 10**9)]


def band_of(c: int) -> tuple[int, int]:
    for lo, hi in BANDS:
        if lo <= c <= hi:
            return (lo, hi)
    return BANDS[-1]


def band_label(b: tuple[int, int]) -> str:
    lo, hi = b
    return str(lo) if lo == hi else (f"{lo}-{hi}" if hi < 10**9 else f"{lo}+")


def load(directory: str):
    """Group every timeline event by trajectory; keep the trainer's gen spans."""
    by_traj: dict[str, list[dict]] = collections.defaultdict(list)
    gens: list[dict] = []
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
                    if e["name"] == "gen":
                        gens.append(e)
                elif e.get("traj"):
                    by_traj[e["traj"]].append(e)
    return by_traj, sorted(gens, key=lambda e: e["t"])


def profile_step(by_traj, gen) -> None:
    t0, t1 = gen["t"], gen["t"] + gen.get("dur", 0.0)
    episodes, requests, tool_s = [], [], 0.0
    for events in by_traj.values():
        env = next((e for e in events if e["name"] == "episode" and e.get("t_end")), None)
        if env is None or not (t0 - 1 <= env["t"] <= t1 + 1):
            continue
        episodes.append((env["t"], env["t_end"]))
        for e in events:
            if e["name"] == "generate" and e.get("decode_start") and e.get("decode_finished"):
                requests.append(
                    {
                        "busy": e.get("request_scheduled") or e["decode_start"],  # GPU work starts
                        "d0": e["decode_start"],
                        "d1": e["decode_finished"],
                        "out": e.get("completion_tokens", 0),
                        "prompt": e.get("prompt_tokens", 0),
                        "cached": e.get("cached_tokens", 0),
                        "prefill_s": e.get("prefill_s", 0.0),
                        "queue_s": e.get("queue_s", 0.0),
                    }
                )
            elif e["name"] == "tool_call":
                tool_s += e.get("dur", 0.0)
    if not episodes or not requests:
        return

    start, end = min(a for a, _ in episodes), max(b for _, b in episodes)
    span = end - start

    # In-flight concurrency: +1 when a request starts occupying the GPU, -1 when
    # its last token is sampled.
    times, conc, running = [], [], 0
    for t, delta in sorted([(r["busy"], 1) for r in requests] + [(r["d1"], -1) for r in requests]):
        running += delta
        times.append(t)
        conc.append(running)

    wall = collections.Counter()
    for i in range(len(times) - 1):
        dt = times[i + 1] - times[i]
        if dt > 0:
            wall[band_of(conc[i])] += dt
    tokens, decode, count = collections.Counter(), collections.Counter(), collections.Counter()
    for r in requests:
        i = bisect.bisect_right(times, r["d0"]) - 1
        b = band_of(conc[i] if i >= 0 else 0)
        tokens[b] += r["out"]
        decode[b] += r["d1"] - r["d0"]
        count[b] += 1

    total_tok = sum(tokens.values()) or 1
    print(
        f"\n=== step {gen.get('step')}: span {span:.0f}s | {len(episodes)} episodes | "
        f"{len(requests):,} requests | {total_tok/1e6:.2f}M decoded tokens | "
        f"peak in-flight {max(conc)}"
    )
    print(
        f"{'in-flight':>12} {'wall s':>8} {'% span':>7} {'requests':>9} {'tokens':>11} "
        f"{'% tok':>6} {'ms/tok/req':>11} {'cluster tok/s':>13}"
    )
    for b in BANDS:
        if not wall[b] and not count[b]:
            continue
        ms = (decode[b] / tokens[b] * 1000) if tokens[b] else 0.0
        print(
            f"{band_label(b):>12} {wall[b]:8.0f} {wall[b]/span:7.1%} {count[b]:9,d} "
            f"{tokens[b]:11,d} {tokens[b]/total_tok:6.1%} {ms:11.1f} "
            f"{(tokens[b]/wall[b] if wall[b] else 0):13,.0f}"
        )

    durations = sorted(b - a for a, b in episodes)
    q = sum(r["queue_s"] for r in requests)
    pre = sum(r["prefill_s"] for r in requests)
    dec = sum(r["d1"] - r["d0"] for r in requests)
    ep_wall = sum(durations)

    # Throughput, cluster-wide over the rollout span. "prompt seen" is nominal:
    # multi-turn rollout re-sends the whole transcript every turn, so it is
    # enormous and mostly served from the prefix cache -- the uncached remainder
    # is the prefill the GPU actually runs.
    out_tok = sum(r["out"] for r in requests)
    prompt_tok = sum(r["prompt"] for r in requests)
    cached_tok = sum(r["cached"] for r in requests)
    uncached = prompt_tok - cached_tok
    print(
        f"  throughput: decode {out_tok/span:,.0f} tok/s ({out_tok/1e6:.2f}M over the span) | "
        f"prefill {uncached/span:,.0f} tok/s uncached "
        f"({prompt_tok/1e6:,.0f}M prompt seen, {cached_tok/prompt_tok:.1%} from the prefix cache)"
        if prompt_tok
        else f"  throughput: decode {out_tok/span:,.0f} tok/s"
    )
    print(
        f"  rate: {len(requests)/span:.1f} requests/s, {len(episodes)/(span/60):.1f} episodes/min, "
        f"{out_tok/len(requests):.0f} output tokens/request | mean in-flight "
        f"{dec/span:.0f} decoding + {pre/span:.1f} prefilling"
    )
    print(
        f"  episode duration  p50 {durations[len(durations)//2]:.0f}s  "
        f"p90 {durations[int(len(durations)*0.9)]:.0f}s  "
        f"p99 {durations[int(len(durations)*0.99)]:.0f}s  max {durations[-1]:.0f}s  "
        f"mean {statistics.mean(durations):.0f}s"
    )
    print(
        f"  request time: decode {dec/(dec+q+pre):.1%}, prefill {pre/(dec+q+pre):.1%}, "
        f"queue {q/(dec+q+pre):.1%}   |   episode wall: decode {dec/ep_wall:.1%}, "
        f"tool {tool_s/ep_wall:.1%}, other {1-(dec+tool_s)/ep_wall:.1%}"
    )
    # Wave depth: how many episodes each slot runs back to back. It is what makes
    # the drain thin -- a one-wave batch would expose the whole duration spread.
    slots = max(_concurrency_peak(episodes), 1)
    print(
        f"  waves: {len(episodes)} episodes over {slots} container slots = "
        f"{len(episodes)/slots:.1f} deep"
    )


def _concurrency_peak(intervals: list[tuple[float, float]]) -> int:
    running = peak = 0
    for _, delta in sorted([(a, 1) for a, _ in intervals] + [(b, -1) for _, b in intervals]):
        running += delta
        peak = max(peak, running)
    return peak


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="timeline shard directory (analysis/<run>/timeline)")
    args = ap.parse_args()
    by_traj, gens = load(args.dir)
    if not gens:
        raise SystemExit(f"no trainer 'gen' spans in {args.dir}; nothing to attribute episodes to")
    for gen in gens:
        profile_step(by_traj, gen)


if __name__ == "__main__":
    main()
