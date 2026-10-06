#!/usr/bin/env python
"""削 trace: strip the run's fullest trace one rung at a time and see what the adapters lose.

This is the acceptance test of the adapter layer in ``SIMULATOR.md`` §15, step 1.
One run that recorded everything (timeline shards *and* the conversation dump) is
read five ways, each rung keeping less than the one above:

  R0  full timeline              every duration and token count traced; answers in observed
  R1  task durations stripped    ``dur`` / ``t_end`` gone from tool_call, container_start,
                                 score, patch_recover, cleanup -> estimated from priors
  R2  R1 + usage stripped        ``prompt_tokens`` / ``completion_tokens`` gone from generate
  R3  R2 + timestamps stripped   no ``t`` / ``t_end`` anywhere, no server timestamps
  R4  conversation only          ``messages-<pid>.jsonl`` and nothing else

Three questions per rung. Does the adapter still produce a valid workload, or
refuse (the honest answer when it cannot, and recorded as such)? Realised N times
under N seeds, is the p10-p90 of each aggregate at least as wide as the rung
above? Does it cover R0's point?

There is no engine yet, so the aggregates are what the IR alone determines:
task-seconds by kind and in total, the per-session serial task time (p50 / p90 /
max across sessions), and the token totals. They are exactly what the engine
will be fed, so an interval that misses here misses downstream too.

Priors come from a different run (``--prior-run``), as ``check`` insists. The
scale on each prior (how far off the whole distribution may be) is the spread of
per-knob *mean* durations across the runs in ``--scale-runs``, relative to the
prior run, widened to include 1.0. Means, not medians, because the quantities
checked are sums.

R4 cannot tell which training step an episode belonged to (no timestamps), so
its input is the conversation dump filtered to the sessions R0 holds. The
unfiltered count is reported next to it: that is what text alone would include.

    python scripts/degrade_trace.py --run 20261006-030010 --step 1 \
        --prior-run 20261005-113503 --scale-runs 20261005-143426 20260912-072938
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from simulator.adapters import check, priors_from_workload  # noqa: E402
from simulator.adapters.agentic_messages import AgenticMessagesAdapter  # noqa: E402
from simulator.adapters.agentic_timeline import AgenticTimelineAdapter  # noqa: E402
from simulator.adapters.base import AdapterError, Conversion, Prior, Priors  # noqa: E402
from simulator.ir import IRError, Uniform, Workload, validate  # noqa: E402
from simulator.realise import knobs, realise  # noqa: E402

USER = os.environ.get("USER", "nobody")
TASK_EVENTS = ("tool_call", "container_start", "score", "patch_recover", "cleanup")
TIME_FIELDS = ("t", "t_end", "dur", "request_received", "request_scheduled", "decode_start",
               "decode_finished", "prefill_finished", "cached_tokens")
KINDS = ("tool", "container_start", "eval", "patch_recover", "cleanup")
METRICS = ("task_s/total", *(f"task_s/{k}" for k in KINDS),
           "session_task_s/p50", "session_task_s/p90", "session_task_s/max",
           "tokens/prompt", "tokens/completion", "tokens/unique_prefill", "tokens/prefix")

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# stripping a timeline
# ---------------------------------------------------------------------------
def rewrite(src: Path, dst: Path, edit) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    for shard in sorted(src.glob("timeline-*.jsonl")):
        with open(shard) as fin, open(dst / shard.name, "w") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fout.write(json.dumps(edit(ev)) + "\n")
    return dst


def _drop(ev: dict, names, fields) -> dict:
    if names is None or ev.get("name") in names:
        for f in fields:
            ev.pop(f, None)
    return ev


def edit_r1(ev: dict) -> dict:
    return _drop(ev, TASK_EVENTS, ("dur", "t_end"))


def edit_r2(ev: dict) -> dict:
    return _drop(edit_r1(ev), ("generate", "episode"), ("prompt_tokens", "completion_tokens"))


def edit_r3(ev: dict) -> dict:
    return _drop(edit_r2(ev), None, TIME_FIELDS)


def filter_messages(src: Path, dst: Path, keep: set[str]) -> tuple[int, int]:
    """Copy the conversation dump keeping only episodes whose traj id is in ``keep``."""
    dst.mkdir(parents=True, exist_ok=True)
    total = kept = 0
    for f in sorted(src.glob("messages-*.jsonl")):
        with open(f) as fin, open(dst / f.name, "w") as fout:
            for line in fin:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("kind") == "meta":
                    fout.write(line if line.endswith("\n") else line + "\n")
                elif r.get("kind") == "episode":
                    total += 1
                    if r.get("traj") in keep:
                        kept += 1
                        fout.write(line if line.endswith("\n") else line + "\n")
    return total, kept


# ---------------------------------------------------------------------------
# priors
# ---------------------------------------------------------------------------
def load_timeline(path: Path, step: int, priors: Priors) -> Conversion:
    conv = AgenticTimelineAdapter(step=step).convert(path, priors)
    check(conv, priors)
    return conv


def _mean(xs) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _median(xs) -> float:
    return _q(sorted(xs), 0.5) if xs else float("nan")


def build_priors(prior: Conversion, scale_convs: dict[str, Conversion]) -> tuple[Priors, list[dict]]:
    base = priors_from_workload(prior.workload)
    others = {run: priors_from_workload(c.workload) for run, c in scale_convs.items()}
    out, table = [], []
    for knob, p in sorted(base.by_knob.items()):
        m0 = _mean(p.dist.values)
        ratios = {run: _mean(o.by_knob[knob].dist.values) / m0
                  for run, o in others.items() if knob in o.by_knob and m0 > 0}
        lo, hi = min([1.0, *ratios.values()]), max([1.0, *ratios.values()])
        scale = Uniform(lo, hi) if hi > lo else None
        out.append(Prior(knob=knob, dist=p.dist, source=p.source, scale=scale))
        table.append({"knob": knob, "n": len(p.dist.values), "mean_s": m0,
                      "median_s": _median(p.dist.values),
                      "mean_ratio": {r: round(v, 3) for r, v in ratios.items()},
                      "scale": [lo, hi] if scale else None})
    return Priors.of(*out), table


def without_scale(w: Workload) -> Workload:
    def fix(obj):
        if not obj.unknown:
            return obj
        return replace(obj, unknown={k: replace(u, scale=None) for k, u in obj.unknown.items()})
    sessions = tuple(replace(s, requests=tuple(fix(r) for r in s.requests),
                             tasks=tuple(fix(t) for t in s.tasks)) for s in w.sessions)
    return replace(w, sessions=sessions, prefixes=tuple(fix(p) for p in w.prefixes))


# ---------------------------------------------------------------------------
# aggregates and intervals
# ---------------------------------------------------------------------------
def _q(xs_sorted, p: float) -> float:
    if not xs_sorted:
        return float("nan")
    pos = p * (len(xs_sorted) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs_sorted) - 1)
    return xs_sorted[lo] + (xs_sorted[hi] - xs_sorted[lo]) * (pos - lo)


def aggregates(w: Workload) -> dict[str, float]:
    by_kind: Counter = Counter()
    per_session = []
    prompt = completion = unique = 0
    for s in w.sessions:
        tot = 0.0
        for t in s.tasks:
            by_kind[t.kind] += t.duration_s
            tot += t.duration_s
        per_session.append(tot)
        for r in s.requests:
            prompt += r.prompt_tokens
            completion += r.completion_tokens
            unique += r.prompt_tokens - r.shared_prefix_tokens
    per_session.sort()
    out = {"task_s/total": float(sum(by_kind.values()))}
    for k in KINDS:
        out[f"task_s/{k}"] = float(by_kind.get(k, 0.0))
    out["session_task_s/p50"] = _q(per_session, 0.5)
    out["session_task_s/p90"] = _q(per_session, 0.9)
    out["session_task_s/max"] = per_session[-1] if per_session else float("nan")
    out["tokens/prompt"] = float(prompt)
    out["tokens/completion"] = float(completion)
    out["tokens/unique_prefill"] = float(unique)
    out["tokens/prefix"] = float(sum(p.tokens for p in w.prefixes))
    return out


def interval(w: Workload, seeds: int, label: str) -> dict[str, tuple[float, float, float]]:
    """p10 / p50 / p90 of every aggregate over ``seeds`` realisations; a point when nothing is unknown."""
    if not knobs(w):
        a = aggregates(w)
        return {k: (v, v, v) for k, v in a.items()}
    samples: dict[str, list[float]] = defaultdict(list)
    for seed in range(seeds):
        rw = realise(w, seed=seed)
        if seed == 0:
            validate(rw)
        for k, v in aggregates(rw).items():
            samples[k].append(v)
        if seed % 8 == 7:
            log(f"  {label}: {seed + 1}/{seeds} realisations")
    return {k: (_q(sorted(v), 0.1), _q(sorted(v), 0.5), _q(sorted(v), 0.9)) for k, v in samples.items()}


def structure(conv: Conversion) -> dict:
    w, o = conv.workload, conv.observations
    kinds: Counter = Counter(t.kind for s in w.sessions for t in s.tasks)
    prov: Counter = Counter()
    for s in w.sessions:
        for n in s.nodes():
            for f, src in n.prov.items():
                prov[f"{f}:{src}"] += 1
    return {"sessions": len(w.sessions), "groups": len({s.group for s in w.sessions}),
            "requests": sum(len(s.requests) for s in w.sessions),
            "tasks": dict(sorted(kinds.items())), "prefixes": len(w.prefixes),
            "estimated_fields": dict(sorted(knobs(w).items())),
            "prov": dict(sorted(prov.items())),
            "observed_requests": len(o.requests), "observed_sessions": len(o.sessions),
            "observed_phases": len(o.phases), "notes": list(conv.notes)}


# ---------------------------------------------------------------------------
# R4 against R0, session by session
# ---------------------------------------------------------------------------
def compare_sessions(r0: Workload, r4: Workload) -> dict:
    a = {s.id: s for s in r0.sessions}
    b = {s.id: s for s in r4.sessions}
    both = sorted(set(a) & set(b))
    out = {"in_both": len(both), "only_full": len(set(a) - set(b)), "only_text": len(set(b) - set(a))}
    same_shape = same_turns = same_tools = 0
    task_delta: Counter = Counter()
    comp_diff: Counter = Counter()
    prompt_exact = prompt_first_exact = prefix_exact = n_req = 0
    comp0 = comp4 = prompt0 = prompt4 = 0
    for sid in both:
        s0, s4 = a[sid], b[sid]
        k0 = Counter(t.kind for t in s0.tasks)
        k4 = Counter(t.kind for t in s4.tasks)
        if len(s0.requests) == len(s4.requests):
            same_turns += 1
        if k0["tool"] == k4["tool"]:
            same_tools += 1
        if len(s0.requests) == len(s4.requests) and k0 == k4:
            same_shape += 1
        for k in set(k0) | set(k4):
            if k0[k] != k4[k]:
                task_delta[f"{k}:{k4[k] - k0[k]:+d}"] += 1
        if s0.requests and s4.requests:
            if s0.requests[0].shared_prefix_tokens == s4.requests[0].shared_prefix_tokens:
                prefix_exact += 1
            if s0.requests[0].prompt_tokens == s4.requests[0].prompt_tokens:
                prompt_first_exact += 1
        for q0, q4 in zip(s0.requests, s4.requests):
            n_req += 1
            comp_diff[q4.completion_tokens - q0.completion_tokens] += 1
            prompt_exact += q0.prompt_tokens == q4.prompt_tokens
            comp0 += q0.completion_tokens
            comp4 += q4.completion_tokens
            prompt0 += q0.prompt_tokens
            prompt4 += q4.prompt_tokens
    out.update({
        "same_shape": same_shape, "same_turn_count": same_turns, "same_tool_count": same_tools,
        "task_count_deltas": dict(task_delta.most_common(12)),
        "requests_paired": n_req,
        "completion_exact": comp_diff[0], "completion_diff_hist": dict(sorted(comp_diff.items())),
        "prompt_exact": prompt_exact, "first_prompt_exact": prompt_first_exact,
        "prefix_exact": prefix_exact,
        "completion_total_rel_err": (comp4 - comp0) / comp0 if comp0 else float("nan"),
        "prompt_total_rel_err": (prompt4 - prompt0) / prompt0 if prompt0 else float("nan"),
    })
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _fmt(v: float, metric: str) -> str:
    if metric.startswith("tokens/"):
        return f"{v / 1e6:.3f}M"
    if metric.startswith("task_s/"):
        return f"{v / 3600:.2f}h"
    return f"{v:.1f}s"


def _covers(point: float, lohi: tuple[float, float, float]) -> str:
    lo, _, hi = lohi
    if lo == hi:
        return "point" if abs(lo - point) <= 1e-9 * max(1.0, abs(point)) else f"off {100 * (lo - point) / point:+.1f}%"
    return "yes" if lo <= point <= hi else ("below" if hi < point else "above")


def report(args, prior_table, rungs, r4_filter, cmp) -> str:
    L = []
    L.append(f"# 削 trace: run {args.run} step {args.step}\n")
    L.append(f"prior run {args.prior_run} step {args.prior_step}; scale from "
             f"{', '.join(args.scale_runs) or 'nothing (no scale)'}; {args.seeds} realisations per rung.\n")
    L.append("## Priors\n")
    L.append("| knob | n | mean | median | mean ratio vs prior (other runs) | scale | this run's mean ratio (the answer) |")
    L.append("|---|---:|---:|---:|---|---|---:|")
    for row in prior_table:
        sc = row['scale'] and f"[{row['scale'][0]:.3f}, {row['scale'][1]:.3f}]"
        L.append(f"| {row['knob']} | {row['n']} | {row['mean_s']:.2f}s | {row['median_s']:.2f}s | "
                 f"{row['mean_ratio']} | {sc} | {row.get('this_run_mean_ratio')} |")
    L.append("\n## Rungs\n")
    L.append("| rung | trace still has | result | sessions | requests | tasks | estimated fields | observed req/sess | wall |")
    L.append("|---|---|---|---:|---:|---:|---|---|---:|")
    for r in rungs:
        if r["status"] != "ok":
            L.append(f"| {r['name']} | {r['label']} | **refused**: {r['detail']} | | | | | | {r['wall_s']:.0f}s |")
            continue
        st = r["structure"]
        L.append(f"| {r['name']} | {r['label']} | workload valid | {st['sessions']} | {st['requests']} | "
                 f"{sum(st['tasks'].values())} | {st['estimated_fields'] or '-'} | "
                 f"{st['observed_requests']}/{st['observed_sessions']} | {r['wall_s']:.0f}s |")
    L.append(f"\nR4 input: conversation dump holds {r4_filter[0]} episodes; {r4_filter[1]} are in R0's step and were kept. "
             f"Text alone cannot make that cut.\n")
    full = next(r for r in rungs if r["name"] == "R0")
    for r in rungs:
        if r["status"] != "ok" or r["name"] == "R0":
            continue
        L.append(f"## {r['name']} ({r['label']}) against R0\n")
        L.append("| metric | R0 point | p10 | p50 | p90 | covers R0? | width/point | no-scale p10–p90 | covers? |")
        L.append("|---|---:|---:|---:|---:|---|---:|---|---|")
        for m in METRICS:
            p = full["scaled"][m][1]
            lo, mid, hi = r["scaled"][m]
            ulo, _, uhi = r["unscaled"][m]
            width = (hi - lo) / p if p else float("nan")
            L.append(f"| {m} | {_fmt(p, m)} | {_fmt(lo, m)} | {_fmt(mid, m)} | {_fmt(hi, m)} | "
                     f"{_covers(p, r['scaled'][m])} | {100 * width:.1f}% | "
                     f"{_fmt(ulo, m)}–{_fmt(uhi, m)} | {_covers(p, r['unscaled'][m])} |")
        L.append("")
    if cmp:
        L.append("## R4 against R0, session by session\n")
        n = cmp["in_both"] or 1
        L.append(f"- sessions in both: {cmp['in_both']} (only in R0: {cmp['only_full']}, only in R4: {cmp['only_text']})")
        L.append(f"- same DAG shape (turns and task kinds): {cmp['same_shape']} ({100 * cmp['same_shape'] / n:.1f}%); "
                 f"same turn count {cmp['same_turn_count']}, same tool-call count {cmp['same_tool_count']}")
        L.append(f"- task count deltas (text minus timeline, sessions): {cmp['task_count_deltas'] or 'none'}")
        q = cmp["requests_paired"] or 1
        L.append(f"- completion tokens exact: {cmp['completion_exact']}/{cmp['requests_paired']} "
                 f"({100 * cmp['completion_exact'] / q:.1f}%); total off by {100 * cmp['completion_total_rel_err']:+.2f}%")
        L.append(f"- prompt tokens exact: {cmp['prompt_exact']}/{cmp['requests_paired']} "
                 f"({100 * cmp['prompt_exact'] / q:.1f}%); first prompt exact {cmp['first_prompt_exact']}/{cmp['in_both']}; "
                 f"group prefix exact {cmp['prefix_exact']}/{cmp['in_both']}; total off by {100 * cmp['prompt_total_rel_err']:+.2f}%")
        hist = cmp["completion_diff_hist"]
        tail = {k: v for k, v in hist.items() if k != 0}
        L.append(f"- completion diff histogram (text minus timeline, non-zero): {tail}")
    L.append("\n## Notes per rung\n")
    for r in rungs:
        if r["status"] == "ok":
            L.append(f"**{r['name']}**")
            for nte in r["structure"]["notes"]:
                L.append(f"- {nte}")
            L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--prior-run", required=True)
    ap.add_argument("--prior-step", type=int, default=1)
    ap.add_argument("--scale-runs", nargs="*", default=[], help="run ids whose step --prior-step gives the scale")
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--group-size", type=int, default=8, help="rollout.n, for the text-only adapter")
    ap.add_argument("--timelines", type=Path, default=Path(f"/data0/shared/{USER}/agentic-timelines"))
    ap.add_argument("--trajectories", type=Path, default=Path(f"/data0/shared/{USER}/agentic-trajectories"))
    ap.add_argument("--work", type=Path, required=True, help="where the stripped copies go")
    ap.add_argument("--out", type=Path, required=True, help="report.md and result.json land here")
    ap.add_argument("--skip-text", action="store_true", help="leave out R4 (no tokenizer run)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tl = args.timelines / args.run

    log(f"prior run {args.prior_run} step {args.prior_step}")
    prior_conv = load_timeline(args.timelines / args.prior_run, args.prior_step, Priors())
    scale_convs = {}
    for run in args.scale_runs:
        log(f"scale run {run} step {args.prior_step}")
        scale_convs[run] = load_timeline(args.timelines / run, args.prior_step, Priors())
    priors, prior_table = build_priors(prior_conv, scale_convs)
    for row in prior_table:
        log(f"  prior {row['knob']}: n={row['n']} mean={row['mean_s']:.2f}s ratios={row['mean_ratio']} scale={row['scale']}")

    rungs: list[dict] = []

    def run_rung(name: str, label: str, convert) -> Conversion | None:
        t = time.time()
        log(f"{name}: {label}")
        try:
            conv = convert()
        except (AdapterError, IRError) as exc:
            msg = str(exc).split("\n")[0]
            log(f"  refused: {msg}")
            rungs.append({"name": name, "label": label, "status": "refused", "detail": msg,
                          "wall_s": time.time() - t})
            return None
        st = structure(conv)
        log(f"  {st['sessions']} sessions, {st['requests']} requests, {sum(st['tasks'].values())} tasks, "
            f"estimated {st['estimated_fields']}")
        scaled = interval(conv.workload, args.seeds, name)
        unscaled = interval(without_scale(conv.workload), args.seeds, name + "/no-scale") \
            if knobs(conv.workload) else scaled
        rungs.append({"name": name, "label": label, "status": "ok", "structure": st,
                      "scaled": scaled, "unscaled": unscaled, "wall_s": time.time() - t})
        return conv

    r0 = run_rung("R0", "full timeline", lambda: load_timeline(tl, args.step, priors))
    if r0 is None:
        print("R0 did not convert; nothing to degrade", file=sys.stderr)
        return 2
    keep = {s.id for s in r0.workload.sessions}
    own = priors_from_workload(r0.workload)
    for row in prior_table:
        p = own.by_knob.get(row["knob"])
        row["this_run_mean_ratio"] = round(_mean(p.dist.values) / row["mean_s"], 3) if p and row["mean_s"] else None

    run_rung("R1", "task durations stripped",
             lambda: load_timeline(rewrite(tl, args.work / "r1", edit_r1), args.step, priors))
    run_rung("R2", "R1 + usage stripped",
             lambda: load_timeline(rewrite(tl, args.work / "r2", edit_r2), args.step, priors))
    run_rung("R3", "R2 + timestamps stripped",
             lambda: load_timeline(rewrite(tl, args.work / "r3", edit_r3), args.step, priors))

    r4_filter, cmp = (0, 0), None
    if not args.skip_text:
        log("R4: filtering the conversation dump to R0's sessions")
        r4_filter = filter_messages(args.trajectories / args.run, args.work / "r4", keep)
        log(f"  {r4_filter[1]} of {r4_filter[0]} episodes kept")

        def convert_text():
            adapter = AgenticMessagesAdapter(group_size=args.group_size)
            conv = adapter.convert(args.work / "r4", priors)
            check(conv, priors)
            return conv

        r4 = run_rung("R4", "conversation only", convert_text)
        if r4 is not None:
            cmp = compare_sessions(r0.workload, r4.workload)

    text = report(args, prior_table, rungs, r4_filter, cmp)
    (args.out / "report.md").write_text(text)
    (args.out / "result.json").write_text(json.dumps(
        {"args": {k: str(v) for k, v in vars(args).items()}, "priors": prior_table, "rungs": rungs,
         "r4_filter": r4_filter, "r4_vs_r0": cmp}, indent=1, default=str))
    print(text)
    log(f"done; report at {args.out / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
