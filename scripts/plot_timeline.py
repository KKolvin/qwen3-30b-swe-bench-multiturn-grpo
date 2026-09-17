#!/usr/bin/env python
"""Render a run's timeline shards as an HTML report.

    python scripts/plot_timeline.py --dir analysis/<experiment>/timeline

Companion to :mod:`scripts.profile_rollout`, which prints the same reductions as
text. This one draws them, as two views behind a nav bar:

* **overview** — where the step's wall clock goes, then one rollout phase at a
  time: the trajectory trace, how occupancy decays over the span, what
  throughput the server gets at each in-flight band, how long episodes take, and
  what the tool calls cost -- grouped by the program each call ran (``pytest``,
  ``find``, ``edit:view``), not by the tool that ran it, and sortable by total
  wall clock, call count or per-call latency.
* **metrics** — every per-step scalar as a number, not a plot: ``metrics.csv``
  (what the run printed to its log, so what W&B got) plus the timeline's own
  ``rollout/*`` reductions, in a browsable tag tree with METRICS.md's own
  one-line meaning against each tag.

One page per run, self-contained (no network), written next to the shards as
``timeline.html``.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import csv
import glob
import json
import os
import re
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


# ---------------------------------------------------------------------------
# what a tool call actually ran
# ---------------------------------------------------------------------------
# A timeline event's ``tool`` field names the registry entry -- ``bash``,
# ``str_replace_based_edit_tool``, ``submit`` -- which is three buckets for the
# ~100k calls in a step and says nothing about what any of them cost: a ``find``
# over /testbed and an ``ls`` are both "bash", a 0.15s ``view`` and a 300s test
# run are both "the edit tool" and "a tool call". What costs wall clock is the
# program the call actually ran, so that is the unit reported here: the edit
# tool's sub-command, or the first real program on the bash line.
EDIT_TOOL = "str_replace_based_edit_tool"

MAX_LABEL = 40  # a label is a group name, not the command; the trace hover has the rest

# Segment separators, matched outside quotes only (see :func:`shell_segments`).
_SEP = re.compile(r"&&|\|\||;|\||\n")
# ``DJANGO_SETTINGS_MODULE=settings python x.py`` runs python, not an assignment.
_ENV_ASSIGN = re.compile(r"^(?:\w+=(?:'[^']*'|\"[^\"]*\"|\S*)\s+)+")
# Prefixes that run something else; the something else is what took the time.
_WRAPPER = re.compile(
    r"^(?:sudo|nohup|command|exec|time|env|stdbuf(?:\s+-\S+)*"
    r"|timeout(?:\s+-\S+)*\s+[\d.]+[smhd]?)\s+"
)
# Segments that only move the shell around. ``cd /testbed && pytest`` is a
# pytest call; charging its 90 seconds to ``cd`` would be the whole point missed.
_NAVIGATION = {"cd", "export", "source", ".", "set", "unset", "pushd", "popd", "true", ":", "eval"}
_PYTHON = re.compile(r"^python[\d.]*$")


def shell_segments(command: str) -> list[str]:
    """Split on ``&&`` / ``||`` / ``;`` / ``|`` / newline, ignoring quoted text.

    Quote-aware rather than :mod:`re`-only because the model writes
    ``grep -r "a|b" .`` and ``find . -name '*.py;*'``: splitting those on the
    quoted separator would attribute the call to whatever follows the quote.
    """
    out: list[str] = []
    buf: list[str] = []
    quote = ""
    i = 0
    while i < len(command):
        ch = command[i]
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = ""
            elif ch == "\\" and quote == '"' and i + 1 < len(command):
                i += 1
                buf.append(command[i])
        elif ch in "'\"":
            quote = ch
            buf.append(ch)
        elif ch in "&|;\n":
            m = _SEP.match(command, i)
            if m:
                out.append("".join(buf))
                buf = []
                i = m.end()
                continue
            buf.append(ch)
        else:
            buf.append(ch)
        i += 1
    out.append("".join(buf))
    return [seg.strip() for seg in out if seg.strip()]


def segment_program(segment: str) -> str | None:
    """The program one shell segment runs, as a group label, or None if empty.

    ``python`` keeps its module or test runner (``python -m pytest``,
    ``manage.py test``, ``runtests.py``): those are different workloads with
    different timeouts, and collapsing them into one ``python`` row hides the
    only tool calls that legitimately run for minutes. A dotted module is cut at
    its first component, so a hundred one-off ``python -m test_pkg_dir.thing``
    calls group under one row instead of a hundred.
    """
    seg = segment.lstrip("({ \t")
    while True:
        stripped = _WRAPPER.sub("", _ENV_ASSIGN.sub("", seg))
        if stripped == seg:
            break
        seg = stripped
    tokens = seg.split()
    if not tokens:
        return None
    prog = os.path.basename(tokens[0].strip("'\"")) or tokens[0]
    if prog == "py.test":
        prog = "pytest"
    if _PYTHON.match(prog):
        rest = list(tokens[1:])
        while rest and rest[0].startswith("-") and rest[0] != "-m":
            rest.pop(0)
        if rest and rest[0] == "-m" and len(rest) > 1:
            return "python -m " + os.path.basename(rest[1]).split(".")[0]
        if rest:
            script = os.path.basename(rest[0])
            if script == "manage.py" and rest[1:2] == ["test"]:
                return "manage.py test"
            if script == "runtests.py":
                return "runtests.py"
        return "python"
    return prog


def call_action(event: dict) -> str:
    """The group a ``tool_call`` event belongs to: ``pytest``, ``edit:view``, ...

    Names the model invented are kept apart from real programs (``unknown:find``
    is the model calling a ``find`` *tool*, which returns an error observation in
    microseconds; ``find`` is the shell one, which can run for a minute).
    """
    tool = event.get("tool") or "?"
    command = event.get("command") or ""
    if tool == EDIT_TOOL:
        return "edit:" + (command.split(" ", 1)[0][:MAX_LABEL] or "?")
    if tool == "submit":
        return "submit"
    if tool != "bash":
        return "unknown:" + tool[:24]
    segments = shell_segments(command)
    for seg in segments:
        prog = segment_program(seg)
        if prog and prog not in _NAVIGATION:
            return prog[:MAX_LABEL]
    first = segment_program(segments[0]) if segments else None
    return (first or "?")[:MAX_LABEL]


def action_rows(events: list[dict]) -> list[dict]:
    """Per-action call count, wall clock and per-call distribution, worst first.

    Total and per-call are different questions with different answers -- 39k
    ``edit:view`` calls at 0.18s cost more than 500 ``pytest`` calls at 0.6s,
    and only one of the two is worth making faster -- so both are reported and
    the table is sortable by either.
    """
    by_action: dict[str, dict] = {}
    for e in events:
        action = call_action(e)
        row = by_action.setdefault(action, {"action": action, "tool": e.get("tool", "?"), "d": [], "err": 0})
        row["d"].append(e.get("dur", 0.0))
        if e.get("returncode") not in (0, None):
            row["err"] += 1
    total = sum(sum(r["d"]) for r in by_action.values()) or 1.0
    rows = []
    for r in by_action.values():
        d = sorted(r["d"])
        n = len(d)
        rows.append({
            "action": r["action"],
            # Which registry entry ran it; a bash row and an ``unknown:`` row can
            # carry the same program name and are not the same call.
            "tool": "edit" if r["tool"] == EDIT_TOOL else r["tool"],
            "calls": n,
            "s": round(sum(d), 2),
            "pct": sum(d) / total,
            "mean": sum(d) / n,
            "p50": d[n // 2],
            "p90": d[min(n - 1, int(n * 0.9))],
            "max": d[-1],
            "err": r["err"],
        })
    return sorted(rows, key=lambda r: -r["s"])


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


def trace_row(events, env, start: float, instances: list, actions: list) -> dict:
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
    flat, action_codes, cur = [], [], env["t"]

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
            name = call_action(e)
            if name not in actions:
                actions.append(name)
            action_codes.append(actions.index(name))
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
        "tt": action_codes,
    }


# Occupancy regions. A rollout phase is not one regime: it fills, holds at the
# concurrency the container cap allows, drains once the admission queue is empty
# and nothing refills a freed slot, and then sits at zero requests while the
# grading containers and verl finish. Only the last three cost anything, and
# naming them separately is what makes the cost attributable.
PLATEAU_FRAC = 0.9  # "at capacity" = within 10% of the phase's peak in-flight


def occupancy_regions(times, conc, start: float, gen_end: float,
                      grade_end: float | None) -> dict:
    """Split the phase into ramp / plateau / drain / grading / assembly.

    ``times``/``conc`` are the in-flight step function (``conc[i]`` holds from
    ``times[i]`` to ``times[i+1]``). Each region carries the slot-seconds lost
    against the peak, which is what makes two regions comparable: a short region
    at zero occupancy can cost more than a long one that is merely off-peak.
    """
    peak = max(conc) if conc else 0
    if not peak:
        return {}

    def lost(a: float, b: float) -> float:
        """Slot-seconds below peak over [a, b), the step function integrated."""
        total = 0.0
        for i, t in enumerate(times[:-1]):
            lo, hi = max(t, a), min(times[i + 1], b)
            if hi > lo:
                total += (peak - conc[i]) * (hi - lo)
        # past the last event the engine is empty, so the whole peak is lost
        tail_lo, tail_hi = max(times[-1], a), min(gen_end, b)
        if tail_hi > tail_lo:
            total += peak * (tail_hi - tail_lo)
        return total

    hi = [i for i, c in enumerate(conc) if c >= PLATEAU_FRAC * peak]
    p0 = times[hi[0]] if hi else start
    p1 = times[hi[-1]] if hi else start
    busy_end = times[-1]  # last token of the last request
    g_end = min(max(grade_end or busy_end, busy_end), gen_end)

    cuts = [("ramp", start, p0), ("plateau", p0, p1), ("drain", p1, busy_end),
            ("grading", busy_end, g_end), ("assembly", g_end, gen_end)]
    regions = [{"name": name, "t0": a - start, "t1": b - start,
                "dur": b - a, "lost": lost(a, b)} for name, a, b in cuts if b - a > 0.5]
    total = sum(r["lost"] for r in regions) or 1.0
    for r in regions:
        r["pct_lost"] = r["lost"] / total
        r["pct_span"] = r["dur"] / max(gen_end - start, 1e-9)
        r["lost"] = round(r["lost"], 1)
        r["dur"] = round(r["dur"], 1)
        r["t0"], r["t1"] = round(r["t0"], 1), round(r["t1"], 1)
    return {"peak": peak, "lost_total": round(total, 1), "regions": regions}


def step_payload(by_traj, gen, phases, instances: list, action_names: list) -> dict | None:
    """Every reduction one step's charts need, in one JSON-able dict."""
    t0, t1 = gen["t"], gen["t"] + gen.get("dur", 0.0)
    episodes, requests, tool_events, grading, traced = [], [], [], [], []
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
                tool_events.append(e)
            elif e["name"] in ("eval_wait", "score") and e.get("t_end"):
                grading.append(e)
    if not episodes or not requests:
        return None

    start = min(e["t"] for e in episodes)
    end = max(e["t_end"] for e in episodes)
    span = end - start
    # The series run to the end of the *gen phase*, not the last episode: the
    # phase keeps running after the agent loops stop -- grading containers, then
    # verl assembling the batch -- and that trailing flat zero is the point.
    full = max(t0 + gen.get("dur", 0.0) - start, span)

    # In-flight concurrency: +1 when a request starts occupying the GPU, -1 when
    # its last token is sampled. That is ``decode_start``, NOT ``request_scheduled``:
    # scheduled fires ~0.4ms after the request arrives, when the server accepts it
    # into its *waiting* queue, and the real wait for a decode slot is the 200ms
    # median between the two. Counting from scheduled put queued requests on the
    # GPU-occupancy curve and pushed it above max_num_seqs, which a decode batch
    # cannot exceed.
    busy = [(e["decode_start"], e["decode_finished"]) for e in requests]
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
    dt = max(full / SERIES_POINTS, 0.5)
    nb = int(full / dt) + 1
    ep_grid, req_grid, tok_grid, gr_grid = [0.0] * nb, [0.0] * nb, [0.0] * nb, [0.0] * nb

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
    if grading:
        occupancy([(e["t"], e["t_end"]) for e in grading], gr_grid)
    for e in requests:
        spread(e["decode_start"], e["decode_finished"], tok_grid, e.get("completion_tokens", 0))

    occ = occupancy_regions(times, conc, start, start + full,
                            max((e["t_end"] for e in grading), default=None))

    durations = sorted(e["t_end"] - e["t"] for e in episodes)
    n = len(durations)
    slots = max(concurrency_peak([(e["t"], e["t_end"]) for e in episodes]), 1)
    ideal = sum(durations) / slots  # a perfect packing of the same work
    tool_s = sum(e.get("dur", 0.0) for e in tool_events)
    dec_s = sum(e["decode_finished"] - e["decode_start"] for e in requests)
    q_s = sum(e.get("queue_s", 0.0) for e in requests)
    pre_s = sum(e.get("prefill_s", 0.0) for e in requests)
    ep_wall = sum(durations)
    out_tok = sum(e.get("completion_tokens", 0) for e in requests)
    prompt_tok = sum(e.get("prompt_tokens", 0) for e in requests)
    cached_tok = sum(e.get("cached_tokens", 0) for e in requests)

    trace = sorted(
        (trace_row(ev, next(e for e in ev if e["name"] == "episode" and e.get("t_end")), start,
                   instances, action_names) for ev in traced),
        key=lambda r: r["s"],
    )

    exits = collections.Counter(e.get("exit_status", "?") for e in episodes)

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
            "grading": [round(v, 1) for v in gr_grid],
            "tok_s": [round(v / dt, 1) for v in tok_grid],
        },
        "occupancy": occ,
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
        "tool_calls": len(tool_events),
        "actions": action_rows(tool_events),
    }


# ---------------------------------------------------------------------------
# per-step scalars: metrics.csv, METRICS.md, and the timeline's own reductions
# ---------------------------------------------------------------------------
# The report's metrics view is a browser over every scalar the run produced per
# step. Two sources feed it: ``metrics.csv`` next to the report -- the metric x
# step table scripts/build_metrics_csv.py extracts from run.log, i.e. exactly
# what verl logged to W&B -- and the reductions only the timeline can produce,
# emitted here under ``rollout/``. METRICS.md documents one tag per table row;
# those sentences ride along so a tag can be read without leaving the page.

_CODE = re.compile(r"`([^`]+)`")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_BRACE = re.compile(r"^(.*)\{([^}]*)\}(.*)$")


def expand_braces(name: str) -> list[str]:
    """``critic/score/{mean,max,min}`` -> the three tags it documents."""
    m = _BRACE.match(name)
    if not m:
        return [name]
    head, body, tail = m.groups()
    return [x for opt in body.split(",") for x in expand_braces(head + opt.strip() + tail)]


def load_docs(path: str) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """``({tag: meaning}, [(prefix, meaning)])`` from the METRICS.md tables.

    A row's first cell names what the sentence after it documents: one tag,
    several (``srv/prompt_tokens``, ``srv/generation_tokens``), a brace family
    (``prompt_length/{mean,max,min}``), a wildcard (``rollout_corr/*``) or a
    placeholder (``traj/exit/<Status>``). The last two become prefixes, matched
    longest-first and only where no row names the tag outright. The first row to
    claim a tag wins, which keeps the reference sections ahead of the
    known-dead-rows table at the end. Markdown is stripped so the sentence reads
    inside a table cell.
    """
    if not os.path.exists(path):
        return {}, []
    docs: dict[str, str] = {}
    patterns: dict[str, str] = {}
    with open(path) as fh:
        for line in fh:
            cells = line.strip().split("|")
            if len(cells) < 4 or not line.startswith("|"):  # not a table row
                continue
            names, desc = cells[1], _MD_LINK.sub(r"\1", cells[2]).replace("**", "").replace("`", "").strip()
            if not desc or set(desc) <= set("- :"):  # the header separator
                continue
            for code in _CODE.findall(names):
                for name in expand_braces(code.strip()):
                    if "*" in name or "<" in name:
                        patterns.setdefault(name.split("*")[0].split("<")[0], desc)
                    else:
                        docs.setdefault(name, desc)
    return docs, sorted(patterns.items(), key=lambda kv: -len(kv[0]))


def load_metrics_csv(path: str) -> tuple[list[int], dict[str, dict[int, float]]]:
    """``metrics.csv`` as (steps, {tag: {step: value}}). Blank cells stay missing.

    A blank cell means the metric was not emitted at that step -- step 0 is the
    pre-training validation pass and carries only ``val-*`` -- so it is dropped
    rather than read as zero.
    """
    if not os.path.exists(path):
        return [], {}
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return [], {}
    steps = [int(c.rsplit("_", 1)[-1]) for c in rows[0][1:]]
    values: dict[str, dict[int, float]] = {}
    for row in rows[1:]:
        if not row or not row[0]:
            continue
        cells = {}
        for step, cell in zip(steps, row[1:]):
            try:
                cells[step] = float(cell)
            except (TypeError, ValueError):
                continue
        if cells:
            values[row[0]] = cells
    return steps, values


# Timeline-only reductions, as per-step scalars. These are the numbers the
# charts are drawn from -- the ones the step line in run.log cannot carry
# because they need every request's own instants -- so the metrics view can
# answer a question about a step without scrolling its charts.
ROLLOUT_DOCS = {
    "rollout/span_s": "Wall clock from the first episode admitted to the last one finished. Shorter than the gen phase, which also waits on grading and batch assembly.",
    "rollout/gen_phase_s": "The trainer's whole generate span, rollout plus the grading and assembly tail.",
    "rollout/gen_share_of_step": "Share of the training step spent in generate. The rest is the training half, during which the inference server is idle.",
    "rollout/episodes": "Episodes that ran in this rollout.",
    "rollout/episodes_per_min": "Episode completion rate over the span -- the throughput number a container-slot change has to move.",
    "rollout/container_slots": "Peak concurrent episodes, i.e. how many container slots the run actually held.",
    "rollout/requests": "Generate calls served, one per assistant turn.",
    "rollout/tokens_out": "Tokens decoded across every request of the rollout.",
    "rollout/tokens_prompt": "Prompt tokens seen, cached ones included.",
    "rollout/prefix_cache_hit_rate": "Share of prompt tokens served from SGLang's prefix cache instead of being prefilled again.",
    "rollout/decode_tok_s": "Cluster-wide decode throughput over the span: tokens out / span.",
    "rollout/mean_inflight": "Mean requests actually decoding on the GPU, averaged over the span (decode seconds / span).",
    "rollout/peak_inflight": "Most requests decoding at once.",
    "rollout/tail_waste_s": "Span minus the ideal packing of the same episodes into the same slots: what the tail costs.",
    "rollout/tail_waste_ratio": "Tail waste as a share of the span.",
    "rollout/episode_p50_s": "Median episode duration.",
    "rollout/episode_p90_s": "p90 episode duration.",
    "rollout/episode_p99_s": "p99 episode duration.",
    "rollout/episode_max_s": "The longest episode of the step.",
    "rollout/straggler_ratio": "p99 / median episode duration.",
    "rollout/turns_per_episode": "Mean assistant turns per episode.",
    "rollout/tokens_per_turn": "Mean tokens decoded per request.",
    "rollout/tool_calls": "Tool calls executed in this rollout.",
    "rollout/tool_calls_per_episode": "Mean tool calls per episode.",
    "rollout/tool_wall_s": "Tool execution wall clock, summed over every call.",
    "rollout/decode_wall_s": "Decode wall clock, summed over every request.",
    "rollout/decode_share_of_episode": "Decode as a share of summed episode wall clock -- the rest is tool calls and the agent loop's own overhead.",
    "rollout/queue_wall_s": "Time requests spent waiting for a decode slot, summed.",
    "rollout/prefill_wall_s": "Prefill wall clock, summed over every request.",
    "rollout/reward_mean": "Mean reward over the episodes of this rollout.",
    "rollout/resolved_rate": "Share of episodes the SWE-bench harness marked resolved.",
    "rollout/exit/": "Share of episodes that ended with this exit status.",
    "rollout/occupancy/": "The rollout's regions: how long each lasted and the slot-seconds it left on the floor against the phase's peak in-flight.",
}


def derived_metrics(steps: list[dict]) -> dict[str, dict[int, float]]:
    """``{rollout/<tag>: {step: value}}`` from the per-step payloads.

    A validation pass gets the same reductions under ``val-rollout/``: it is a
    rollout at the same step number as a training one, so it needs its own
    prefix for the same reason verl keeps ``val-core/*`` apart from ``critic/*``.
    """
    out: dict[str, dict[int, float]] = collections.defaultdict(dict)
    for st in steps:
        k = st.get("step")
        if k is None:
            continue
        prefix = "rollout/" if st["phase"] == "train" else "val-rollout/"
        d, r, w = st["durations"], st["rates"], st["ep_wall"]
        ep_wall = w["decode"] + w["tool"] + w["other"]
        phase_dur = st["phases"].get("step", {}).get("dur", 0.0)
        vals = {
            "span_s": st["span"],
            "gen_phase_s": st["gen_dur"],
            "gen_share_of_step": st["gen_dur"] / phase_dur if phase_dur else None,
            "episodes": st["episodes"],
            "episodes_per_min": r["ep_min"],
            "container_slots": st["slots"],
            "requests": st["requests"],
            "tokens_out": st["out_tok"],
            "tokens_prompt": st["prompt_tok"],
            "prefix_cache_hit_rate": st["cached_tok"] / st["prompt_tok"] if st["prompt_tok"] else None,
            "decode_tok_s": r["tok_s"],
            "mean_inflight": r["mean_inflight"],
            "peak_inflight": st["peak_inflight"],
            "tail_waste_s": st["tail_waste_s"],
            "tail_waste_ratio": st["tail_waste_s"] / st["span"] if st["span"] else None,
            "episode_p50_s": d["p50"],
            "episode_p90_s": d["p90"],
            "episode_p99_s": d["p99"],
            "episode_max_s": d["max"],
            "straggler_ratio": d["p99"] / d["p50"] if d["p50"] else None,
            "turns_per_episode": r["turns"],
            "tokens_per_turn": r["tok_per_req"],
            "tool_calls": st["tool_calls"],
            "tool_calls_per_episode": st["tool_calls"] / st["episodes"] if st["episodes"] else None,
            "tool_wall_s": w["tool"],
            "decode_wall_s": w["decode"],
            "decode_share_of_episode": w["decode"] / ep_wall if ep_wall else None,
            "queue_wall_s": st["req_time"]["queue"],
            "prefill_wall_s": st["req_time"]["prefill"],
            "reward_mean": r["reward"],
            "resolved_rate": r["resolved"],
        }
        for status, n in st["exits"]:
            vals[f"exit/{status}"] = n / st["episodes"] if st["episodes"] else None
        for region in st.get("occupancy", {}).get("regions", []):
            vals[f"occupancy/{region['name']}/dur_s"] = region["dur"]
            vals[f"occupancy/{region['name']}/lost_slot_s"] = region["lost"]
        for tag, v in vals.items():
            if v is not None:
                out[prefix + tag][k] = round(float(v), 6)
    return out


def metrics_payload(steps: list[dict], csv_path: str, docs_path: str) -> dict:
    """Every per-step scalar as one tag table: ``{steps, tags, docs}``.

    Tags are dense over the union of both sources' steps, with ``None`` where a
    tag was not emitted -- which is what lets the view say "n steps" per tag
    instead of silently averaging over a different denominator per row.
    """
    csv_steps, by_tag = load_metrics_csv(csv_path)
    derived = derived_metrics(steps)
    by_tag.update(derived)
    all_steps = sorted(set(csv_steps) | {s for cells in derived.values() for s in cells})
    docs, patterns = load_docs(docs_path)
    for tag in by_tag:
        if tag in docs:
            continue
        # the validation copy of a rollout reduction means the same thing
        if tag.startswith("val-rollout/"):
            docs[tag] = "Validation pass. " + (ROLLOUT_DOCS.get("rollout/" + tag.split("/", 1)[1], "") or "")
            continue
        # a group's own line, then the nearest documented prefix
        parts = tag.split("/")
        found = ROLLOUT_DOCS.get(tag) or next(
            (ROLLOUT_DOCS["/".join(parts[:i]) + "/"] for i in range(len(parts) - 1, 0, -1)
             if "/".join(parts[:i]) + "/" in ROLLOUT_DOCS), "")
        docs[tag] = found or next((d for prefix, d in patterns if tag.startswith(prefix)), "")
    return {
        "steps": all_steps,
        "tags": {t: [cells.get(s) for s in all_steps] for t, cells in sorted(by_tag.items())},
        # only the tags present, so METRICS.md's ~170 rows do not all ride along
        "docs": {t: d for t, d in docs.items() if d and t in by_tag},
        "csv": os.path.basename(csv_path) if csv_steps else None,
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


def build(directory: str, experiment: str | None = None, metrics_csv: str | None = None) -> dict:
    by_traj, train = load(directory)
    gens = [e for e in train if e["name"] in ROLLOUT_PHASES]
    if not gens:
        raise SystemExit(f"no trainer rollout spans in {directory}; nothing to attribute episodes to")
    by_step: dict = collections.defaultdict(dict)
    for e in train:
        if e.get("t_end"):
            by_step[e.get("step")][e["name"]] = {"dur": e.get("dur", 0.0), "t": e["t"], "t_end": e["t_end"]}
    steps, instances, action_names = [], [], []
    for gen in gens:
        payload = step_payload(by_traj, gen, by_step.get(gen.get("step"), {}), instances, action_names)
        if payload:
            steps.append(payload)
    here = os.path.dirname(os.path.abspath(__file__))
    return {
        "experiment": experiment or experiment_name(directory),
        "metrics": metrics_payload(
            steps,
            metrics_csv or os.path.join(run_dir(directory), "metrics.csv"),
            os.path.join(os.path.dirname(here), "METRICS.md"),
        ),
        "kinds": KINDS,
        "instances": instances,
        "action_names": action_names,
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
    ap.add_argument("--metrics", default=None,
                    help="metrics.csv for the metrics view (default: the one in the run dir)")
    args = ap.parse_args()
    data = build(args.dir, args.experiment, args.metrics)
    here = os.path.dirname(os.path.abspath(__file__))
    # The committed template is the "_example" file: it carries the __DATA__
    # placeholder, not a rendered run. Prefer an un-suffixed name if one appears.
    tmpl = next(os.path.join(here, n) for n in ("timeline_report.html", "timeline_report_example.html")
                if os.path.exists(os.path.join(here, n)))
    with open(tmpl) as fh:
        html = fh.read()
    html = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    out = args.out or os.path.join(run_dir(args.dir), "timeline.html")
    with open(out, "w") as fh:
        fh.write(html)
    print(f"wrote {out} ({os.path.getsize(out)/1e3:.0f} kB, {len(data['steps'])} steps, "
          f"{len(data['metrics']['tags'])} metric tags)")


if __name__ == "__main__":
    main()
