"""Adapter for this repo's own run timelines: a directory of ``timeline-<pid>.jsonl`` shards.

This is the trace with the most in it, which makes it the format sample for the
degradation test of `SIMULATOR.md` §15, not the "best" input: it carries the
answers right next to the workload, and the whole job here is to keep them apart
(§3, §6). Per event:

======================  ====================================================
``generate``            ``prompt_tokens`` / ``completion_tokens`` -> Request, traced.
                        The server timestamps and ``cached_tokens`` -> observed.
``tool_call``           -> Task ``tool``, duration traced. Several in one turn
                        are a *chain*, because the harness runs them one after
                        another (``agent_loop._execute``); fanning them out
                        would invent parallelism this harness never had.
``container_start``     -> Task on the ``container`` pool, and the start of the
                        session's lease on it.
``patch_recover``       -> Task inside the container, when present.
``cleanup``             -> Task inside the container; the lease ends after it.
``score``               -> Task ``eval`` on the ``eval`` pool.
``admission_wait``      -> observed wait on ``container``.
``eval_wait``           -> observed wait on ``eval``.
``episode``             -> observed session span. Also how a session is placed
                        in a training step (below).
``format_error``        No work: the next ``generate`` simply follows the last.
trainer ``cat=train``   -> observed training phases, and the step windows.
======================  ====================================================

**One workload is one training step.** The thing being predicted is a step's
wall clock (§0), and the sessions of one step are released together, which is
what ``release_offset_s = 0`` means. Episodes are placed in a step by the
trainer's ``gen`` span that contains them, as ``scripts/build_timeline.py`` does;
validation windows and other steps are skipped and counted in the notes.

**A group shares one prefix.** The eight rollouts of one problem send the same
first prompt (0 groups out of 256 disagreed in step 0 of run 20260912-072938),
so each group gets a :class:`~simulator.ir.Prefix` of that length and turn 1
names it. The system prompt that *every* group shares is not separable from the
problem text without the tokenizer, so it is not written, and the notes say so.

**What this trace does not say.** Why generation stopped: ``ContextLimit`` means
the *context* filled, not that the request hit its cap, so ``stop`` stays
``"unknown"``. The cap itself: the config bundle's job. Whether a tool call was
killed on timeout: the event has no flag, so ``duration_is_cap`` is never set.

**Missing fields.** A duration that is not on the event is derived from
``t_end - t`` when both are there, and otherwise drawn from the caller's prior for
that task kind (:data:`~simulator.adapters.base.KNOB_BY_KIND`). Missing token
counts are refused: ``prompt_tokens`` of turn k+1 is tied to turn k's prompt and
completion by the pure-append rule, and the IR draws each unknown field
independently, so there is no honest way to estimate one without the other yet.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from simulator.adapters.base import KNOB_BY_KIND, AdapterError, Conversion, Priors
from simulator.ir import Lease, Prefix, Request, Session, Source, Task, Unknown, Workload
from simulator.observed import Observations, ObservedRequest, ObservedSession

NAME = "agentic-timeline"
_SHARD = "timeline-*.jsonl"

Event = dict[str, Any]


class AgenticTimelineAdapter:
    """See the module docstring. ``step`` picks the training step when a run has several."""

    name = NAME

    def __init__(self, step: int | None = None):
        self.step = step

    def claims(self, path: Path) -> bool:
        return bool(_shards(path))

    def convert(self, path: Path, priors: Priors) -> Conversion:
        events = _read(_shards(path))
        train = [e for e in events if e.get("cat") == "train"]
        traj = [e for e in events if e.get("cat") == "traj" and "traj" in e]
        step, t0, t1 = self._window(train, path)
        run = path.name if path.is_dir() else path.parent.name
        source = f"run:{run}/step{step}"

        episodes = {e["traj"]: e for e in traj if e["name"] == "episode"}
        chosen = sorted((e["traj"] for e in episodes.values() if t0 <= e["t"] <= t1),
                        key=lambda k: (episodes[k]["t"], k))
        if not chosen:
            raise AdapterError(f"no episodes fall in step {step}'s rollout window of {path}")
        by_traj: dict[str, list[Event]] = defaultdict(list)
        for e in traj:
            if e["traj"] in episodes and t0 <= episodes[e["traj"]]["t"] <= t1:
                by_traj[e["traj"]].append(e)

        tally = _Tally()
        prefixes: list[Prefix] = []
        sessions: list[Session] = []
        obs_req: list[ObservedRequest] = []
        obs_sess: list[ObservedSession] = []
        for group, members in _groups(chosen, episodes, step):
            prefix = _prefix(group, members, by_traj, tally)
            if prefix is not None:
                prefixes.append(prefix)
            for sid in members:
                s, oreq, osess = _session(sid, group, prefix, by_traj[sid], priors, tally)
                sessions.append(s)
                obs_req.extend(oreq)
                if osess is not None:
                    obs_sess.append(osess)

        phases = tuple((e["name"], e["t"], e["t_end"]) for e in train
                       if e.get("step") == step and e.get("t_end") is not None)
        skipped = len(episodes) - len(chosen)
        if skipped:
            tally.note(f"{skipped} episodes outside step {step}'s rollout window skipped "
                       f"(validation, or other steps)")
        workload = Workload(sessions=tuple(sessions), source=source, prefixes=tuple(prefixes),
                            meta={"adapter": NAME, "run": run, "step": step,
                                  "shards": [p.name for p in _shards(path)]})
        observations = Observations(source=source, requests=tuple(obs_req),
                                    sessions=tuple(obs_sess), phases=phases)
        return Conversion(adapter=NAME, workload=workload, observations=observations,
                          notes=tally.notes(len(sessions)))

    def _window(self, train: list[Event], path: Path) -> tuple[int, float, float]:
        spans = {e["step"]: (e["t"], e["t_end"]) for e in train
                 if e.get("name") == "gen" and isinstance(e.get("step"), int) and "t_end" in e}
        if not spans:
            raise AdapterError(
                f"{path} has no trainer 'gen' span, so its episodes cannot be placed in a "
                f"training step; include the trainer's timeline shard"
            )
        step = self.step
        if step is None:
            if len(spans) > 1:
                raise AdapterError(
                    f"{path} holds steps {sorted(spans)}; a workload is one step, pass "
                    f"AgenticTimelineAdapter(step=...)"
                )
            step = next(iter(spans))
        if step not in spans:
            raise AdapterError(f"{path} has no step {step}; it holds {sorted(spans)}")
        t0, t1 = spans[step]
        return step, t0, t1


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def _shards(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.glob(_SHARD))
    if path.is_file() and path.match(_SHARD):
        return [path]
    return []


def _read(shards: Iterable[Path]) -> list[Event]:
    """Every event in every shard. A truncated last line, from a killed run, is skipped."""
    out: list[Event] = []
    for p in shards:
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def _groups(chosen: list[str], episodes: dict[str, Event],
            step: int) -> list[tuple[str, list[str]]]:
    members: dict[str, list[str]] = defaultdict(list)
    for sid in chosen:
        members[f"step{step}/{episodes[sid]['instance_id']}"].append(sid)
    return sorted(members.items())


# ---------------------------------------------------------------------------
# one group's prefix, one session's DAG
# ---------------------------------------------------------------------------
def _prefix(group: str, members: list[str], by_traj: dict[str, list[Event]],
            tally: "_Tally") -> Prefix | None:
    firsts = []
    for sid in members:
        g1 = [e for e in by_traj[sid] if e["name"] == "generate" and e.get("turn") == 1]
        if g1 and g1[0].get("prompt_tokens") is not None:
            firsts.append(g1[0]["prompt_tokens"])
    if not firsts:
        return None
    if len(set(firsts)) > 1:
        tally.uneven_groups += 1
    return Prefix(id=f"{group}/prompt", tokens=min(firsts), prov={"tokens": "derived"})


def _session(sid: str, group: str, prefix: Prefix | None, evs: list[Event], priors: Priors,
             tally: "_Tally") -> tuple[Session, list[ObservedRequest], ObservedSession | None]:
    one = {n: [e for e in evs if e["name"] == n] for n in
           ("episode", "container_start", "patch_recover", "cleanup", "score",
            "admission_wait", "eval_wait")}
    gens = sorted((e for e in evs if e["name"] == "generate"), key=lambda e: e["turn"])
    tools: dict[int, list[Event]] = defaultdict(list)
    for e in evs:
        if e["name"] == "tool_call":
            tools[e["turn"]].append(e)

    tasks: list[Task] = []
    requests: list[Request] = []
    observed: list[ObservedRequest] = []

    def task(tid: str, kind: str, ev: Event | None, after: tuple[str, ...],
             pool: str | None = None) -> Task:
        dur, src, unk = _duration(ev or {}, kind, priors, tally)
        prov: dict[str, Source] = {"duration_s": src}
        unknown = {"duration_s": unk} if unk is not None else {}
        t = Task(id=tid, session=sid, duration_s=dur, pool=pool, kind=kind, after=after,
                 prov=prov, unknown=unknown)
        tasks.append(t)
        return t

    start = task(f"{sid}/start", "container_start", _first(one["container_start"]), (),
                 pool="container")
    after: tuple[str, ...] = (start.id,)
    prev: Request | None = None
    for g in gens:
        turn = g["turn"]
        prompt, completion = g.get("prompt_tokens"), g.get("completion_tokens")
        if prompt is None or completion is None:
            raise AdapterError(
                f"generate turn {turn} of {sid!r} has no token counts; this adapter cannot "
                f"estimate them, because turn k+1's prompt is tied to turn k's prompt and "
                f"completion and the IR draws unknown fields independently"
            )
        rid = f"{sid}/gen{turn}"
        if prev is None:
            parent = prefix.id if prefix is not None else None
            shared = prefix.tokens if prefix is not None else 0
        else:
            parent, shared = prev.id, prev.prompt_tokens + prev.completion_tokens
        req = Request(id=rid, session=sid, prompt_tokens=prompt, prefix_parent=parent,
                      shared_prefix_tokens=shared, completion_tokens=completion, after=after,
                      prov={"prompt_tokens": "traced", "completion_tokens": "traced",
                            "shared_prefix_tokens": "derived"})
        requests.append(req)
        tally.requests += 1
        o = _observed_request(rid, g)
        if o is not None:
            observed.append(o)
        after = (rid,)
        # Serial: the harness runs a turn's tool calls one after another.
        for i, tc in enumerate(sorted(tools.get(turn, ()), key=lambda e: e["t"])):
            t = task(f"{sid}/tool{turn}.{i}", "tool", tc, after)
            after = (t.id,)
        prev = req

    for name, kind in (("patch_recover", "patch_recover"), ("cleanup", "cleanup")):
        ev = _first(one[name])
        if ev is not None:
            t = task(f"{sid}/{kind}", kind, ev, after)
            after = (t.id,)
    release_after = after[0]
    task(f"{sid}/eval", "eval", _first(one["score"]), after, pool="eval")

    session = Session(id=sid, requests=tuple(requests), tasks=tuple(tasks), group=group,
                      leases=(Lease(pool="container", acquire_before=start.id,
                                    release_after=release_after),))
    return session, observed, _observed_session(sid, one)


def _first(evs: list[Event]) -> Event | None:
    return min(evs, key=lambda e: e["t"]) if evs else None


def _duration(ev: Event, kind: str, priors: Priors,
              tally: "_Tally") -> tuple[float, Source, Unknown | None]:
    """A task's duration and where it came from, per field (`SIMULATOR.md` §6)."""
    if ev.get("dur") is not None:
        return float(ev["dur"]), "traced", None
    if ev.get("t") is not None and ev.get("t_end") is not None:
        tally.derived[kind] += 1
        return float(ev["t_end"] - ev["t"]), "derived", None
    knob = KNOB_BY_KIND[kind]
    u = priors.require(knob, missing=f"{kind} durations")
    tally.estimated[knob] += 1
    return 0.0, "estimated", u


def _observed_request(rid: str, g: Event) -> ObservedRequest | None:
    """The answer half of a generate event, or None when the trace was stripped of it."""
    needed = ("decode_start", "decode_finished")
    arrival = g.get("request_received", g.get("t"))
    if arrival is None or g.get("cached_tokens") is None or any(g.get(k) is None for k in needed):
        return None
    return ObservedRequest(id=rid, t_arrival=float(arrival), t_first_token=float(g["decode_start"]),
                           t_last_token=float(g["decode_finished"]),
                           cached_tokens=int(g["cached_tokens"]), replica=g.get("replica"))


def _observed_session(sid: str, one: dict[str, list[Event]]) -> ObservedSession | None:
    ep = _first(one["episode"])
    if ep is None or ep.get("t_end") is None:
        return None
    waits: dict[str, float] = {}
    for name, pool in (("admission_wait", "container"), ("eval_wait", "eval")):
        w = _first(one[name])
        if w is not None and w.get("dur") is not None:
            waits[pool] = float(w["dur"])
    return ObservedSession(id=sid, t_start=float(ep["t"]), t_end=float(ep["t_end"]), waits_s=waits)


class _Tally:
    """What the conversion had to do without, for :attr:`Conversion.notes`."""

    def __init__(self) -> None:
        self.requests = 0
        self.uneven_groups = 0
        self.derived: Counter[str] = Counter()
        self.estimated: Counter[str] = Counter()
        self._extra: list[str] = []

    def note(self, s: str) -> None:
        self._extra.append(s)

    def notes(self, n_sessions: int) -> tuple[str, ...]:
        out = [
            f"{self.requests} requests: max_new_tokens left unset for the config bundle to "
            f"fill; stop is 'unknown' because the timeline does not record why generation ended",
            "tool_call events carry no timeout flag, so duration_is_cap is False on every task",
            f"one prefix per group ({n_sessions} sessions): the system prompt shared across "
            f"groups is not separable from the problem text without the tokenizer",
        ]
        if self.uneven_groups:
            out.append(f"{self.uneven_groups} groups whose first prompts differ in length; the "
                       f"prefix is the shortest")
        for kind, n in sorted(self.derived.items()):
            out.append(f"{n} {kind} durations derived from t_end - t (no dur on the event)")
        for knob, n in sorted(self.estimated.items()):
            out.append(f"{n} durations estimated from prior {knob!r}")
        return tuple(out + self._extra)
