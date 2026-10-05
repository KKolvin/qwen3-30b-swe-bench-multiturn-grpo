"""The workload IR: everything a simulator is allowed to be told about a workload.

This is the interface between the workload layer and the system layer, and it is
drawn where it is for one reason: the system layer must not be able to tell which
generator produced its input. A SWE-bench agentic replay and a synthetic open-loop
serving load both arrive here as the same node types, so a simulator
calibrated on one cannot be quietly biased toward it. See `SIMULATOR.md` §4 and §13.

Three rules shape every field below.

**No timestamps.** A workload carries *durations* and *structure*, never instants.
When something happens is what the simulator computes; if an arrival time were an
input, the closed loop would be short-circuited and the prediction would be
partly a copy of its own answer. The only temporal input is
:attr:`Session.release_offset_s`, which is what makes an open-loop arrival process
expressible at all -- and under sync GRPO every session leaves it at 0.0.

**No observed quantities.** Anything visible in a real timeline is an output
(`SIMULATOR.md` §3). The sharpest case is prefix reuse: this file carries
:attr:`Request.shared_prefix_tokens`, the number of leading tokens a request
*structurally* shares with its parent, which is a property of the token stream.
It does not carry how many of those tokens were actually served from cache -- that
depends on the radix tree, its LRU eviction, the page size and the routing, and is
exactly what the simulator has to get right. The measured `cached_tokens` lives in
:mod:`simulator.observed` and is a validation target.

**Anything may be unknown.** A trace from an unfamiliar harness is missing fields
that ours records, and a converter facing a required field has no choice but to
invent a number that then looks exactly like a measured one. So every carrier of a
scalar carries two side tables: :attr:`Request.prov`, where each field came from,
and :attr:`Request.unknown`, the distribution to draw it from when it came from
nowhere. The engine reads neither. :mod:`simulator.realise` turns the
distributions into concrete values under a seed before the engine sees the
workload, which is how "the engine must not branch on provenance"
(`SIMULATOR.md` §6) survives contact with a workload that is half estimated.

Units are seconds (float) and tokens (int), stated in every field that has one.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Iterator, Literal, Protocol

NodeId = str
SessionId = str
PrefixId = str

#: Why a request stopped generating. ``"length"`` means it hit its own
#: ``max_new_tokens``, so a counterfactual that moves that cap invalidates the
#: replayed ``completion_tokens`` for this request and no other. ``"unknown"`` is
#: the honest answer for a trace that recorded only the conversation, and is the
#: default for that reason: a converter that cannot tell should not have to pick.
StopReason = Literal["eos", "length", "unknown"]

#: Where one field's value came from (`SIMULATOR.md` §6). ``"traced"`` was read
#: out of the trace. ``"config"`` came from the recipe or the server args.
#: ``"derived"`` was reconstructed from traced data by a rule that can be written
#: down -- a tokenizer over the rendered messages, arithmetic across neighbouring
#: turns. ``"estimated"`` came from a distribution, and is the only source that
#: may carry an :class:`Unknown`.
Source = Literal["traced", "config", "derived", "estimated"]


class Dist(Protocol):
    """A distribution, addressed by quantile.

    Quantiles rather than samples because the sensitivity table of
    `SIMULATOR.md` §6 walks one knob from p10 to p90 with every other knob pinned
    at p50, and that sweep has to address the same distribution, the same way, as
    the sampling that produces the interval.
    """

    def quantile(self, q: float) -> float:
        """Value at ``q`` in [0, 1]."""


@dataclass(frozen=True)
class Uniform:
    """A range and nothing behind it: "the training phases take 20 to 40 minutes"."""

    lo: float
    hi: float

    def quantile(self, q: float) -> float:
        return self.lo + (self.hi - self.lo) * min(max(q, 0.0), 1.0)


@dataclass(frozen=True)
class Empirical:
    """Values measured somewhere else: another run, another harness, another cap.

    Kept sorted, so a workload with 95,403 tasks sharing one of these does not
    re-sort it 95,403 times.
    """

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(sorted(self.values)))

    def quantile(self, q: float) -> float:
        xs = self.values
        if not xs:
            raise IRError("Empirical with no values")
        pos = min(max(q, 0.0), 1.0) * (len(xs) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(xs) - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


@dataclass(frozen=True)
class Unknown:
    """A field nobody measured, and where to get a number for it.

    :attr:`knob` is the name this field answers to in the sensitivity table: the
    fields under one knob share its draw, and the table has one row per knob, not
    one row per node. It is also what the provenance report prints, so it should
    read like a row label -- ``"tool_duration"``, ``"train_phases"``.

    :attr:`dist` is the shape of the quantity: what *one* node's value looks like.
    Every field draws its own, so the spread the queue is built out of survives
    into the simulation.

    :attr:`scale` is how wrong that shape might be as a whole, as a multiplier,
    and it is a separate question. One draw per knob per realisation multiplies
    every field under that knob, which is the only part of an estimate that does
    not wash out: across 20,000 tool tasks and 64 realisations, per-node draws
    alone moved the predicted total by 2.1%, under the 4.0% to 16.8% that two
    supposedly identical replicas of one run differ by
    (:mod:`simulator.observed`). An interval that tight is not a prediction. So a
    borrowed distribution needs a ``scale`` saying how far it might be off -- two
    runs of the same harness with different container caps give one -- and
    ``None`` is a claim that the distribution is right and only the per-node value
    was missing.

    It is deliberately not inferred from :attr:`dist`. How widely tool times vary
    between tool calls says nothing about how wrong their median is, and reading
    one as the other puts the whole weight of the sensitivity table on whichever
    quantity happens to have the heaviest tail.
    """

    dist: Dist
    knob: str
    scale: Dist | None = None


@dataclass(frozen=True)
class Request:
    """One inference call: a prompt, and the tokens that were generated from it.

    ``completion_tokens`` is replayed ground truth, not a prediction: what the
    model wrote does not depend on TP, KV dtype, batch caps or scheduling.
    """

    id: NodeId
    session: SessionId

    #: Total prompt length in tokens, including the shared prefix.
    prompt_tokens: int

    #: What this request's prompt extends: a :class:`Request` (its full sequence,
    #: ``prompt_tokens + completion_tokens``), a :class:`Prefix`, or ``None`` for a
    #: genuinely cold prompt. An agentic turn names the previous turn; the *first*
    #: turn names the prefix its group shares (system prompt plus problem text), not
    #: ``None`` -- in run 20260912-072938 turn-1 prompts were 82.6% cache hits.
    #: A harness that compacts or summarises its context still fits: it names the
    #: ancestor it still shares a leading run of tokens with, and shares fewer.
    prefix_parent: NodeId | PrefixId | None

    #: Leading tokens identical to the parent's content. This is structural
    #: potential -- what *could* be reused -- not what a cache actually served.
    #: Zero when ``prefix_parent`` is ``None``.
    shared_prefix_tokens: int

    #: Tokens generated. See the class docstring on invariance.
    completion_tokens: int

    #: Why generation stopped. ``"unknown"`` when the trace does not say, which is
    #: most traces; the engine never reads this field, so an unknown here costs
    #: nothing until a counterfactual moves ``max_new_tokens``.
    stop: StopReason = "unknown"

    #: The generation cap that was in force, in tokens, or ``None`` when neither
    #: the trace nor the config said. An **input to admission**: SGLang reserves KV
    #: for ``min(max_new_tokens - generated, 4096) * new_token_ratio`` tokens per
    #: running request. It must stay the real cap -- setting it to
    #: ``completion_tokens`` would hand admission the answer, and retraction would
    #: silently never fire (`SIMULATOR.md` §4, §7). The replayed length is enforced
    #: by the worker emitting EOS, not by this field. The engine needs a number
    #: here: :func:`require_runnable` is where a missing one is caught, and the
    #: config bundle is what fills it.
    max_new_tokens: int | None = None

    #: Nodes that must all complete before this one becomes runnable. Empty means
    #: it hangs off the session's release. More than one is a join, which is how an
    #: assistant turn that fired three tool calls at once waits for all three --
    #: ordinary in harnesses that batch tool calls, and not expressible as a chain.
    after: tuple[NodeId, ...] = ()

    #: Delay after the last of those completions, in seconds. Almost always 0.0 --
    #: real think time belongs in a :class:`Task`, which can contend for a
    #: resource; this is for pure delay with no work behind it.
    offset_s: float = 0.0

    #: Where each field's value came from, by field name. A field missing from
    #: this table is unstated, which the provenance report lists as such rather
    #: than guessing.
    prov: dict[str, Source] = field(default_factory=dict)

    #: Fields nobody measured, by field name. Whatever sits in the field itself is
    #: a placeholder that :mod:`simulator.realise` overwrites. Every key here must
    #: be ``"estimated"`` in :attr:`prov`, so a distribution cannot hide behind a
    #: field that claims to have been traced.
    unknown: dict[str, Unknown] = field(default_factory=dict)


@dataclass(frozen=True)
class Task:
    """Non-inference work: a tool call, a container start, a grading run.

    The system layer schedules these against resource pools but knows nothing
    about what they are. ``kind`` is a reporting label only -- **the engine must
    not branch on its value**, or the workload-neutrality of `SIMULATOR.md` §13 is
    gone.
    """

    id: NodeId
    session: SessionId

    #: Measured duration in seconds. For a task that spawns a subagent (some
    #: :class:`Session` names it in :attr:`Session.spawned_by`) this is only the
    #: overhead around the child, not the child's span: how long the parent waits
    #: is the simulator's to compute.
    duration_s: float

    #: ``True`` when ``duration_s`` is a timeout cap rather than the time the work
    #: took -- the work was killed, not finished, so the real duration is only known
    #: to be >= it. The engine does not read this; it marks a censored measurement so
    #: tool-time stats do not treat it as real work (`SIMULATOR.md` §4).
    duration_is_cap: bool = False

    #: Resource pool this task needs a slot from for its duration, or ``None`` if
    #: it runs inside a slot the session already holds (see :class:`Lease`).
    #: A tool call runs inside the session's own container and is ``None``;
    #: reward grading needs its own container and names that pool.
    pool: str | None = None

    #: Free-form label for reporting: ``"tool"``, ``"container_start"``, ``"eval"``.
    kind: str = ""

    after: tuple[NodeId, ...] = ()
    offset_s: float = 0.0

    #: See :attr:`Request.prov`.
    prov: dict[str, Source] = field(default_factory=dict)

    #: See :attr:`Request.unknown`.
    unknown: dict[str, Unknown] = field(default_factory=dict)


@dataclass(frozen=True)
class Prefix:
    """Content shared across sessions: a system prompt, a problem statement.

    A node in a workload-level content tree. It says how long the content is and
    what it extends, never what it is -- the simulator materialises token ids
    from this structure itself. A prefix is **not** a node of any session's DAG:
    sharing content is not a timing dependency. Which session pays the prefill
    and which ones hit the cache is decided by arrival order and routing, which
    is output (`SIMULATOR.md` §4).
    """

    id: PrefixId

    #: Length in tokens from the root of the tree, including ``parent``'s content
    #: -- absolute, like :attr:`Request.prompt_tokens`.
    tokens: int

    parent: PrefixId | None = None

    #: See :attr:`Request.prov`.
    prov: dict[str, Source] = field(default_factory=dict)

    #: See :attr:`Request.unknown`.
    unknown: dict[str, Unknown] = field(default_factory=dict)


@dataclass(frozen=True)
class Lease:
    """A resource slot held across several nodes.

    An agentic session holds one container from before its first task until after
    its last, which is not expressible as per-task acquisition: the slot must not
    be released and re-won between turns. Capacities are **not** here -- the
    workload names a pool, the system config says how big it is.
    """

    pool: str
    acquire_before: NodeId
    release_after: NodeId


@dataclass(frozen=True)
class Session:
    """A correlated sequence of nodes.

    ``session`` is meaningful to the system layer -- it is the sticky-routing key,
    which is how one session's turns stay on one replica and keep their prefix
    cache warm (see ``src/agentic_grpo/agent_loop.py:322``). It is deliberately
    not called an episode: the system layer may know that requests are
    correlated, and must not know what correlates them.
    """

    id: SessionId
    requests: tuple[Request, ...]
    tasks: tuple[Task, ...] = ()
    leases: tuple[Lease, ...] = ()

    #: Membership in a barrier group, e.g. a GRPO group of n=8 rollouts of one
    #: problem. The workload says which sessions belong together; what a barrier
    #: does with that is orchestration config, not workload.
    group: str | None = None

    #: Seconds after the workload's own t=0 at which this session is released.
    #: 0.0 for every session of a burst-released sync batch; an open-loop arrival
    #: process expresses itself entirely through this field. Must be 0.0 when
    #: :attr:`spawned_by` is set: a subagent is released by its parent, not by the
    #: clock.
    release_offset_s: float = 0.0

    #: The :class:`Task` in another session that spawned this one -- a subagent.
    #: This session is released when that task becomes runnable, and that task is
    #: not complete until this session is.
    #:
    #: It is the only dependency allowed to cross a session boundary. ``after`` may
    #: not, because coupling two sibling sessions' timing is orchestration dressed
    #: up as workload. A subagent blocking its parent is the workload's own shape,
    #: and a harness that spawns them cannot be written down at all without this:
    #: folding the child's requests into the parent would give them the parent's
    #: routing key, and a subagent's prompt shares almost nothing with it.
    spawned_by: NodeId | None = None

    def nodes(self) -> Iterator[Request | Task]:
        yield from self.requests
        yield from self.tasks


@dataclass(frozen=True)
class Workload:
    """A complete workload, and where it came from.

    ``source`` is not decoration. A held-out validation set is only held out if
    you can tell which workload a result came from, and a calibration that
    accidentally saw a run trace is only detectable if the trace says so.
    """

    sessions: tuple[Session, ...]

    #: Provenance: a run id for a replay, a generator name and its parameters for
    #: a synthetic load.
    source: str

    #: The shared-content tree that requests may name as ``prefix_parent``.
    prefixes: tuple[Prefix, ...] = ()

    #: Anything a generator wants to record about itself. Never read by the engine.
    #: :mod:`simulator.realise` writes its seed and its per-knob quantiles here, so
    #: a realised workload says how it was drawn.
    meta: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
# A schema without its invariants is half a schema, and two of these invariants
# are load-bearing claims about this workload that were measured rather than
# assumed -- see `SIMULATOR.md` §3.


class IRError(ValueError):
    """A workload that violates the IR's invariants."""


def validate(w: Workload) -> None:
    """Raise :class:`IRError` on the first violation found.

    Checks, in order: unique ids; the prefix tree is a tree that only grows
    toward its leaves; ``after`` and ``prefix_parent`` resolve; no cycles; leases
    bracket a real span; subagent spawns point at a real task in another session
    and do not form a loop; the provenance tables name real fields; and the token
    arithmetic holds -- ``shared_prefix_tokens`` may not exceed either the
    request's own prompt or the parent's content. That last one is the
    generalisation of the pure-append property, which held on 95,403 of 95,403
    adjacent turn pairs in run 20260912-072938; a converter that gets prefix
    reconstruction wrong trips it.

    It does **not** check that the workload is complete enough to run: a field may
    legitimately be unknown here. That is :func:`require_runnable`.
    """
    prefixes = _check_prefixes(w.prefixes)
    by_id: dict[NodeId, Request | Task] = {}
    owner: dict[NodeId, SessionId] = {}
    for s in w.sessions:
        for n in s.nodes():
            if n.id in by_id or n.id in prefixes:
                raise IRError(f"duplicate node id {n.id!r}")
            if n.session != s.id:
                raise IRError(f"node {n.id!r} claims session {n.session!r}, is in {s.id!r}")
            by_id[n.id] = n
            owner[n.id] = s.id

    for s in w.sessions:
        own = {n.id for n in s.nodes()}
        for n in s.nodes():
            _check_prov(n)
            if n.offset_s < 0:
                raise IRError(f"node {n.id!r} has negative offset_s")
            for dep in n.after:
                if dep not in own:
                    # Cross-session dependencies would make one session's timing an
                    # input to another's, which is orchestration, not workload. The
                    # one real exception is a subagent: Session.spawned_by.
                    raise IRError(f"node {n.id!r} depends on {dep!r}, outside its session")
        for r in s.requests:
            _check_request(r, by_id, prefixes)
        for t in s.tasks:
            if t.duration_s < 0:
                raise IRError(f"task {t.id!r} has negative duration_s")
        _check_acyclic(s)
        for ls in s.leases:
            if ls.acquire_before not in own or ls.release_after not in own:
                raise IRError(f"lease on {ls.pool!r} in session {s.id!r} references a foreign node")
        if s.release_offset_s < 0:
            raise IRError(f"session {s.id!r} has negative release_offset_s")
        if s.spawned_by is not None and s.release_offset_s:
            raise IRError(
                f"session {s.id!r} is spawned by {s.spawned_by!r} and also has a "
                f"release_offset_s of {s.release_offset_s}: a subagent is released by its parent"
            )
    _check_spawns(w, by_id, owner)


def require_runnable(w: Workload) -> None:
    """Raise unless every field the engine reads carries a value.

    The door the engine checks at, and the reason the rest of the IR can afford to
    say "I don't know". Two things pass :func:`validate` and stop here: a field
    still holding an :class:`Unknown`, meaning :mod:`simulator.realise` has not
    run, and a ``max_new_tokens`` nobody filled, which is the config bundle's job.

    ``stop`` may stay ``"unknown"``. The engine never reads it; it guards
    counterfactuals that move the cap (`SIMULATOR.md` §1).
    """
    for carrier in (*(n for s in w.sessions for n in s.nodes()), *w.prefixes):
        if carrier.unknown:
            raise IRError(
                f"{type(carrier).__name__.lower()} {carrier.id!r} still draws "
                f"{sorted(carrier.unknown)} from a distribution; realise the workload first"
            )
    for s in w.sessions:
        for r in s.requests:
            if r.max_new_tokens is None:
                raise IRError(
                    f"request {r.id!r} has no max_new_tokens; the config bundle must supply "
                    f"the cap that was in force"
                )


def _check_prov(obj: Request | Task | Prefix) -> None:
    names = {f.name for f in fields(obj)} - {"prov", "unknown"}
    what = f"{type(obj).__name__.lower()} {obj.id!r}"
    for k in obj.prov:
        if k not in names:
            raise IRError(f"{what} records provenance for {k!r}, which is not one of its fields")
    for k, u in obj.unknown.items():
        if k not in names:
            raise IRError(f"{what} marks {k!r} unknown, which is not one of its fields")
        got = obj.prov.get(k)
        if got != "estimated":
            raise IRError(
                f"{what} draws {k!r} from a distribution but calls its provenance {got!r}; "
                f"a drawn value is 'estimated'"
            )
        if not u.knob:
            raise IRError(f"{what} leaves the sensitivity knob for {k!r} unnamed")
        _check_dist(u.dist, f"{what} field {k!r}")
        if u.scale is not None:
            _check_dist(u.scale, f"{what} field {k!r} scale")


def _check_dist(d: Dist, what: str) -> None:
    if isinstance(d, Uniform):
        if d.hi < d.lo:
            raise IRError(f"{what} has a Uniform whose hi {d.hi} is below its lo {d.lo}")
    elif isinstance(d, Empirical):
        if not d.values:
            raise IRError(f"{what} has an Empirical with no values")
    elif not callable(getattr(d, "quantile", None)):
        raise IRError(f"{what} has a distribution with no quantile method")


def _check_prefixes(ps: tuple[Prefix, ...]) -> dict[PrefixId, Prefix]:
    by_id: dict[PrefixId, Prefix] = {}
    for p in ps:
        if p.id in by_id:
            raise IRError(f"duplicate prefix id {p.id!r}")
        if p.tokens <= 0:
            raise IRError(f"prefix {p.id!r} has non-positive tokens")
        _check_prov(p)
        by_id[p.id] = p
    for p in ps:
        if p.parent is None:
            continue
        parent = by_id.get(p.parent)
        if parent is None:
            raise IRError(f"prefix {p.id!r} names missing parent {p.parent!r}")
        # Strictly longer than its parent, so the parent chain cannot cycle.
        if p.tokens <= parent.tokens:
            raise IRError(
                f"prefix {p.id!r} has {p.tokens} tokens, not more than its parent "
                f"{parent.id!r} ({parent.tokens})"
            )
    return by_id


def _check_request(r: Request, by_id: dict[NodeId, Request | Task],
                   prefixes: dict[PrefixId, Prefix]) -> None:
    for name, v in (("prompt_tokens", r.prompt_tokens),
                    ("completion_tokens", r.completion_tokens),
                    ("shared_prefix_tokens", r.shared_prefix_tokens)):
        if v < 0:
            raise IRError(f"request {r.id!r} has negative {name}")
    if r.max_new_tokens is not None:
        if r.max_new_tokens < 0:
            raise IRError(f"request {r.id!r} has negative max_new_tokens")
        if r.completion_tokens > r.max_new_tokens:
            raise IRError(
                f"request {r.id!r} generated {r.completion_tokens} tokens, past its cap of "
                f"{r.max_new_tokens}"
            )
        if r.stop == "length" and r.completion_tokens != r.max_new_tokens:
            raise IRError(
                f"request {r.id!r} stopped on length at {r.completion_tokens} tokens but its "
                f"cap is {r.max_new_tokens}"
            )
    if r.shared_prefix_tokens > r.prompt_tokens:
        raise IRError(
            f"request {r.id!r} shares {r.shared_prefix_tokens} tokens but its prompt is "
            f"{r.prompt_tokens}"
        )
    if r.prefix_parent is None:
        if r.shared_prefix_tokens:
            raise IRError(f"request {r.id!r} shares a prefix with no parent")
        return
    if r.prefix_parent in prefixes:
        full = prefixes[r.prefix_parent].tokens
    else:
        parent = by_id.get(r.prefix_parent)
        if parent is None:
            raise IRError(f"request {r.id!r} names missing prefix parent {r.prefix_parent!r}")
        if not isinstance(parent, Request):
            raise IRError(f"request {r.id!r} has a non-request prefix parent {r.prefix_parent!r}")
        full = parent.prompt_tokens + parent.completion_tokens
    if r.shared_prefix_tokens > full:
        raise IRError(
            f"request {r.id!r} shares {r.shared_prefix_tokens} tokens with {r.prefix_parent!r}, "
            f"whose full sequence is only {full}"
        )


def _check_acyclic(s: Session) -> None:
    """Iterative DFS over a real DAG: a node may wait on several others."""
    deps = {n.id: n.after for n in s.nodes()}
    done: set[NodeId] = set()
    for root in deps:
        if root in done:
            continue
        onstack = {root}
        stack = [(root, iter(deps[root]))]
        while stack:
            node, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                stack.pop()
                onstack.discard(node)
                done.add(node)
                continue
            if nxt in onstack:
                raise IRError(f"cycle through node {nxt!r} in session {s.id!r}")
            if nxt not in done:
                onstack.add(nxt)
                stack.append((nxt, iter(deps[nxt])))


def _check_spawns(w: Workload, by_id: dict[NodeId, Request | Task],
                  owner: dict[NodeId, SessionId]) -> None:
    spawner = {s.id: s.spawned_by for s in w.sessions if s.spawned_by is not None}
    for sid, node_id in spawner.items():
        node = by_id.get(node_id)
        if node is None:
            raise IRError(f"session {sid!r} is spawned by missing node {node_id!r}")
        if owner[node_id] == sid:
            raise IRError(f"session {sid!r} is spawned by its own node {node_id!r}")
        if not isinstance(node, Task):
            raise IRError(
                f"session {sid!r} is spawned by {node_id!r}, which is a request; the parent's "
                f"wait for a subagent is a task"
            )
    for sid in spawner:
        seen = {sid}
        cur = sid
        while cur in spawner:
            cur = owner[spawner[cur]]
            if cur in seen:
                raise IRError(f"spawn cycle through session {cur!r}")
            seen.add(cur)
