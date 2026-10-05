"""What the real system did: validation targets, quarantined from the inputs.

Every field here is an **output** of the thing being simulated. Per
`SIMULATOR.md` §3, feeding any of it to the simulator makes the comparison
vacuous, so it lives in its own module with one rule:

    the engine, the scheduler and the cost model must not import this module.

Only the diff harness may. That rule is mechanically checkable, which is the
whole point of the separation -- ``simulator.ir`` and ``simulator.observed``
could have been one file, and then the discipline would rest on nobody
mistyping an import.

The three record types correspond to the three diff levels of `SIMULATOR.md` §12:
:class:`ObservedGauge` alone drives the first (cost model only),
:class:`ObservedRequest` drives the paired per-request diff of the second
(scheduler), and :class:`ObservedSession` plus the gauge series drive the
emergent-structure diff of the third (full closed loop).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from simulator.ir import NodeId, SessionId


@dataclass(frozen=True)
class ObservedRequest:
    """What one inference call actually cost. Paired to a :class:`~simulator.ir.Request` by id.

    99,646 of these in run 20260912-072938, which is what makes the second-level diff a
    paired comparison rather than a distribution match.
    """

    id: NodeId

    #: Wall-clock instants, seconds, on the one clock comparable across the
    #: trainer, the workers and SGLang (``time.time()``). Kept absolute here
    #: precisely because these are targets: normalising them would lose the
    #: alignment between a request and the server state that produced it.
    t_arrival: float
    t_first_token: float
    t_last_token: float

    #: Tokens the server reported serving from its prefix cache. The simulator
    #: derives its own from the radix model; this is what it is scored against.
    #: It is reliably *less* than the structural share, by p50 113 tokens in run
    #: 20260912-072938, because matches are quantised to ``page_size`` and the
    #: last page is held back. A radix model without paging predicts 0 here.
    cached_tokens: int

    #: Which replica served it. Absent in every trace recorded before this was
    #: instrumented, which is why the second-level diff cannot yet be run
    #: (`SIMULATOR.md` §9, the one gap that costs more than a lookup).
    replica: str | None = None

    @property
    def ttft_s(self) -> float:
        return self.t_first_token - self.t_arrival

    @property
    def decode_s(self) -> float:
        return self.t_last_token - self.t_first_token


@dataclass(frozen=True)
class ObservedSession:
    """End-to-end span of one session, and the phases inside it."""

    id: SessionId
    t_start: float
    t_end: float

    #: Seconds spent waiting for a resource slot rather than doing work, by pool
    #: name. The reward-eval pool is the one that matters here: its per-worker
    #: semaphore of 4 puts ``eval_wait`` at p90 248s and max 755s, which is 14.5%
    #: of a step spent in a queue ([[step-budget-and-async-math]]).
    waits_s: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ObservedGauge:
    """One scrape of a server's own counters. Sampled at 1Hz in this repo.

    A simulator comparing against these must emit the same *windowed* quantity at
    the same cadence, not an instantaneous value -- ``gen_throughput`` in
    particular is an average over the server's stats window.

    The irreducible noise floor is known: the run carries two independent
    replicas under statistically identical load, and across 2,862 paired busy
    samples they differ by 4.0% on ``running_reqs``, 16.8% on ``gen_throughput``,
    and 2.7% on median step time. Agreement below that is not evidence.
    """

    t: float
    replica: str
    running_reqs: float
    queued_reqs: float
    sum_seq_lens: float
    token_usage: float
    gen_throughput: float
    cache_hit_rate: float
    retracted_reqs: float = 0.0
    max_total_num_tokens: float = 0.0


@dataclass(frozen=True)
class Observations:
    """Everything measured about one run of one workload.

    ``source`` must match the :attr:`~simulator.ir.Workload.source` it is diffed
    against; a mismatch means the comparison is between a workload and somebody
    else's timeline.
    """

    source: str
    requests: tuple[ObservedRequest, ...] = ()
    sessions: tuple[ObservedSession, ...] = ()
    gauges: tuple[ObservedGauge, ...] = ()

    #: Opaque blocking spans the simulator replays rather than models -- the
    #: training phases. ``(name, t_start, t_end)``, seconds.
    phases: tuple[tuple[str, float, float], ...] = ()
