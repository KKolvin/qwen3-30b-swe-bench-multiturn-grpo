"""First-hand rollout drain signal, scraped from SGLang's own metrics.

The drain phase (when the inference server can no longer keep the GPU saturated
and the running batch starts decaying to idle) is authoritative only at the
*server*. Inferring it from client-side request timing conflates queued requests
with running ones. SGLang publishes the truth on its Prometheus ``/metrics``
endpoint (launch with ``--enable-metrics``):

* ``sglang:num_running_reqs`` - the actual running batch (GPU occupancy)
* ``sglang:num_queue_reqs``   - the waiting queue depth
* ``sglang:gen_throughput``   - decode tokens/s
* ``sglang:token_usage``      - KV-cache utilisation

:class:`SGLangServerMonitor` polls that endpoint on a background thread (~1 req/s,
negligible) and locates the drain start as the last instant the server was still
saturated - i.e. the last sample with ``num_queue_reqs > 0`` or
``num_running_reqs >= capacity``. After that the queue is empty and the batch can
only shrink, so the GPU begins to idle.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger("agentic_grpo.server_monitor")

_PREFIX = "sglang:"

# Scrape the rollout server DIRECTLY, never through an HTTP proxy. urllib honours
# http_proxy/HTTP_PROXY from the environment, and this host exports
# HTTP_PROXY=http://127.0.0.1:12233 with no_proxy covering only localhost -- so
# every scrape of the node IP was proxied and came back 404, which _fetch could
# not distinguish from an idle server. The rollout server is always on the
# cluster's own network, so a proxy is never correct here.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def parse_prometheus(text: str) -> dict[str, float]:
    """Parse Prometheus exposition text into ``{metric_name: value}``.

    The ``sglang:`` prefix is stripped and values are summed across label sets
    (e.g. data-parallel series), which is what we want for running/queue counts.
    Non-finite samples (NaN/Inf) are skipped.
    """
    out: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            if "{" in line:
                name = line[: line.index("{")]
                value = line.rsplit(None, 1)[1]
            else:
                name, value = line.split(None, 1)
        except (ValueError, IndexError):
            continue
        name = name.strip()
        if name.startswith(_PREFIX):
            name = name[len(_PREFIX) :]
        try:
            v = float(value.strip())
        except ValueError:
            continue
        if not math.isfinite(v):
            continue
        out[name] = out.get(name, 0.0) + v
    return out


@dataclass
class ServerSample:
    t: float                  # true wall clock (time.time)
    raw: dict[str, float]     # full parsed /metrics snapshot

    @property
    def running(self) -> float:
        return self.raw.get("num_running_reqs", 0.0)

    @property
    def queue(self) -> float:
        return self.raw.get("num_queue_reqs", 0.0) + self.raw.get("num_grammar_queue_reqs", 0.0)

    @property
    def throughput(self) -> float:
        return self.raw.get("gen_throughput", 0.0)

    @property
    def token_usage(self) -> float:
        return self.raw.get("token_usage", 0.0)


class SGLangServerMonitor:
    """Background poller for SGLang's ``/metrics`` -> per-step drain metrics."""

    def __init__(self, metrics_url: str, capacity: int | None = None, interval: float = 1.0):
        self.metrics_url = metrics_url
        self.capacity = capacity
        self.interval = max(interval, 0.05)
        self._samples: list[ServerSample] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._checkpoint = 0.0  # wall time up to which samples were already summarised
        self._warned = False  # one-shot guard for _warn_once (a poll loop would spam)

    # -- lifecycle -----------------------------------------------------
    def start(self) -> "SGLangServerMonitor":
        if self._thread is not None:
            return self
        self._checkpoint = time.time()
        self._thread = threading.Thread(target=self._run, name="sglang-metrics", daemon=True)
        self._thread.start()
        logger.info("SGLang metrics poller started: %s (interval=%.2fs)", self.metrics_url, self.interval)
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 2.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = self._fetch()
            if sample is not None:
                with self._lock:
                    self._samples.append(sample)
            self._stop.wait(self.interval)

    def _fetch(self) -> ServerSample | None:
        try:
            with _OPENER.open(self.metrics_url, timeout=2.0) as resp:
                text = resp.read().decode("utf-8", "replace")
        except Exception as exc:  # network hiccup / server busy -> skip this tick
            self._warn_once(f"request failed: {exc!r}")
            return None
        d = parse_prometheus(text)
        if "num_running_reqs" not in d:
            # A 200 that lacks the gauges means we are talking to the wrong thing
            # (a proxy, a different service) or metrics are off server-side. Both
            # used to be indistinguishable from "server idle".
            self._warn_once(
                f"response has no sglang:num_running_reqs "
                f"(first 120 chars: {text[:120]!r}) -- srv/* will stay empty"
            )
            return None
        return ServerSample(t=time.time(), raw=d)

    def _warn_once(self, msg: str) -> None:
        """Log a scrape problem the first time only; a poll loop would spam."""
        if not self._warned:
            self._warned = True
            logger.warning("server_monitor: %s (url=%s)", msg, self.metrics_url)

    # -- analysis ------------------------------------------------------
    def _snapshot(self) -> list[ServerSample]:
        with self._lock:
            return list(self._samples)

    def _window(self, t_start: float | None, t_end: float | None) -> list[ServerSample]:
        return [
            s
            for s in self._snapshot()
            if (t_start is None or s.t >= t_start) and (t_end is None or s.t <= t_end)
        ]

    def _drain(self, samples: list[ServerSample], capacity: int | None) -> dict[str, float]:
        """Drain metrics: when the server stopped saturating the GPU (ground truth).

        Restricts to the busy region (``running > 0``) so the trailing training
        phase (no generation) is excluded automatically.
        """
        busy = [s for s in samples if s.running > 0]
        if not busy:
            return {}
        phase_start, phase_end = busy[0].t, busy[-1].t
        cap = capacity or self.capacity or max(s.running for s in samples)

        # Last instant the server was still saturated: queue not yet empty, or
        # the running batch still at/above capacity. After this it only drains.
        drain_start = phase_start
        for s in samples:
            if phase_start <= s.t <= phase_end and (s.queue > 0 or s.running >= cap):
                drain_start = s.t

        gpu_wall = max(phase_end - phase_start, 0.0)
        drain_window = max(phase_end - drain_start, 0.0)
        return {
            "srv/drain_start_offset_s": drain_start - phase_start,
            "srv/drain_window_s": drain_window,
            "srv/drain_ratio": (drain_window / gpu_wall) if gpu_wall > 0 else 0.0,
            "srv/gpu_busy_s": gpu_wall,
            "srv/running_peak": max(s.running for s in samples),
            "srv/capacity": float(cap),
            "srv/token_usage_peak": max((s.token_usage for s in samples), default=0.0),
            "srv/samples": float(len(samples)),
        }

    def _latency_breakdown(self, samples: list[ServerSample]) -> dict[str, float]:
        """Server-side latency + token breakdown over the window (ground truth).

        Histogram means come from delta(_sum)/delta(_count) between the first and
        last sample; counters from their delta. This is the authoritative
        prefill-vs-decode-vs-queue split, with no client-side timing at all.
        """
        if len(samples) < 2:
            return {}
        first, last = samples[0], samples[-1]

        def dmean(sum_key: str, count_key: str) -> float:
            ds = last.raw.get(sum_key, 0.0) - first.raw.get(sum_key, 0.0)
            dc = last.raw.get(count_key, 0.0) - first.raw.get(count_key, 0.0)
            return (ds / dc) if dc > 0 else 0.0

        def delta(key: str) -> float:
            d = last.raw.get(key, 0.0) - first.raw.get(key, 0.0)
            return d if d >= 0 else 0.0

        busy = [s for s in samples if s.running > 0]
        tputs = [s.throughput for s in busy if s.throughput > 0]
        hit_rates = [s.raw.get("cache_hit_rate", 0.0) for s in busy]
        prompt = delta("prompt_tokens_total")
        cached = delta("cached_tokens_total")
        return {
            # prefill vs decode vs queue -- all first-hand from the server
            "srv/ttft_mean_s": dmean("time_to_first_token_seconds_sum", "time_to_first_token_seconds_count"),
            "srv/inter_token_latency_mean_s": dmean(
                "inter_token_latency_seconds_sum", "inter_token_latency_seconds_count"
            ),
            "srv/e2e_latency_mean_s": dmean(
                "e2e_request_latency_seconds_sum", "e2e_request_latency_seconds_count"
            ),
            "srv/queue_time_mean_s": dmean("queue_time_seconds_sum", "queue_time_seconds_count"),
            # token accounting from the server's own counters
            "srv/prompt_tokens": prompt,
            "srv/generation_tokens": delta("generation_tokens_total"),
            "srv/cached_tokens": cached,
            "srv/prefix_cache_hit_frac": (cached / prompt) if prompt > 0 else 0.0,
            "srv/num_requests": delta("num_requests_total"),
            "srv/gen_throughput_mean": (sum(tputs) / len(tputs)) if tputs else 0.0,
            "srv/cache_hit_rate_mean": (sum(hit_rates) / len(hit_rates)) if hit_rates else 0.0,
        }

    def drain_summary(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        capacity: int | None = None,
    ) -> dict[str, float]:
        """Full server-side drain + latency breakdown over ``[t_start, t_end]``."""
        samples = self._window(t_start, t_end)
        drain = self._drain(samples, capacity)
        if not drain:
            return {}
        return {**drain, **self._latency_breakdown(samples)}

    def summarize_since_last(self, capacity: int | None = None) -> dict[str, float]:
        """Server-side metrics for samples since the previous call; advances + trims."""
        now = time.time()
        out = self.drain_summary(t_start=self._checkpoint, t_end=now, capacity=capacity)
        self._checkpoint = now
        with self._lock:
            self._samples = [s for s in self._samples if s.t >= now]
        return out


# ---------------------------------------------------------------------------
# Shared singleton (env-configured), used by both the verl patch and standalone.
# ---------------------------------------------------------------------------
_SHARED: SGLangServerMonitor | None = None
_SHARED_INIT = False


def metrics_url_from_base(base_url: str) -> str:
    """Derive the ``/metrics`` URL from an OpenAI-style base url (.../v1)."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return f"{root}/metrics"


def discover_metrics_url() -> str | None:
    """Resolve a rollout replica's ``/metrics`` URL from verl's Ray actors.

    verl launches each SGLang replica on an EPHEMERAL port, so there is no static
    URL to configure -- which is why drain metrics silently collected nothing for
    every run before this existed. But it registers each replica as a *named* Ray
    actor, ``sglang_server_<replica_rank>_<node_rank>``, exposing
    ``get_server_address()``; that is how the rollout worker finds its own server
    (verl/workers/rollout/sglang_rollout/sglang_rollout.py). We reuse the same
    handle to build the scrape URL.

    Returns None (never raises) when Ray is unavailable, no replica is registered
    yet, or the address cannot be fetched -- the monitor is diagnostics, and must
    never be able to take a training run down.
    """
    try:
        import ray
    except ImportError:
        return None
    try:
        if not ray.is_initialized():
            return None
        # Reward-model and teacher servers share the prefix; exclude them so we
        # scrape the ROLLOUT engine whose drain phase we actually care about.
        # all_namespaces=True: the rollout servers are registered by verl's worker
        # actors, which are not guaranteed to share this process's Ray namespace.
        # Scoping to the current namespace can make discovery return nothing --
        # which is exactly how this silently collected zero drain metrics.
        try:
            visible = ray.util.list_named_actors(all_namespaces=True)
        except TypeError:  # older ray without the kwarg
            visible = ray.util.list_named_actors()
        # all_namespaces=True yields dicts {name, namespace}; the scoped call
        # yields plain strings. Normalise to (name, namespace) pairs.
        pairs = [
            (a["name"], a.get("namespace")) if isinstance(a, dict) else (a, None)
            for a in visible
        ]
        names = [
            (n, ns)
            for n, ns in pairs
            if n.startswith("sglang_server_")
            and not n.startswith(("sglang_server_reward_", "sglang_server_teacher_"))
        ]
        if not names:
            # WARNING, not silence: this is the branch that made drain metrics
            # vanish without a trace. Print what IS registered so a failed run is
            # diagnosable instead of just empty.
            logger.warning(
                "server_monitor: no rollout sglang_server_* actor found; drain "
                "metrics disabled this step. Visible named actors: %s",
                ", ".join(f"{n}@{ns}" for n, ns in sorted(pairs)) or "(none)",
            )
            return None
        # One URL per monitor, so with several replicas (TP < n_gpus) this samples
        # ONE of them: running/queue counts are then per-replica, not cluster-wide.
        # Say so rather than let the numbers look global.
        if len(names) > 1:
            logger.warning(
                "server_monitor: %d rollout replicas registered (%s); scraping only %s -- "
                "srv/* metrics describe that replica, not the whole cluster.",
                len(names),
                ", ".join(f"{n}@{ns}" for n, ns in sorted(names)),
                sorted(names)[0][0],
            )
        name, namespace = sorted(names)[0]
        # Pass the namespace explicitly: the actor may live outside this process's
        # namespace, in which case a bare get_actor(name) raises ValueError.
        handle = (
            ray.get_actor(name, namespace=namespace) if namespace else ray.get_actor(name)
        )
        address, port = ray.get(handle.get_server_address.remote())
        # verl brackets IPv6 literals before building URLs; match that or the URL
        # is unparseable.
        host = f"[{address}]" if ":" in str(address) else address
        url = f"http://{host}:{port}/metrics"
        # WARNING not INFO: nothing configures the agentic_grpo logger below
        # WARNING in a training run, so an INFO line here is invisible and success
        # is indistinguishable from silent failure. This fires once per run.
        logger.warning("server_monitor: scraping %s (Ray actor %s@%s)", url, name, namespace)
        return url
    except Exception as exc:  # noqa: BLE001 - diagnostics must not break training
        logger.warning("server_monitor: Ray discovery of /metrics failed: %r", exc)
        return None


def get_shared_monitor() -> SGLangServerMonitor | None:
    """Lazily build+start the shared monitor, or return None.

    The URL comes from ``AGENTIC_SGLANG_METRICS_URL`` if set, otherwise it is
    discovered from verl's Ray actors (see :func:`discover_metrics_url`).

    Env:
      * ``AGENTIC_SGLANG_METRICS_URL``   - full ``/metrics`` URL; overrides discovery
      * ``AGENTIC_SGLANG_METRICS``       - set to "0"/"off"/"false" to disable entirely
      * ``AGENTIC_MAX_RUNNING_REQUESTS`` - server capacity C (else observed peak)
      * ``AGENTIC_METRICS_POLL_INTERVAL``- seconds between scrapes (default 1.0)
    """
    global _SHARED, _SHARED_INIT
    if _SHARED_INIT:
        return _SHARED
    if os.environ.get("AGENTIC_SGLANG_METRICS", "").lower() in {"0", "off", "false", "no"}:
        _SHARED_INIT = True
        return None
    url = os.environ.get("AGENTIC_SGLANG_METRICS_URL") or discover_metrics_url()
    if not url:
        # Deliberately do NOT latch here. This is called once per step, and on the
        # very first call the replica may not be registered yet; latching would
        # disable drain metrics for the entire run over a startup race.
        return None
    _SHARED_INIT = True
    cap_env = os.environ.get("AGENTIC_MAX_RUNNING_REQUESTS", "")
    interval_env = os.environ.get("AGENTIC_METRICS_POLL_INTERVAL", "")
    capacity = int(cap_env) if cap_env.isdigit() else None
    try:
        interval = float(interval_env) if interval_env else 1.0
    except ValueError:
        interval = 1.0
    _SHARED = SGLangServerMonitor(url, capacity=capacity, interval=interval).start()
    return _SHARED
