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
            with urllib.request.urlopen(self.metrics_url, timeout=2.0) as resp:
                text = resp.read().decode("utf-8", "replace")
        except Exception:  # network hiccup / server busy -> skip this tick
            return None
        d = parse_prometheus(text)
        if "num_running_reqs" not in d:
            return None
        return ServerSample(t=time.time(), raw=d)

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


def get_shared_monitor() -> SGLangServerMonitor | None:
    """Lazily build+start the shared monitor from env, or return None.

    Env:
      * ``AGENTIC_SGLANG_METRICS_URL``   - full ``/metrics`` URL (required to enable)
      * ``AGENTIC_MAX_RUNNING_REQUESTS`` - server capacity C (else observed peak)
      * ``AGENTIC_METRICS_POLL_INTERVAL``- seconds between scrapes (default 1.0)
    """
    global _SHARED, _SHARED_INIT
    if _SHARED_INIT:
        return _SHARED
    _SHARED_INIT = True
    url = os.environ.get("AGENTIC_SGLANG_METRICS_URL")
    if not url:
        return None
    cap_env = os.environ.get("AGENTIC_MAX_RUNNING_REQUESTS", "")
    interval_env = os.environ.get("AGENTIC_METRICS_POLL_INTERVAL", "")
    capacity = int(cap_env) if cap_env.isdigit() else None
    try:
        interval = float(interval_env) if interval_env else 1.0
    except ValueError:
        interval = 1.0
    _SHARED = SGLangServerMonitor(url, capacity=capacity, interval=interval).start()
    return _SHARED
