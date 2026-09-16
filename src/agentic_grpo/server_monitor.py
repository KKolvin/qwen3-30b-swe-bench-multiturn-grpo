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

Token accounting moved between SGLang versions and this is a live hazard: the
build in use (2026-09) publishes one ``sglang:realtime_tokens_total`` counter
split by a ``mode`` label (``prefill_compute`` / ``prefill_cache`` / ``decode``)
instead of the separate ``prompt_tokens_total`` / ``cached_tokens_total`` /
``generation_tokens_total`` counters, and serves no TTFT, inter-token or
end-to-end latency histogram on this endpoint. (Those three ARE defined in this
build's ``metrics/collector.py``, on ``TokenizerMetricsCollector`` -- they live in
the tokenizer-manager process rather than the scheduler, so whether they reach
this registry is a launch-path question, not a rename. The archived raw response
written by :class:`_ScrapeDump` settles it per run; do not assume either way.)
Run 20260908-052218 reported six srv/* fields as a clean 0.0 for two whole steps
because of this -- a zero is indistinguishable from an idle server. Lookups
therefore take a list of candidate names, and anything with no source is OMITTED
from the payload (with a one-time warning) rather than defaulted.
``tests/test_server_monitor.py`` pins this against a captured payload.

:class:`SGLangServerMonitor` polls that endpoint on a background thread (~1 req/s,
negligible) and locates the drain start as the last instant the server was still
saturated - i.e. the last sample with ``num_queue_reqs > 0`` or
``num_running_reqs >= capacity``. After that the queue is empty and the batch can
only shrink, so the GPU begins to idle.

Two things it does beyond the srv/* summary:

* **Every replica.** ``tensor_model_parallel_size < n_gpus`` makes verl run
  ``n_gpus/tp`` independent SGLang servers. All of them are scraped and merged
  (:func:`merge_replicas`); scraping one and labelling it srv/* would halve
  running_peak and gen_throughput and read like a throughput regression.
* **Everything is kept.** The parse covers the whole ``sglang:`` namespace -- ~90
  metrics, an order of magnitude more than srv/* reports -- and :class:`_ScrapeDump`
  writes each tick to ``srv-metrics-<pid>.jsonl`` beside the run's timeline, plus
  the first response verbatim. srv/* is the wandb summary; the JSONL is the data.
"""

from __future__ import annotations

import atexit
import logging
import math
import os
import re
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


# Labels that SPLIT a metric into different quantities rather than into replicas
# of the same one. Summing over tp_rank/pp_rank/moe_ep_rank is right (they are
# parallel shards of one server), but summing over these would add unlike things
# -- ``realtime_tokens_total`` carries prompt and generated tokens in the SAME
# metric, separated only by ``mode``. For each of these a label-qualified key
# ``name|label=value`` is emitted alongside the summed bare name.
_SPLIT_LABELS = ("mode", "stage")

_LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"')


def parse_prometheus(text: str) -> dict[str, float]:
    """Parse Prometheus exposition text into ``{metric_name: value}``.

    The ``sglang:`` prefix is stripped and values are summed across label sets
    (e.g. data-parallel series), which is what we want for running/queue counts.
    Non-finite samples (NaN/Inf) are skipped.

    Metrics carrying one of :data:`_SPLIT_LABELS` ALSO get a per-value key,
    ``"realtime_tokens_total|mode=decode"``, because for those the bare sum mixes
    quantities that are not the same thing. The bare name is still emitted, so
    every existing lookup keeps working.

    Histogram ``_bucket`` series are the exception: their ``le`` label is a
    CUMULATIVE threshold, so adding the buckets together is not a quantity at
    all -- ``sum(le=0.1, le=0.5, le=+Inf, ...)`` counts the same request once per
    bucket it falls under. Those are emitted ONLY as ``name|le=<threshold>`` and
    the bare ``name_bucket`` key is withheld, so nothing can read the bogus sum
    by accident. ``_sum``/``_count`` carry no ``le`` and are unaffected, which is
    what :meth:`SGLangServerMonitor._latency_breakdown` uses for histogram means.
    """
    out: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        labels = ""
        try:
            if "{" in line:
                name = line[: line.index("{")]
                labels = line[line.index("{") + 1 : line.rindex("}")]
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
        is_bucket = name.endswith("_bucket")
        if not is_bucket:
            out[name] = out.get(name, 0.0) + v
        if labels:
            for label, val in _LABEL_RE.findall(labels):
                if label in _SPLIT_LABELS or (is_bucket and label == "le"):
                    key = f"{name}|{label}={val}"
                    out[key] = out.get(key, 0.0) + v
    return out


def _first_not_none(*vals: float | None) -> float | None:
    """First non-None value. NOT ``a or b``: 0.0 is a legitimate counter delta."""
    for v in vals:
        if v is not None:
            return v
    return None


# How to combine one metric across REPLICAS (tp_size < n_gpus gives several
# independent SGLang servers, each with its own KV pool and scheduler). Summing
# is right for counts and counters -- running requests, queued requests, tokens,
# gen_throughput -- and that is the default.
#
# _MERGE_MAX: pressure gauges, where the cluster is constrained by its WORST
# replica. Averaging KV utilisation across a full replica and an empty one
# reports comfortable headroom that neither of them has.
_MERGE_MAX = frozenset(
    {"token_usage", "swa_token_usage", "mamba_usage", "pending_prealloc_token_usage", "utilization"}
)
# _MERGE_MEAN: rate/ratio gauges, already normalised per replica. Summing two
# hit rates gives 1.9; taking the max reports the luckier replica.
_MERGE_MEAN = frozenset(
    {"cache_hit_rate", "new_token_ratio", "spec_accept_rate", "spec_accept_length", "eplb_balancedness"}
)


def merge_replicas(raws: list[dict[str, float]]) -> dict[str, float]:
    """Combine per-replica /metrics snapshots into one cluster-wide dict.

    Only the merged view feeds srv/*; the per-replica dicts are written to the
    JSONL dump untouched, so a merge rule that turns out wrong for some metric
    can be recomputed offline without re-running anything.
    """
    if len(raws) == 1:
        return dict(raws[0])
    out: dict[str, float] = {}
    counts: dict[str, int] = {}
    for raw in raws:
        for k, v in raw.items():
            base = k.split("|", 1)[0]
            if base in _MERGE_MAX:
                out[k] = max(out[k], v) if k in out else v
            else:
                out[k] = out.get(k, 0.0) + v
            counts[k] = counts.get(k, 0) + 1
    for k in out:
        if k.split("|", 1)[0] in _MERGE_MEAN and counts[k]:
            out[k] /= counts[k]
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


class _ScrapeDump:
    """Persist every scrape, so the run keeps more than the dozen srv/* fields.

    The poller already parses the WHOLE ``sglang:`` namespace into
    :attr:`ServerSample.raw` -- ~90 metrics at 1 Hz -- and then throws all but a
    dozen of them away when the process exits. srv/* is the wandb summary;
    this is the underlying data, written next to the run's timeline (same
    ``AGENTIC_TIMELINE_DIR``, hence ``analysis/<run>/timeline/``) in the same
    JSONL shape, at 1 s resolution rather than Prometheus's 10 s.

    Two streams:

    * ``srv-metrics-<pid>.jsonl`` -- one line per replica per tick,
      ``{"t":, "url":, "replica":, <every parsed metric>}``. PER-REPLICA and
      unmerged on purpose: :func:`merge_replicas` has to pick sum/max/mean per
      metric, and a wrong guess there is only recoverable if the inputs survived.
    * ``srv-metrics-raw-<replica>.txt`` -- the first successful scrape verbatim.
      Which metrics a build actually EXPORTS is not the same question as which
      ones ``collector.py`` defines (the scheduler and the tokenizer manager are
      different processes and only one of them reaches this registry), and it is
      the question that decides whether ttft/e2e can ever be filled in. One
      archived response answers it for good and doubles as parser test material.

    Diagnostics: every failure disables the dump and leaves the poller running.
    """

    def __init__(self, directory: str, urls: list[str]):
        from agentic_grpo.config import int_env
        from agentic_grpo.timeline import TimelineWriter

        # 20, not the timeline's 400: at 1 line/s/replica a 400-line buffer means
        # ~3 minutes of samples lost whenever a run dies, and dying runs are
        # exactly the ones worth having the samples for.
        self._writer = TimelineWriter(
            directory, flush_every=int_env("AGENTIC_SRV_METRICS_FLUSH", 20), prefix="srv-metrics"
        )
        self.path = self._writer.path
        self.directory = directory
        self._names = {u: _replica_name(u, urls) for u in urls}
        self._raw_written: set[str] = set()
        atexit.register(self.flush)

    def write(self, t: float, url: str, raw: dict[str, float]) -> None:
        self._writer.emit([{"t": t, "url": url, "replica": self._names.get(url, url), **raw}])

    def write_raw_once(self, url: str, text: str) -> None:
        if url in self._raw_written:
            return
        self._raw_written.add(url)
        name = self._names.get(url, "0")
        path = os.path.join(self.directory, f"srv-metrics-raw-{name}.txt")
        try:
            os.makedirs(self.directory, exist_ok=True)
            with open(path, "w") as fh:
                fh.write(f"# captured {time.strftime('%Y-%m-%dT%H:%M:%S')} from {url}\n{text}")
            logger.warning("server_monitor: archived first /metrics response -> %s", path)
        except Exception:  # noqa: BLE001 - diagnostics must never break training
            logger.warning("server_monitor: could not archive raw /metrics", exc_info=True)

    def flush(self) -> None:
        self._writer.flush()


def _replica_name(url: str, urls: list[str]) -> str:
    """Stable short label for a replica: its index in the discovered order.

    The port is ephemeral (a new one every run), so it cannot name anything that
    outlives the run; the index matches verl's replica_rank ordering because
    discovery sorts the Ray actor names.
    """
    try:
        return str(urls.index(url))
    except ValueError:
        return "0"


class SGLangServerMonitor:
    """Background poller for SGLang's ``/metrics`` -> per-step drain metrics."""

    def __init__(
        self,
        metrics_urls: str | list[str],
        capacity: int | None = None,
        interval: float = 1.0,
        dump_dir: str | None = None,
    ):
        self.metrics_urls = [metrics_urls] if isinstance(metrics_urls, str) else list(metrics_urls)
        # Kept for log lines and warning messages; metrics_urls is the truth.
        self.metrics_url = ", ".join(self.metrics_urls)
        # PER-SERVER, matching SGLang's --max-running-requests: the cluster-wide
        # threshold is derived in _cluster_capacity, because `running` is summed
        # over replicas but the env var that supplies this is a launch flag.
        self.capacity = capacity
        self.interval = max(interval, 0.05)
        self._samples: list[ServerSample] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._checkpoint = 0.0  # wall time up to which samples were already summarised
        self._warned: set[str] = set()  # guard for _warn_once (a poll loop would spam)
        self._dump = _ScrapeDump(dump_dir, self.metrics_urls) if dump_dir else None

    # -- lifecycle -----------------------------------------------------
    def start(self) -> "SGLangServerMonitor":
        if self._thread is not None:
            return self
        self._checkpoint = time.time()
        self._thread = threading.Thread(target=self._run, name="sglang-metrics", daemon=True)
        self._thread.start()
        logger.info(
            "SGLang metrics poller started: %d replica(s) %s (interval=%.2fs, dump=%s)",
            len(self.metrics_urls),
            self.metrics_url,
            self.interval,
            self._dump.path if self._dump else "off",
        )
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 2.0)
            self._thread = None
        if self._dump is not None:
            self._dump.flush()

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = self._fetch()
            if sample is not None:
                with self._lock:
                    self._samples.append(sample)
            self._stop.wait(self.interval)

    def _fetch(self) -> ServerSample | None:
        """One tick: scrape every replica, dump the raw views, return the merge."""
        t = time.time()
        raws: list[dict[str, float]] = []
        for url in self.metrics_urls:
            raw = self._fetch_one(url)
            if raw is None:
                continue
            raws.append(raw)
            if self._dump is not None:
                self._dump.write(t, url, raw)
        if not raws:
            return None
        return ServerSample(t=t, raw=merge_replicas(raws))

    def _fetch_one(self, url: str) -> dict[str, float] | None:
        try:
            with _OPENER.open(url, timeout=2.0) as resp:
                text = resp.read().decode("utf-8", "replace")
        except Exception as exc:  # network hiccup / server busy -> skip this tick
            self._warn_once(f"request failed: {exc!r}", url)
            return None
        d = parse_prometheus(text)
        if "num_running_reqs" not in d:
            # A 200 that lacks the gauges means we are talking to the wrong thing
            # (a proxy, a different service) or metrics are off server-side. Both
            # used to be indistinguishable from "server idle".
            self._warn_once(
                f"response has no sglang:num_running_reqs "
                f"(first 120 chars: {text[:120]!r}) -- srv/* will stay empty",
                url,
            )
            return None
        if self._dump is not None:
            self._dump.write_raw_once(url, text)
        return d

    def _warn_once(self, msg: str, url: str | None = None) -> None:
        """Log a scrape problem the first time only; a poll loop would spam.

        Keyed by message so a per-replica failure is not masked by an unrelated
        warning that happened to fire first.
        """
        if msg in self._warned:
            return
        self._warned.add(msg)
        logger.warning("server_monitor: %s (url=%s)", msg, url or self.metrics_url)

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
        cap = capacity or max(s.running for s in samples)

        # Last instant the server was still saturated: queue not yet empty, or
        # the running batch still at/above capacity. After this it only drains.
        drain_start = phase_start
        for s in samples:
            if phase_start <= s.t <= phase_end and (s.queue > 0 or s.running >= cap):
                drain_start = s.t

        gpu_wall = max(phase_end - phase_start, 0.0)
        drain_window = max(phase_end - drain_start, 0.0)
        return {
            "srv/drain_window_s": drain_window,
            "srv/drain_ratio": (drain_window / gpu_wall) if gpu_wall > 0 else 0.0,
            "srv/gpu_busy_s": gpu_wall,
            "srv/running_peak": max(s.running for s in samples),
            "srv/token_usage_peak": max((s.token_usage for s in samples), default=0.0),
            "srv/sample_count": float(len(samples)),
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

        def pick(*keys: str) -> str | None:
            """The first of ``keys`` this server actually publishes, or None.

            SGLang renames its metrics between versions, and a missing name is
            indistinguishable from an idle server once it has been defaulted to
            0.0 -- run 20260908-052218 logged srv/ttft_mean_s, srv/num_requests,
            srv/prompt_tokens, srv/generation_tokens, srv/cached_tokens and
            srv/prefix_cache_hit_rate as a clean 0.0 for two full steps because
            the counters behind them no longer exist under those names. A metric
            with no source is now OMITTED rather than reported as zero.
            """
            for k in keys:
                if k in last.raw:
                    return k
            return None

        def dmean(sum_keys: tuple[str, ...], count_keys: tuple[str, ...]) -> float | None:
            sk, ck = pick(*sum_keys), pick(*count_keys)
            if sk is None or ck is None:
                return None
            ds = last.raw.get(sk, 0.0) - first.raw.get(sk, 0.0)
            dc = last.raw.get(ck, 0.0) - first.raw.get(ck, 0.0)
            return (ds / dc) if dc > 0 else 0.0

        def delta(*keys: str) -> float | None:
            """Counter increase over the window, summed over every key present."""
            found = [k for k in keys if k in last.raw]
            if not found:
                return None
            d = sum(last.raw.get(k, 0.0) - first.raw.get(k, 0.0) for k in found)
            return d if d >= 0 else 0.0

        busy = [s for s in samples if s.running > 0]
        tputs = [s.throughput for s in busy if s.throughput > 0]

        # Token accounting. Newer SGLang folds all three into ONE counter split by
        # a `mode` label (HELP: "mode: prefill_compute, prefill_cache, decode"),
        # which is why parse_prometheus emits label-qualified keys -- the bare
        # `realtime_tokens_total` is prompt+generated added together and is not a
        # meaningful quantity. Older builds published separate *_total counters;
        # both spellings are accepted so this works either way.
        prompt = delta(
            "realtime_tokens_total|mode=prefill_compute",
            "realtime_tokens_total|mode=prefill_cache",
        )
        if prompt is None:
            prompt = delta("prompt_tokens_total")
        cached = delta("realtime_tokens_total|mode=prefill_cache")
        if cached is None:
            cached = delta("cached_tokens_total")
        generated = delta("realtime_tokens_total|mode=decode")
        if generated is None:
            generated = delta("generation_tokens_total")

        out: dict[str, float] = {
            # queue_time_seconds is also published as per_stage_req_latency_seconds
            # {stage="prefill_waiting"} -- verified to be the same quantity (sum
            # 525.2 vs 524.6 over an identical count of 6536), so it is only a
            # fallback, not a second signal.
            "srv/queue_time_mean_s": dmean(
                ("queue_time_seconds_sum", 'per_stage_req_latency_seconds_sum|stage=prefill_waiting'),
                ("queue_time_seconds_count", 'per_stage_req_latency_seconds_count|stage=prefill_waiting'),
            ),
            # Not served on this endpoint in the build in use (2026-09) -- see the
            # module docstring: defined on TokenizerMetricsCollector, which lives
            # in another process. Left in place so they light up if they ever
            # appear; dropped from the payload meanwhile rather than logged as 0.0.
            # srv-metrics-raw-*.txt is how you check whether they have.
            "srv/ttft_mean_s": dmean(
                ("time_to_first_token_seconds_sum",), ("time_to_first_token_seconds_count",)
            ),
            "srv/inter_token_latency_mean_s": dmean(
                ("inter_token_latency_seconds_sum",), ("inter_token_latency_seconds_count",)
            ),
            "srv/e2e_latency_mean_s": dmean(
                ("e2e_request_latency_seconds_sum",), ("e2e_request_latency_seconds_count",)
            ),
            "srv/prompt_tokens": prompt,
            "srv/generation_tokens": generated,
            "srv/cached_tokens": cached,
            # From the server's own token counters. Nothing to do with
            # reward/eval_cache_hit_rate, which is the SWE-bench result cache.
            "srv/prefix_cache_hit_rate": (
                (cached / prompt) if (cached is not None and prompt) else None
            ),
            # requests admitted in the window. num_requests_total is gone in the
            # current build; queue_time_seconds_count is the same population (every
            # request is queued before prefill) and is a counter, so its delta is
            # the request count.
            "srv/num_requests": _first_not_none(
                delta("num_requests_total"), delta("queue_time_seconds_count")
            ),
            "srv/gen_throughput_mean": (sum(tputs) / len(tputs)) if tputs else 0.0,
        }
        missing = sorted(k for k, v in out.items() if v is None)
        if missing:
            self._warn_once(
                "no source on this SGLang build for "
                + ", ".join(missing)
                + " -- omitted rather than logged as 0.0"
            )
        return {k: v for k, v in out.items() if v is not None}

    def drain_summary(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        capacity: int | None = None,
    ) -> dict[str, float]:
        """Full server-side drain + latency breakdown over ``[t_start, t_end]``.

        ``capacity`` is PER-SERVER (the --max-running-requests launch flag); it is
        scaled by the replica count here because the samples are cluster-wide.
        """
        samples = self._window(t_start, t_end)
        drain = self._drain(samples, self._cluster_capacity(capacity))
        if not drain:
            return {}
        return {**drain, **self._latency_breakdown(samples)}

    def _cluster_capacity(self, capacity: int | None) -> float | None:
        """Per-server running-batch cap -> cluster-wide, or None to use the peak.

        With tensor_model_parallel_size < n_gpus verl runs several independent
        SGLang servers, each admitting up to --max-running-requests. `running` is
        summed over them, so comparing it against the per-server flag would call
        the cluster saturated at half load and report a drain that never started.
        """
        per_server = capacity if capacity is not None else self.capacity
        if per_server is None:
            return None
        return float(per_server) * len(self.metrics_urls)

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


def discover_metrics_urls() -> list[str]:
    """Resolve EVERY rollout replica's ``/metrics`` URL from verl's Ray actors.

    verl launches each SGLang replica on an EPHEMERAL port, so there is no static
    URL to configure -- which is why drain metrics silently collected nothing for
    every run before this existed. But it registers each replica as a *named* Ray
    actor, ``sglang_server_<replica_rank>_<node_rank>``, exposing
    ``get_server_address()``; that is how the rollout worker finds its own server
    (verl/workers/rollout/sglang_rollout/sglang_rollout.py). We reuse the same
    handles to build the scrape URLs.

    Returns [] (never raises) when Ray is unavailable, no replica is registered
    yet, or an address cannot be fetched -- the monitor is diagnostics, and must
    never be able to take a training run down.
    """
    try:
        import ray
    except ImportError:
        return []
    try:
        if not ray.is_initialized():
            return []
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
            return []
        # EVERY replica, not just the first. tensor_model_parallel_size < n_gpus
        # makes verl run n_gpus/tp independent SGLang servers (see
        # LLMServerManager: num_replicas = world_size // rollout_world_size), each
        # with its own scheduler and KV pool. Scraping one of them and calling the
        # result srv/* halves running_peak and gen_throughput -- which reads
        # exactly like a throughput regression, and is the sort of false signal
        # that a TP change would be judged on. Sorted so replica indices are
        # stable across runs and match verl's replica_rank ordering.
        urls: list[str] = []
        for name, namespace in sorted(names):
            # Pass the namespace explicitly: the actor may live outside this
            # process's namespace, where a bare get_actor(name) raises ValueError.
            handle = (
                ray.get_actor(name, namespace=namespace) if namespace else ray.get_actor(name)
            )
            address, port = ray.get(handle.get_server_address.remote())
            # verl brackets IPv6 literals before building URLs; match that or the
            # URL is unparseable.
            host = f"[{address}]" if ":" in str(address) else address
            urls.append(f"http://{host}:{port}/metrics")
        # WARNING not INFO: nothing configures the agentic_grpo logger below
        # WARNING in a training run, so an INFO line here is invisible and success
        # is indistinguishable from silent failure. This fires once per run.
        logger.warning(
            "server_monitor: scraping %d replica(s): %s (Ray actors %s)",
            len(urls),
            ", ".join(urls),
            ", ".join(f"{n}@{ns}" for n, ns in sorted(names)),
        )
        return urls
    except Exception as exc:  # noqa: BLE001 - diagnostics must not break training
        logger.warning("server_monitor: Ray discovery of /metrics failed: %r", exc)
        return []


def get_shared_monitor() -> SGLangServerMonitor | None:
    """Lazily build+start the shared monitor, or return None.

    The URLs come from ``AGENTIC_SGLANG_METRICS_URL`` if set (comma-separated for
    several replicas), otherwise every rollout replica is discovered from verl's
    Ray actors (see :func:`discover_metrics_urls`).

    Env:
      * ``AGENTIC_SGLANG_METRICS_URL``   - ``/metrics`` URL(s), comma-separated; overrides discovery
      * ``AGENTIC_SGLANG_METRICS``       - set to "0"/"off"/"false" to disable entirely
      * ``AGENTIC_MAX_RUNNING_REQUESTS`` - PER-SERVER capacity C (else observed peak)
      * ``AGENTIC_METRICS_POLL_INTERVAL``- seconds between scrapes (default 1.0)
      * ``AGENTIC_SRV_METRICS_DIR``      - where to dump raw scrapes; defaults to
        ``AGENTIC_TIMELINE_DIR`` so they land beside the run's timeline. Empty
        (and no timeline dir) turns the dump off; srv/* is unaffected either way.
    """
    global _SHARED, _SHARED_INIT
    if _SHARED_INIT:
        return _SHARED
    if os.environ.get("AGENTIC_SGLANG_METRICS", "").lower() in {"0", "off", "false", "no"}:
        _SHARED_INIT = True
        return None
    configured = os.environ.get("AGENTIC_SGLANG_METRICS_URL", "")
    urls = [u.strip() for u in configured.split(",") if u.strip()] or discover_metrics_urls()
    if not urls:
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
    dump_dir = os.environ.get("AGENTIC_SRV_METRICS_DIR")
    if dump_dir is None:
        dump_dir = os.environ.get("AGENTIC_TIMELINE_DIR", "")
    _SHARED = SGLangServerMonitor(
        urls, capacity=capacity, interval=interval, dump_dir=dump_dir or None
    ).start()
    return _SHARED
