"""Binary outcome reward for SWE-bench (SKILL 1.3).

reward = 1.0 if the issue is *resolved* (all FAIL_TO_PASS + PASS_TO_PASS tests
pass after applying the agent's patch) else 0.0.

We reuse the official ``swebench`` harness to grade patches inside the same
dockerized environment the agent acted in, so the reward signal exactly matches
the leaderboard definition of "resolved".

**Caching.** The harness has its own cache and it is keyed on
``(run_id, model_name, instance_id)`` -- the *patch is not in the key*
(``run_evaluation.run_instance``: ``if report_path.exists(): return``). Passing a
constant ``run_id`` therefore graded each instance once, ever, and handed that
verdict to every later sample of the same instance -- across GRPO groups and
across runs. Run 20260820-023256 submitted 3870 patches and wrote 23 reports.
We now give the harness a *unique* eval id per call (so its cache can never hit)
and keep our own cache keyed on the patch bytes, where an identical patch really
does deserve an identical verdict.

**Memory cap.** The harness creates its eval container through docker-py with no
memory limit at all. On run 20261005-143426 two eval containers running the
matplotlib test suite against agent patches reached 263 + 196 GB of host RAM, on
top of the ~1.2 TB the 4-GPU param_offload trainer parks on the host, and Ray's
memory monitor killed the trainer at the start of step 2 (the agent containers
were already capped via configs/agent.yaml). ``_grade_with_harness`` now hands
the harness a ``docker.DockerClient`` proxy whose ``containers.create`` adds
``mem_limit`` (``AGENTIC_EVAL_CONTAINER_MEMORY``, default 32g, same as the agent
containers). A test run the kernel kills under that cap is graded as an eval
error (reward 0, NOT cached, ``reward/eval_oom_rate``) rather than as a failed
patch: "the tests never finished" is not a verdict on the patch.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("agentic_grpo.reward")

# Process-unique suffix. The harness derives BOTH its log dir and its container
# name (``sweb.eval.<instance>.<run_id>``) from run_id, so a shared run_id also
# means two concurrent gradings of the same instance collide on the container
# name -- which only stayed hidden because the stale cache meant the second one
# never ran.
_EVAL_SEQ = itertools.count()


def _cache_dir() -> Path:
    return Path(os.environ.get("AGENTIC_REWARD_CACHE_DIR", "logs/reward_cache"))


def _cache_enabled() -> bool:
    return os.environ.get("AGENTIC_REWARD_CACHE", "1") != "0"


def _eval_timeout() -> int:
    raw = os.environ.get("AGENTIC_EVAL_TIMEOUT", "")
    return int(raw) if raw.isdigit() else 900


def _eval_container_memory() -> str:
    """Memory cap for the harness's eval containers, in docker ``mem_limit`` syntax.

    ``AGENTIC_EVAL_CONTAINER_MEMORY``; default ``32g`` (the agent containers'
    ``--memory`` in configs/agent.yaml); ``0`` disables the cap.
    """
    raw = os.environ.get("AGENTIC_EVAL_CONTAINER_MEMORY", "32g").strip()
    return "" if raw.lower() in {"", "0", "none", "off"} else raw


def _parse_oom_kills(memory_events: str) -> int:
    """``oom_kill N`` from a cgroup v2 ``memory.events`` file; 0 if absent."""
    for line in memory_events.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "oom_kill" and parts[1].isdigit():
            return int(parts[1])
    return 0


class _OOMWatchedContainer:
    """docker-py ``Container`` proxy that reads the cgroup's OOM counter before teardown.

    ``run_instance`` tears the container down in its own ``finally``
    (``cleanup_container`` -> ``container.stop`` -> ``container.remove``) before
    control comes back to us, so the way into ``stop`` is the only moment to ask
    the cgroup what happened. ``docker inspect``'s ``State.OOMKilled`` is not
    enough on its own: it describes the container's init process
    (``tail -f /dev/null``), not the pytest the kernel actually killed;
    ``/sys/fs/cgroup/memory.events`` ``oom_kill`` counts every kill in the
    cgroup (verified on the rootless daemon: a 300 MB hog under a 64m cap ->
    exit 137, ``oom_kill 1``). Everything else is forwarded untouched.
    """

    def __init__(self, inner, sink: dict):
        self._inner = inner
        self._sink = sink

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def _record_oom(self) -> None:
        inner = self._inner
        if inner.name in self._sink:
            return
        kills = 0
        try:
            res = inner.exec_run("cat /sys/fs/cgroup/memory.events")
            kills = _parse_oom_kills(res.output.decode("utf-8", "replace"))
        except Exception as exc:  # container already gone, exec unsupported, ...
            logger.debug("memory.events unreadable for %s: %s", inner.name, exc)
        if not kills:
            try:
                inner.reload()
                if inner.attrs.get("State", {}).get("OOMKilled"):
                    kills = 1
            except Exception:  # pragma: no cover - best-effort fallback
                pass
        self._sink[inner.name] = kills

    def stop(self, *args, **kwargs):
        self._record_oom()
        return self._inner.stop(*args, **kwargs)


class _CappedContainers:
    """``client.containers`` proxy: ``create`` gets ``mem_limit``/``memswap_limit``."""

    def __init__(self, inner, mem_limit: str, sink: dict):
        self._inner = inner
        self._mem_limit = mem_limit
        self._sink = sink

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def create(self, *args, **kwargs):
        kwargs.setdefault("mem_limit", self._mem_limit)
        # memswap == mem means no swap, same as --memory-swap=--memory in agent.yaml.
        kwargs.setdefault("memswap_limit", self._mem_limit)
        return _OOMWatchedContainer(self._inner.create(*args, **kwargs), self._sink)


class _CappedDockerClient:
    """``docker.DockerClient`` proxy for the harness: capped containers, OOM bookkeeping.

    The harness's ``build_container`` calls ``client.containers.create(...)`` and
    passes nothing about memory (only ``cap_add`` from ``docker_specs``), so the
    cap has to be injected underneath it. ``oom_kills`` maps container name ->
    ``oom_kill`` count, filled in as each container is stopped. One proxy per
    grading call, so there is no shared state between the concurrent evals.
    """

    def __init__(self, client, mem_limit: str):
        self._client = client
        self.mem_limit = mem_limit
        self.oom_kills: dict[str, int] = {}
        self.containers = _CappedContainers(client.containers, mem_limit, self.oom_kills)

    def __getattr__(self, name):
        return getattr(self._client, name)


@dataclass
class RewardResult:
    reward: float
    resolved: bool
    # diagnostics surfaced into TrajectoryMetrics / logs
    empty_patch: bool = False
    eval_error: str = ""
    pytest_output_length: int = 0
    # full grading, straight from the harness's report.json (get_eval_report)
    patch_applied: bool = False
    f2p_passed: int = 0
    f2p_total: int = 0
    p2p_passed: int = 0
    p2p_total: int = 0
    # True when this verdict came from our patch-keyed cache rather than a fresh
    # container run. Watch ``reward/eval_cache_hit_rate``: a rate near 1.0 with a
    # diverse batch is the signature of the bug this module used to have.
    eval_cached: bool = False
    # The kernel OOM-killed the test run under the eval container's memory cap
    # (always paired with a non-empty eval_error; never cached).
    eval_oom: bool = False


def compute_reward(
    instance: dict,
    model_patch: str,
    *,
    run_id: str = "grpo-eval",
    timeout: int | None = None,
) -> RewardResult:
    """Grade a single predicted patch against a SWE-bench instance.

    Parameters
    ----------
    instance:
        A SWE-bench dataset row (must contain ``instance_id`` and the gold
        ``test_patch`` / ``FAIL_TO_PASS`` / ``PASS_TO_PASS`` fields).
    model_patch:
        The unified diff produced by the agent (its ``submission``).
    """
    if not model_patch or not model_patch.strip():
        # No patch submitted -> cannot resolve. Cheap short-circuit avoids
        # spinning up a container for an empty diff.
        return RewardResult(reward=0.0, resolved=False, empty_patch=True)

    instance_id = instance["instance_id"]
    patch_key = hashlib.sha256(model_patch.encode("utf-8", "replace")).hexdigest()[:16]

    hit = _cache_read(instance_id, patch_key)
    if hit is not None:
        hit.eval_cached = True
        return hit

    try:
        g = _grade_with_harness(
            instance=instance,
            model_patch=model_patch,
            run_id=run_id,
            patch_key=patch_key,
            timeout=_eval_timeout() if timeout is None else timeout,
        )
    except Exception as exc:  # pragma: no cover - harness/IO failures
        logger.warning("SWE-bench grading failed for %s: %s", instance_id, exc)
        return RewardResult(reward=0.0, resolved=False, eval_error=str(exc))

    if g["eval_error"]:
        # The harness could not produce a verdict (docker failure, test timeout).
        # That is NOT evidence the patch is wrong, so it must not be cached as
        # reward 0 -- doing so would bake one infra hiccup into every future
        # sample of this patch. Surfaces as reward/eval_error_rate.
        return RewardResult(
            reward=0.0, resolved=False, eval_error=g["eval_error"], eval_oom=g.get("eval_oom", False)
        )

    rr = RewardResult(
        reward=1.0 if g["resolved"] else 0.0,
        resolved=g["resolved"],
        pytest_output_length=g["pytest_output_length"],
        patch_applied=g["patch_applied"],
        f2p_passed=g["f2p_passed"],
        f2p_total=g["f2p_total"],
        p2p_passed=g["p2p_passed"],
        p2p_total=g["p2p_total"],
    )
    _cache_write(instance_id, patch_key, rr)
    return rr


def apply_to_metrics(metrics, rr: RewardResult) -> None:
    """Copy harness grading onto a TrajectoryMetrics (single source of truth)."""
    metrics.resolved = rr.resolved
    metrics.reward = rr.reward
    metrics.empty_patch = rr.empty_patch
    metrics.eval_error = rr.eval_error
    metrics.patch_applied = rr.patch_applied
    metrics.f2p_passed = rr.f2p_passed
    metrics.f2p_total = rr.f2p_total
    metrics.p2p_passed = rr.p2p_passed
    metrics.p2p_total = rr.p2p_total
    metrics.pytest_output_length = rr.pytest_output_length
    metrics.eval_cached = rr.eval_cached
    metrics.eval_oom = rr.eval_oom


# --------------------------------------------------------------------------- #
# Patch-keyed cache
# --------------------------------------------------------------------------- #
_CACHED_FIELDS = (
    "reward",
    "resolved",
    "pytest_output_length",
    "patch_applied",
    "f2p_passed",
    "f2p_total",
    "p2p_passed",
    "p2p_total",
)


def _cache_path(instance_id: str, patch_key: str) -> Path:
    return _cache_dir() / instance_id / f"{patch_key}.json"


def _cache_read(instance_id: str, patch_key: str) -> RewardResult | None:
    if not _cache_enabled():
        return None
    path = _cache_path(instance_id, patch_key)
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return RewardResult(**{k: raw[k] for k in _CACHED_FIELDS if k in raw})


def _cache_write(instance_id: str, patch_key: str, rr: RewardResult) -> None:
    if not _cache_enabled():
        return
    path = _cache_path(instance_id, patch_key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so a concurrent reader never sees a half file.
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({k: getattr(rr, k) for k in _CACHED_FIELDS}))
        tmp.replace(path)
    except OSError as exc:  # pragma: no cover - cache is best-effort
        logger.debug("reward cache write failed for %s/%s: %s", instance_id, patch_key, exc)


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
def _grade_with_harness(
    instance: dict,
    model_patch: str,
    run_id: str,
    patch_key: str,
    timeout: int,
) -> dict:
    """Run the official SWE-bench harness and reuse the report it produces.

    We call the harness's single-instance entry point (``run_instance``), which
    applies the patch, runs the eval script in docker and grades via
    ``get_eval_report`` internally, writing the full report (resolved,
    patch-applied, per-test ``tests_status``) to ``report.json``. We then read
    that file rather than re-deriving anything ourselves. The instance image is
    already cached from the agent's rollout, so this is patch + test only.

    ``eval_id`` is unique per call, which is load-bearing twice over: it defeats
    the harness's instance-only report cache, and it keeps the container name
    (``sweb.eval.<instance>.<eval_id>``) unique so the 8 samples of a GRPO group
    can grade the same instance concurrently.
    """
    # Imported lazily so unit tests / metric code don't require the full harness.
    import docker  # type: ignore
    from swebench.harness.constants import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION  # type: ignore
    from swebench.harness.run_evaluation import run_instance  # type: ignore
    from swebench.harness.test_spec.test_spec import make_test_spec  # type: ignore

    instance_id = instance["instance_id"]
    eval_id = f"{run_id}-{patch_key}-{os.getpid()}-{next(_EVAL_SEQ)}"
    # namespace="swebench" -> instance_image_key becomes the published name
    # `swebench/sweb.eval.x86_64.<id>_1776_...`, so run_instance pulls the prebuilt
    # image (through the daemon-level registry mirror) and reuses the exact image the
    # rollout cached, instead of building it locally from the env-image chain.
    test_spec = make_test_spec(instance, namespace="swebench")
    prediction = {
        KEY_INSTANCE_ID: instance_id,
        KEY_MODEL: eval_id,
        KEY_PREDICTION: model_patch,
    }

    mem_limit = _eval_container_memory()
    client = docker.from_env()
    if mem_limit:
        client = _CappedDockerClient(client, mem_limit)
    # The prune has to be in a finally: anything raising between here and the
    # report read (docker gone, a malformed report.json) would otherwise leave
    # the tree behind forever, which is exactly how logs/run_evaluation grew to
    # tens of MB of directories nobody reads. keep=True until proven otherwise,
    # so a raise is the case AGENTIC_KEEP_FAILED_EVAL_LOGS=1 preserves.
    keep = True
    try:
        # run_instance swallows every exception and returns completed=False, so its
        # return value is the only in-band signal that grading did not happen.
        outcome = run_instance(
            test_spec=test_spec,
            pred=prediction,
            rm_image=False,        # reuse cached instance images across the run
            force_rebuild=False,
            client=client,
            run_id=eval_id,
            timeout=timeout,
            rewrite_reports=False,
        )
        out = _read_harness_report(run_id=eval_id, model_name=eval_id, instance_id=instance_id)
        oom_kills = sum(client.oom_kills.values()) if mem_limit else 0
        if oom_kills:
            # The kernel killed the test run inside the container's cgroup. Whatever
            # the harness then parsed out of the truncated log (usually "every test
            # failed", a report.json with resolved=False) is not a verdict on the
            # patch, so it must neither count as reward 0 nor be cached. Surfaces
            # as reward/eval_oom_rate (a subset of reward/eval_error_rate); the
            # same instances showing up there step after step means a legitimate
            # suite needs more than the cap -- raise AGENTIC_EVAL_CONTAINER_MEMORY.
            out["eval_error"] = (
                f"test run OOM-killed under the {mem_limit} eval container memory cap "
                f"(oom_kill={oom_kills})"
            )
            out["eval_oom"] = True
            out["resolved"] = False
        elif not out["report_found"]:
            # No report.json means run_instance raised. Two very different causes land
            # here and only the harness's own log separates them:
            #   * the diff would not apply -- the agent wrote a bad patch. A genuine
            #     reward 0, and cacheable; counting it as an eval error would put a
            #     double-digit number into reward/eval_error_rate, the one metric that
            #     is supposed to mean "stop the run, the grader is broken".
            #   * anything else (docker, test timeout) -- a real eval error.
            if _apply_failed(eval_id, instance_id):
                out["patch_applied"] = False
            else:
                out["eval_error"] = (
                    f"harness produced no report (completed={bool(outcome and outcome.get('completed'))})"
                )
        keep = bool(out["eval_error"])
        return out
    finally:
        _prune_eval_logs(eval_id, keep=keep)


def _apply_failed(eval_id: str, instance_id: str) -> bool:
    """True if the harness log says the diff would not apply to the repo."""
    from swebench.harness.constants import (  # type: ignore
        APPLY_PATCH_FAIL,
        LOG_INSTANCE,
        RUN_EVALUATION_LOG_DIR,
    )

    log = RUN_EVALUATION_LOG_DIR / eval_id / eval_id / instance_id / LOG_INSTANCE
    try:
        return APPLY_PATCH_FAIL in log.read_text(errors="replace")
    except OSError:
        return False


def _read_harness_report(run_id: str, model_name: str, instance_id: str) -> dict:
    """Parse the harness's own ``report.json`` (get_eval_report output).

    Returns resolved / patch-applied / FAIL_TO_PASS & PASS_TO_PASS pass counts,
    all taken straight from what the harness graded - no re-grading here.
    """
    from swebench.harness.constants import (  # type: ignore
        FAIL_TO_PASS,
        LOG_REPORT,
        LOG_TEST_OUTPUT,
        PASS_TO_PASS,
        RUN_EVALUATION_LOG_DIR,
    )

    log_dir = RUN_EVALUATION_LOG_DIR / run_id / model_name.replace("/", "__") / instance_id
    out = {
        "resolved": False,
        "patch_applied": False,
        "f2p_passed": 0,
        "f2p_total": 0,
        "p2p_passed": 0,
        "p2p_total": 0,
        "pytest_output_length": 0,
        "report_found": False,
        "eval_error": "",
        "eval_oom": False,
    }

    test_output = log_dir / LOG_TEST_OUTPUT
    if test_output.exists():
        out["pytest_output_length"] = test_output.stat().st_size

    report_path = log_dir / LOG_REPORT
    if not report_path.exists():
        return out  # eval didn't complete (e.g. patch failed to apply / timeout)

    report = json.loads(report_path.read_text()).get(instance_id, {})
    out["report_found"] = True
    out["resolved"] = bool(report.get("resolved", False))
    out["patch_applied"] = bool(report.get("patch_successfully_applied", False))
    status = report.get("tests_status", {})
    for key, prefix in ((FAIL_TO_PASS, "f2p"), (PASS_TO_PASS, "p2p")):
        bucket = status.get(key, {})
        passed = len(bucket.get("success", []))
        failed = len(bucket.get("failure", []))
        out[f"{prefix}_passed"] = passed
        out[f"{prefix}_total"] = passed + failed
    return out


def _prune_eval_logs(eval_id: str, keep: bool) -> None:
    """Delete one eval's harness log tree once its report has been read.

    A unique eval_id means a fresh directory per grading -- 2048 per step, each
    holding patch.diff, eval.sh, the full pytest output and run_instance.log. We
    have already extracted everything we need into RewardResult, so the tree is
    dead weight. That went for the failures too: keeping them left 54MB of
    logs/run_evaluation that no analysis ever opened, on the 99%-full repo disk.
    Everything is pruned by default; set AGENTIC_KEEP_FAILED_EVAL_LOGS=1 to keep
    the failing trees while actually debugging a grader problem.
    """
    if keep and os.environ.get("AGENTIC_KEEP_FAILED_EVAL_LOGS", "0") != "0":
        return
    from swebench.harness.constants import RUN_EVALUATION_LOG_DIR  # type: ignore

    try:
        shutil.rmtree(RUN_EVALUATION_LOG_DIR / eval_id, ignore_errors=True)
    except OSError as exc:  # pragma: no cover - best-effort cleanup
        logger.debug("could not prune eval logs for %s: %s", eval_id, exc)
