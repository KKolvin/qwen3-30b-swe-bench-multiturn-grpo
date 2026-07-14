"""Binary outcome reward for SWE-bench (SKILL 1.3).

reward = 1.0 if the issue is *resolved* (all FAIL_TO_PASS + PASS_TO_PASS tests
pass after applying the agent's patch) else 0.0.

We reuse the official ``swebench`` harness to grade patches inside the same
dockerized environment the agent acted in, so the reward signal exactly matches
the leaderboard definition of "resolved".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("agentic_grpo.reward")


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


def compute_reward(
    instance: dict,
    model_patch: str,
    *,
    run_id: str = "grpo-eval",
    timeout: int = 1800,
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

    try:
        g = _grade_with_harness(
            instance=instance,
            model_patch=model_patch,
            run_id=run_id,
            timeout=timeout,
        )
    except Exception as exc:  # pragma: no cover - harness/IO failures
        logger.warning("SWE-bench grading failed for %s: %s", instance.get("instance_id"), exc)
        return RewardResult(reward=0.0, resolved=False, eval_error=str(exc))

    return RewardResult(
        reward=1.0 if g["resolved"] else 0.0,
        resolved=g["resolved"],
        pytest_output_length=g["pytest_output_length"],
        patch_applied=g["patch_applied"],
        f2p_passed=g["f2p_passed"],
        f2p_total=g["f2p_total"],
        p2p_passed=g["p2p_passed"],
        p2p_total=g["p2p_total"],
    )


def apply_to_metrics(metrics, rr: RewardResult) -> None:
    """Copy harness grading onto a TrajectoryMetrics (single source of truth)."""
    metrics.resolved = rr.resolved
    metrics.reward = rr.reward
    metrics.patch_applied = rr.patch_applied
    metrics.f2p_passed = rr.f2p_passed
    metrics.f2p_total = rr.f2p_total
    metrics.p2p_passed = rr.p2p_passed
    metrics.p2p_total = rr.p2p_total
    metrics.pytest_output_length = rr.pytest_output_length


def _grade_with_harness(
    instance: dict,
    model_patch: str,
    run_id: str,
    timeout: int,
) -> dict:
    """Run the official SWE-bench harness and reuse the report it produces.

    We call the harness's single-instance entry point (``run_instance``), which
    applies the patch, runs the eval script in docker and grades via
    ``get_eval_report`` internally, writing the full report (resolved,
    patch-applied, per-test ``tests_status``) to ``report.json``. We then read
    that file rather than re-deriving anything ourselves. The instance image is
    already cached from the agent's rollout, so this is patch + test only.
    """
    # Imported lazily so unit tests / metric code don't require the full harness.
    import docker  # type: ignore
    from swebench.harness.constants import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION  # type: ignore
    from swebench.harness.run_evaluation import run_instance  # type: ignore
    from swebench.harness.test_spec.test_spec import make_test_spec  # type: ignore

    instance_id = instance["instance_id"]
    test_spec = make_test_spec(instance)
    prediction = {
        KEY_INSTANCE_ID: instance_id,
        KEY_MODEL: run_id,
        KEY_PREDICTION: model_patch,
    }

    client = docker.from_env()
    run_instance(
        test_spec=test_spec,
        pred=prediction,
        rm_image=False,        # reuse cached instance images across the run
        force_rebuild=False,
        client=client,
        run_id=run_id,
        timeout=timeout,
        rewrite_reports=False,
    )
    return _read_harness_report(run_id=run_id, model_name=run_id, instance_id=instance_id)


def _read_harness_report(run_id: str, model_name: str, instance_id: str) -> dict:
    """Parse the harness's own ``report.json`` (get_eval_report output).

    Returns resolved / patch-applied / FAIL_TO_PASS & PASS_TO_PASS pass counts,
    all taken straight from what the harness graded - no re-grading here.
    """
    import json

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
    }

    test_output = log_dir / LOG_TEST_OUTPUT
    if test_output.exists():
        out["pytest_output_length"] = test_output.stat().st_size

    report_path = log_dir / LOG_REPORT
    if not report_path.exists():
        return out  # eval didn't complete (e.g. patch failed to apply / timeout)

    report = json.loads(report_path.read_text()).get(instance_id, {})
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
