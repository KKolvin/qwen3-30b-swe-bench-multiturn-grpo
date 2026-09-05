"""``submit``: end the episode with ``git diff`` against the base commit as the patch.

Why a tool. The mini-swe-agent protocol was three separate bash turns —
``git diff -- <files> > patch.txt``, ``cat patch.txt``, then
``echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt`` — and on run
20260903-002235 that ceremony was 17,560 calls, 7.4% of all tool calls, 2.85
per episode out of a 40-turn budget. ``cat patch.txt`` alone was the single most
repeated command (6,489x). Meanwhile 933 episodes (15%) ended ``NoToolCall``:
the model believed it was done and never said the marker. Their patches were
only saved by the git-diff fallback in ``agent_loop._release`` — which is the
proof that the ceremony adds nothing ``git diff`` does not already know.

So the harness does the diff. ``submit`` takes no arguments, reads the working
tree's diff against ``base_commit`` inside the container and returns it as the
submission; the loop ends the episode exactly as it did for the marker. An
empty diff is refused (returncode 1, episode continues) so "submit before
editing anything" is an error the model sees rather than a graded zero.

:func:`git_diff_patch` is the same function the end-of-episode fallback uses,
so a patch recovered from a ``TurnLimit`` episode and a patch the model
submitted deliberately are byte-for-byte the same view of the tree.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from agentic_grpo.config import int_env

logger = logging.getLogger("agentic_grpo.submit_tool")

SUBMIT_TOOL_NAME = "submit"

SUBMIT_TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": SUBMIT_TOOL_NAME,
        "description": (
            "Finish the task. Records `git diff` of the repository against the task's base commit as "
            "your patch and ends the session; you cannot run anything afterwards. Takes no arguments. "
            "Call it only once your fix is in place and verified. Untracked files (scripts you created) "
            "are not part of the patch; changes to test files should be reverted first."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_BASE_COMMIT_RE = re.compile(r"\A[0-9a-fA-F]{7,40}\Z")


def submit_tool_enabled() -> bool:
    """``AGENTIC_SUBMIT_TOOL=0`` restores the three-step marker protocol (baseline A/B)."""
    return os.environ.get("AGENTIC_SUBMIT_TOOL", "1") != "0"


def git_diff_patch(env: Any, instance: dict) -> str:
    """The working tree's diff against ``base_commit``, or ``""`` when there is none to trust.

    ``diff <base_commit>`` rather than a bare ``diff``, so the patch survives an
    agent that staged or committed its work; ``core.fileMode=false`` mirrors the
    harness's own eval script. Untracked files (repro scripts, patch.txt itself)
    are excluded by construction -- which is what the submission instructions in
    ``configs/agent.yaml`` ask for anyway.

    Best effort throughout: anything unexpected returns ``""`` and the caller
    keeps whatever it had (the end-of-episode fallback keeps reward 0; the
    ``submit`` tool tells the model the diff was empty).
    """
    base = str(instance.get("base_commit") or "")
    ref = base if _BASE_COMMIT_RE.match(base) else ""
    try:
        out = env.execute({"command": f"git -c core.fileMode=false diff {ref}".rstrip()})
    except Exception as exc:  # noqa: BLE001 - the container may already be gone
        logger.warning("git diff failed for %s: %s", instance.get("instance_id", "?"), exc)
        return ""
    if out.get("returncode") != 0:
        return ""
    patch = out.get("output") or ""
    if not patch.strip():
        return ""
    # A cut diff cannot apply, so an oversized one is dropped whole rather than
    # truncated: both grade 0, but only the truncated one would masquerade as a
    # real submission in patch_applied_rate.
    if len(patch) > int_env("AGENTIC_PATCH_FALLBACK_MAX_BYTES", 1_000_000):
        logger.warning(
            "patch discarded: %d bytes for %s (over cap)", len(patch), instance.get("instance_id", "?"),
        )
        return ""
    return patch


def normalize_args(args: Any) -> dict | None:
    """``submit`` has no arguments: anything the model sends is accepted as ``{}``."""
    return {}


def run_submit(env: Any, instance: dict) -> tuple[dict, str | None]:
    """``(observation, patch)`` on success; ``(observation, None)`` when there is nothing to submit."""
    patch = git_diff_patch(env, instance)
    if not patch:
        return {
            "returncode": 1,
            "output": (
                "Nothing to submit: `git diff` against the base commit is empty, so no tracked source "
                "file has been changed. Make your fix first, then call submit again."
            ),
        }, None
    files = len(re.findall(r"^diff --git ", patch, flags=re.M))
    return {"returncode": 0, "output": f"Submitted: {files} file(s) changed, {patch.count(chr(10))} lines of diff."}, patch


def summarize_call(args: Any) -> str:
    return SUBMIT_TOOL_NAME
