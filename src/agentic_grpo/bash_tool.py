"""``bash``: one shell command per call, run by mini-swe-agent's container environment.

Why a module for a tool whose runner is a dozen lines. :mod:`agentic_grpo.tools`
is the registry: it says which tools are active and adapts each to one ``Tool``
shape, and it should read that way for every tool rather than carry one tool's
implementation inline because that implementation happened to be short. So
everything specific to ``bash`` is here, exactly as everything specific to the
editor is in :mod:`agentic_grpo.editor_tool`: the prompt schema, the shaping of
what the model actually emits into ``{"command": str}``, the runner, and the
one-line summary for the timeline.

What is *not* here is the shell itself. The command runs through ``env.execute``
(mini-swe-agent's ``DockerEnvironment``: ``docker exec`` with the yaml's cwd,
timeout and env), and the submit marker is detected there too: ``env.execute``
raises ``Submitted`` carrying the final patch, which :func:`run_bash` returns as
the second element of its result so the loop ends the episode without ever
sniffing output for the marker itself.

The argument shaping is here because the model is not reliable about the
schema. Qwen3 names the field ``cmd``/``shell``/``script`` at times, sends the
command as a bare string instead of an object, or as an argv list. All of these
carry a runnable command; dropping them as format errors ended episodes early
on the first full-batch run (:mod:`agentic_grpo.tool_calls` covers the other
half of that story, blocks hermes drops entirely).
"""

from __future__ import annotations

import shlex
from typing import Any

BASH_TOOL_NAME = "bash"

# Identical to mini-swe-agent's BASH_TOOL; used when that import is unavailable.
BASH_TOOL_FALLBACK: dict = {
    "type": "function",
    "function": {
        "name": BASH_TOOL_NAME,
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to execute"}},
            "required": ["command"],
        },
    },
}


def bash_tool_schema() -> dict:
    """mini-swe-agent's ``BASH_TOOL`` when importable, else the identical fallback."""
    try:
        from minisweagent.models.utils.actions_toolcall import BASH_TOOL  # type: ignore

        return BASH_TOOL
    except Exception:
        return BASH_TOOL_FALLBACK


# --------------------------------------------------------------------------- #
# argument shaping
# --------------------------------------------------------------------------- #
# Qwen3 does not always name the field ``command``; these are the aliases seen
# in practice. Ordered, because a block may carry more than one.
COMMAND_KEYS = ("command", "cmd", "shell", "script", "bash_command")


def coerce_command(value: Any) -> str:
    """A command written as a list is argv, not JSON — join it back into a line.

    ``shlex.join`` rather than ``" ".join`` because ``["bash", "-c", "echo hi"]``
    means ``bash -c 'echo hi'``; a bare space join would drop the grouping and
    run something else entirely.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return shlex.join(value)
    return ""


def first_command(d: dict) -> str:
    """The first of :data:`COMMAND_KEYS` present in ``d`` that yields a command."""
    for k in COMMAND_KEYS:
        if k in d:
            cmd = coerce_command(d[k])
            if cmd:
                return cmd
    return ""


def normalize_args(args: Any) -> dict | None:
    """``{"command": str}`` from a dict (any alias), a bare string, or nothing."""
    if isinstance(args, str):
        return {"command": args} if args else None
    if isinstance(args, dict):
        cmd = first_command(args)
        return {"command": cmd} if cmd else None
    return None


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def run_bash(env: Any, command: str) -> tuple[dict, str | None]:
    """Run one shell command in the container; ``(observation, submission-or-None)``.

    ``submission`` is non-None only when the command triggered mini-swe-agent's
    submit marker (``env.execute`` raises ``Submitted`` carrying the patch).
    """
    from minisweagent.exceptions import Submitted  # type: ignore

    if not command:
        return {"returncode": -1, "output": "Missing 'command' argument in bash tool call."}, None
    try:
        return env.execute({"command": command}), None
    except Submitted as e:
        submission = e.messages[0].get("extra", {}).get("submission", "") if e.messages else ""
        return {"returncode": 0, "output": ""}, submission


def summarize_call(args: Any) -> str:
    """One line for the timeline/dump: the command itself."""
    return args.get("command", "") if isinstance(args, dict) else "<bad args>"
