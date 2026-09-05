"""The tools the agent can call: one :class:`Tool` per name, in one place.

A tool is its prompt schema, how to shape the model's decoded ``arguments``
into something its runner accepts, the runner itself, and a one-line summary
for the timeline. ``agent_loop`` never names a tool: it advertises
:func:`schemas`, shapes arguments through :mod:`agentic_grpo.tool_calls`, and
executes through :func:`run`. Adding a tool is a new module plus one
:class:`Tool` here.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from typing import Any, Callable

from agentic_grpo.editor_tool import EDIT_TOOL_NAME, EDIT_TOOL_SCHEMA, run_edit_tool, summarize_call

Observation = dict
RunResult = tuple[Observation, "str | None"]  # (observation, submission-or-None)


@dataclass(frozen=True)
class Tool:
    name: str
    schema: dict
    # Decoded ``arguments`` in whatever shape the model produced (dict, bare
    # string, list, None) -> the runner's args, or None when nothing usable is
    # there. Never raises.
    normalize: Callable[[Any], "dict | None"]
    run: Callable[[Any, dict], RunResult]
    summarize: Callable[[dict], str]


# --------------------------------------------------------------------------- #
# bash
# --------------------------------------------------------------------------- #
# Identical to mini-swe-agent's BASH_TOOL; used when that import is unavailable.
BASH_TOOL_FALLBACK = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to execute"}},
            "required": ["command"],
        },
    },
}


def bash_tool_schema() -> dict:
    try:
        from minisweagent.models.utils.actions_toolcall import BASH_TOOL  # type: ignore

        return BASH_TOOL
    except Exception:
        return BASH_TOOL_FALLBACK


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


def normalize_bash(args: Any) -> dict | None:
    """``{"command": str}`` from a dict (any alias), a bare string, or nothing."""
    if isinstance(args, str):
        return {"command": args} if args else None
    if isinstance(args, dict):
        cmd = first_command(args)
        return {"command": cmd} if cmd else None
    return None


def run_bash(env: Any, command: str) -> RunResult:
    """Run one shell command in the container.

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


BASH = Tool(
    name="bash",
    schema=bash_tool_schema(),
    normalize=normalize_bash,
    # Looked up at call time so tests can stub ``run_bash`` on this module.
    run=lambda env, args: run_bash(env, args.get("command", "")),
    summarize=lambda args: args.get("command", ""),
)


# --------------------------------------------------------------------------- #
# str_replace_based_edit_tool
# --------------------------------------------------------------------------- #
def normalize_editor(args: Any) -> dict | None:
    """The editor validates its own fields; only a dict with a sub-command is worth passing on."""
    return args if isinstance(args, dict) and "command" in args else None


EDITOR = Tool(
    name=EDIT_TOOL_NAME,
    schema=EDIT_TOOL_SCHEMA,
    normalize=normalize_editor,
    run=lambda env, args: (run_edit_tool(env, args), None),
    summarize=summarize_call,
)


# --------------------------------------------------------------------------- #
# the active set
# --------------------------------------------------------------------------- #
def edit_tool_enabled() -> bool:
    """``AGENTIC_EDIT_TOOL=0`` runs the bash-only harness from the same code (A/B)."""
    return os.environ.get("AGENTIC_EDIT_TOOL", "1") != "0"


def active() -> list[Tool]:
    """Tools advertised to the model this run, in prompt order."""
    return [BASH, EDITOR] if edit_tool_enabled() else [BASH]


def lookup(name: str) -> Tool | None:
    return next((t for t in active() if t.name == name), None)


def schemas() -> list[dict]:
    return [t.schema for t in active()]


def summarize(name: str, args: dict) -> str:
    """One line for the timeline/dump: the command for bash, ``view /path`` for the editor."""
    return (lookup(name) or BASH).summarize(args)


def unknown_tool_message(name: str) -> str:
    names = " and ".join(f"'{t.name}'" for t in active())
    return (
        f"Unknown tool '{name}'. Available tools: {names}. "
        "Put shell commands in the `command` argument of a `bash` call."
    )


def run(env: Any, name: str, args: dict) -> RunResult:
    """Execute one parsed call. Blocking; the loop runs it in an executor."""
    tool = lookup(name)
    if tool is None:
        return {"returncode": -1, "output": unknown_tool_message(name)}, None
    return tool.run(env, args)
