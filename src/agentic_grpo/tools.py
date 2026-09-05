"""The tools the agent can call: one :class:`Tool` per name, in one place.

A tool is its prompt schema, how to shape the model's decoded ``arguments``
into something its runner accepts, the runner itself, a one-line summary for
the timeline, and the phrase the format-error nudge uses for its arguments.
Each tool's implementation lives in its own module (:mod:`agentic_grpo.bash_tool`,
:mod:`agentic_grpo.editor_tool`); this module only adapts them to one shape and
says which are active. ``agent_loop`` never names a tool: it advertises
:func:`schemas`, shapes arguments through :mod:`agentic_grpo.tool_calls`,
executes through :func:`run`, and words its prompts with :func:`names`,
:func:`call_shapes` and :func:`edit_tool_enabled`. Adding a tool is a new
module plus one :class:`Tool` here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from agentic_grpo import bash_tool, editor_tool, submit_tool

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
    # ``ctx`` is the SWE-bench instance row (base_commit etc.); most tools ignore it.
    run: Callable[[Any, dict, dict], RunResult]
    summarize: Callable[[dict], str]
    # How the malformed-call nudge describes this tool's arguments; it completes
    # the clause ``a "name" of "<name>" with ...`` (see :func:`call_shapes`).
    usage: str


BASH = Tool(
    name=bash_tool.BASH_TOOL_NAME,
    schema=bash_tool.bash_tool_schema(),
    normalize=bash_tool.normalize_args,
    # Resolved on the module at call time so tests can stub ``bash_tool.run_bash``.
    run=lambda env, args, ctx: bash_tool.run_bash(env, args.get("command", "")),
    summarize=bash_tool.summarize_call,
    usage='an "arguments" object holding "command"',
)

EDITOR = Tool(
    name=editor_tool.EDIT_TOOL_NAME,
    schema=editor_tool.EDIT_TOOL_SCHEMA,
    normalize=editor_tool.normalize_args,
    run=lambda env, args, ctx: (editor_tool.run_edit_tool(env, args), None),
    summarize=editor_tool.summarize_call,
    usage='its "command"/"path"/... arguments',
)

SUBMIT = Tool(
    name=submit_tool.SUBMIT_TOOL_NAME,
    schema=submit_tool.SUBMIT_TOOL_SCHEMA,
    normalize=submit_tool.normalize_args,
    run=lambda env, args, ctx: submit_tool.run_submit(env, ctx),
    summarize=submit_tool.summarize_call,
    usage='an empty "arguments" object',
)


# --------------------------------------------------------------------------- #
# the active set
# --------------------------------------------------------------------------- #
def edit_tool_enabled() -> bool:
    """``AGENTIC_EDIT_TOOL=0`` runs the bash-only harness from the same code (A/B).

    Everything that mentions a tool to the model goes through this, directly or
    via :func:`active`: the advertised schemas, the system/instance prompts (as
    the ``edit_tool`` template variable), the format-error nudges and the
    unknown-tool observation. A bash-only run therefore never hears of the editor.
    """
    return os.environ.get("AGENTIC_EDIT_TOOL", "1") != "0"


def submit_tool_enabled() -> bool:
    """``AGENTIC_SUBMIT_TOOL=0`` restores the three-step marker protocol (baseline A/B)."""
    return submit_tool.submit_tool_enabled()


def active() -> list[Tool]:
    """Tools advertised to the model this run, in prompt order."""
    out = [BASH]
    if edit_tool_enabled():
        out.append(EDITOR)
    if submit_tool_enabled():
        out.append(SUBMIT)
    return out


def lookup(name: str) -> Tool | None:
    return next((t for t in active() if t.name == name), None)


def schemas() -> list[dict]:
    return [t.schema for t in active()]


def names(quote: str = "`", sep: str = " or ") -> str:
    """The active tool names for prompt text: ```bash` or `str_replace_based_edit_tool```."""
    return sep.join(f"{quote}{t.name}{quote}" for t in active())


def call_shapes() -> str:
    """One clause per active tool, for the malformed-call nudge.

    ``a "name" of "bash" with an "arguments" object holding "command", or a "name"
    of "str_replace_based_edit_tool" with its "command"/"path"/... arguments``
    """
    return ", or ".join(f'a "name" of "{t.name}" with {t.usage}' for t in active())


def summarize(name: str, args: dict) -> str:
    """One line for the timeline/dump: the command for bash, ``view /path`` for the editor."""
    return (lookup(name) or BASH).summarize(args)


def unknown_tool_message(name: str) -> str:
    available = names(quote="'", sep=" and ")
    return (
        f"Unknown tool '{name}'. Available tools: {available}. "
        "Put shell commands in the `command` argument of a `bash` call."
    )


def run(env: Any, name: str, args: dict, ctx: dict | None = None) -> RunResult:
    """Execute one parsed call. Blocking; the loop runs it in an executor."""
    tool = lookup(name)
    if tool is None:
        return {"returncode": -1, "output": unknown_tool_message(name)}, None
    return tool.run(env, args, ctx or {})
