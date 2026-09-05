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

import re
import shlex
from typing import Any

from agentic_grpo.config import int_env

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
# command policy
# --------------------------------------------------------------------------- #
# Commands that are structurally useless in a SWE-bench container and that the
# model cannot learn to avoid from a sparse reward, refused before they run.
# Measured on run 20260903-002235 (238,597 bash calls, 130 hit the 60s timeout):
#   install / network   781 calls, 4,969s (7.0% of tool time), 21 timeouts --
#                       the env is pre-installed; `git clone` of the upstream
#                       repo is also the model fetching the real fix.
#   servers             771 calls, 2,545s (3.6%), 37 timeouts -- `runserver`
#                       never returns, and 71 episodes tried ~11 times each.
#   whole-repo lint      44 calls,   796s (1.1%), 13 timeouts.
# Each refusal is an instant observation instead of a 60s slot hold, and the
# message says what to do instead.
_WORD = r"(?<![\w./-])"
POLICIES: tuple[tuple[re.Pattern, str], ...] = (
    (
        re.compile(
            _WORD + r"(?:pip[23]?|python[23]?\s+-m\s+pip|conda|mamba|apt-get|apt|yum|dnf|npm|yarn|poetry|uv)"
            r"\s+(?:install|download|update|upgrade|add)\b"
            r"|" + _WORD + r"git\s+clone\b"
            r"|" + _WORD + r"(?:curl|wget)\b"
        ),
        "Blocked: this environment is pre-configured and has no network access. Packages cannot be "
        "installed or downloaded and remote repositories cannot be fetched; work with what is installed.",
    ),
    (
        re.compile(_WORD + r"(?:runserver|flask\s+run|gunicorn|uvicorn|daphne|hypercorn|http\.server|SimpleHTTPServer)\b"),
        "Blocked: a long-running server never returns here and would only hit the command timeout. "
        "Exercise the code with a script or the framework's test client instead.",
    ),
    (
        re.compile(_WORD + r"(?:pylint|flake8|mypy|pyflakes|ruff\s+check)\b(?=[^|;&\n]*(?:--recursive|\s\.(?:\s|$)|\s/testbed/?(?:\s|$)))"),
        "Blocked: linting the whole repository takes minutes and hits the command timeout. "
        "Lint only the files you changed.",
    ),
)


_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")


def policy_message(command: str) -> str | None:
    """The refusal for ``command``, or None when it may run.

    Quoted substrings are masked first, so ``grep -rn "pip install" docs/`` and
    ``echo 'run pip install later'`` are not refused. The price is that
    ``bash -c "pip install x"`` slips through; it then only costs the model the
    seconds the proxy takes to fail, which is not a reward it can learn to seek.
    """
    bare = _QUOTED.sub("''", command)
    for pattern, message in POLICIES:
        if pattern.search(bare):
            return message
    return None


# The commands that count as "ran the tests": the behaviour the edit tool is
# meant to free turn budget for (0.57% of bash calls on run 20260903-002235),
# and the one kind of command that legitimately needs more than the 60s
# environment timeout. A Django or sympy test module can take minutes, and a
# test run killed at 60s teaches the model that running tests is useless.
# ``AGENTIC_TEST_TIMEOUT`` (default 300s) applies to these calls only; every
# other command keeps ``environment.timeout`` from configs/agent.yaml.
# Matched at the START of a shell segment (after ``&&``, ``;``, ``|``), past any
# ``VAR=value`` prefixes, so ``grep -rn pytest docs/`` is a search, not a test run.
TEST_CMD_RE = re.compile(
    r"^(?:(?:\S*/)?python[23]?(?:\.\d+)?\s+(?:-m\s+(?:pytest|unittest)\b|\S*runtests\.py\b|\S*manage\.py\s+test\b)"
    r"|(?:\S*/)?(?:pytest|py\.test|tox)\b"
    r"|\S*runtests\.py\b|\S*manage\.py\s+test\b)"
)
_SEGMENT_SPLIT = re.compile(r"&&|\|\||;|\||\n")
_ENV_ASSIGN = re.compile(r"^(?:\w+=\S*\s+)*")


def test_timeout() -> int:
    return int_env("AGENTIC_TEST_TIMEOUT", 300)


def is_test_command(command: str) -> bool:
    """True if any segment of ``command`` runs a test runner (see TEST_CMD_RE)."""
    for seg in _SEGMENT_SPLIT.split(_QUOTED.sub("''", command)):
        if TEST_CMD_RE.match(_ENV_ASSIGN.sub("", seg.strip())):
            return True
    return False


_TIMEOUT_MSG = (
    "Killed after {secs}s: the command did not finish. It was probably waiting for input, running a "
    "server, or looping. Do not re-run it unchanged; use a different command or a non-interactive form."
)


def _clarify_timeout(env: Any, out: dict, timeout: int | None = None) -> dict:
    """Replace mini-swe-agent's raw ``TimeoutExpired`` text with something the model acts on.

    The stock message is the ``docker exec`` argv followed by ``timed out after N
    seconds``. On run 20260903-002235, after a timeout the model re-issued the
    identical command 43 times out of 130 and the same program timed out again
    65 times: the message was not landing.
    """
    info = str(out.get("exception_info") or "")
    extra = out.get("extra") or {}
    if "timed out" not in info and extra.get("exception_type") != "TimeoutExpired":
        return out
    m = re.search(r"timed out after (\d+(?:\.\d+)?)", info)
    secs = m.group(1) if m else str(timeout or getattr(getattr(env, "config", None), "timeout", "") or "the limit")
    return {**out, "exception_info": _TIMEOUT_MSG.format(secs=secs), "timeout": True}


# --------------------------------------------------------------------------- #
# observation shaping
# --------------------------------------------------------------------------- #
# Commands whose useful output is at the END: a traceback, pytest's failure
# summary, make's first error after pages of progress. Everything else (grep,
# find, ls, git, cat, sed -n) puts its signal at the top. The loop's default is
# verl's ``middle``, which for a test run drops exactly the lines between the
# collection header and the summary -- the failures themselves.
_RUNNER_RE = re.compile(
    _WORD + r"(?:python[23]?(?:\.\d+)?|pytest|py\.test|tox|nose2?|node|make|cmake|ninja|cargo|go|"
    r"runtests\.py|manage\.py|setup\.py|bash|sh)(?![\w./-])"
)


def truncate_side(command: str) -> str:
    """Which side the loop should cut when the observation is over budget.

    ``"left"`` (keep the tail) for anything that runs a program, ``"right"``
    (keep the head) for everything else. Follows verl's naming: the side named
    is the one cut away.
    """
    return "left" if _RUNNER_RE.search(_QUOTED.sub("''", command)) else "right"


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def run_bash(env: Any, command: str) -> tuple[dict, str | None]:
    """Run one shell command in the container; ``(observation, submission-or-None)``.

    ``submission`` is non-None only when the command triggered mini-swe-agent's
    submit marker (``env.execute`` raises ``Submitted`` carrying the patch) --
    kept for the ``AGENTIC_SUBMIT_TOOL=0`` baseline; the default path is the
    ``submit`` tool.
    """
    from minisweagent.exceptions import Submitted  # type: ignore

    if not command:
        return {"returncode": -1, "output": "Missing 'command' argument in bash tool call."}, None
    denied = policy_message(command)
    if denied is not None:
        return {"returncode": -1, "output": denied, "policy": True}, None
    try:
        if is_test_command(command):
            # mini-swe-agent's execute takes a per-call timeout; only test
            # runs get the longer one (see TEST_CMD_RE).
            timeout = test_timeout()
            out = _clarify_timeout(env, env.execute({"command": command}, timeout=timeout), timeout)
        else:
            out = _clarify_timeout(env, env.execute({"command": command}))
        return {**out, "truncate_side": truncate_side(command)}, None
    except Submitted as e:
        submission = e.messages[0].get("extra", {}).get("submission", "") if e.messages else ""
        return {"returncode": 0, "output": ""}, submission


def summarize_call(args: Any) -> str:
    """One line for the timeline/dump: the command itself."""
    return args.get("command", "") if isinstance(args, dict) else "<bad args>"
