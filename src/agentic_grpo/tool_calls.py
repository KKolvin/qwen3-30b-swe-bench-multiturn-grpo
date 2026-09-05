"""Sampled text -> ``(tool name, arguments)``: the strict path and the salvage path.

Strict: verl's hermes ``ToolParser`` already split ``<tool_call>{...}</tool_call>``
into ``FunctionCall(name, arguments_json)``; :func:`from_function_call` decodes
the JSON and hands it to the tool's ``normalize``.

Salvage: ``HermesToolParser.extract_tool_calls`` logs ``Failed to decode tool
call`` and *silently drops* any block whose JSON won't parse, returning an
empty list — indistinguishable from "the model made no tool call at all". The
first full-batch run died on exactly this: every drop ended the episode, so all
rewards were 0 and GRPO advantages were identically zero. The observed causes
are all recoverable, so :func:`salvage` re-reads the raw text and repairs what
it can; only then does the loop tell the model it erred.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from agentic_grpo import tools

# A parsed action. Bash arguments are always ``{"command": str}``; the edit
# tool's are whatever the model sent, validated by ``run_edit_tool``. An empty
# dict means "the call was there but nothing in it was usable" — the runner
# turns that into an error observation, never an exception.
ToolCall = tuple[str, dict]

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_COMMAND_FIELD_RE = re.compile(r'"(?:%s)"\s*:\s*"' % "|".join(tools.COMMAND_KEYS))
_COMMAND_END_RE = re.compile(r'"\s*(?=[,}])')
_NAME_FIELD_RE = re.compile(r'"name"\s*:\s*"([^"]*)"')
_JSON_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f"}


def attempted(text: str) -> bool:
    """True if the model tried to call a tool, even if the call is unusable.

    Only the opening marker is required: a response cut off mid-call has no
    closing tag, and that is a format error to report, not a finished turn.
    """
    return TOOL_CALL_OPEN in text


def unclosed(text: str) -> bool:
    """An opened, never closed block: generation was cut off inside the call."""
    return TOOL_CALL_OPEN in text and TOOL_CALL_CLOSE not in text


def unescape_json_string(raw: str) -> str:
    """Apply the escapes the model got right, leaving unknown ones literal.

    A single left-to-right pass, so ``\\\\n`` becomes a literal backslash + ``n``
    rather than a newline, and a regex like ``\\d+`` inside a bash command
    survives untouched.
    """
    return re.sub(r"\\(.)", lambda m: _JSON_ESCAPES.get(m.group(1), m.group(0)), raw, flags=re.DOTALL)


def _normalize(name: str, args: Any) -> dict | None:
    # An unknown tool name is shaped like bash so its arguments survive to the
    # "Unknown tool" observation instead of being dropped as a format error.
    return (tools.lookup(name) or tools.BASH).normalize(args)


def _bash_like(name: str) -> bool:
    return tools.lookup(name) in (None, tools.BASH)


# --- strict ------------------------------------------------------------------
def from_function_call(tool_call: Any) -> ToolCall:
    """``(name, arguments)`` from a verl ``FunctionCall``. Never raises."""
    name = getattr(tool_call, "name", "") or ""
    raw_args = getattr(tool_call, "arguments", "") or "{}"
    try:
        args = json.loads(raw_args)
    except (json.JSONDecodeError, TypeError):
        args = None
    return name, _normalize(name, args) or {}


# --- salvage -----------------------------------------------------------------
def from_obj(obj: Any) -> ToolCall | None:
    """``(name, arguments)`` from a decoded ``<tool_call>`` payload, or None.

    Accepts the near-miss shapes Qwen3 emits alongside the canonical
    ``{"name": ..., "arguments": {...}}``: ``arguments`` as a JSON string, as
    the bare command string, or absent entirely with the fields at top level —
    the last being what makes hermes raise ``KeyError: 'arguments'`` (the single
    most common failure in the first full-batch run).
    """
    if not isinstance(obj, dict):
        return None
    name = obj["name"] if isinstance(obj.get("name"), str) else tools.BASH.name
    args = obj.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            pass  # left as the bare string: bash takes it as the command, the editor refuses it
    got = _normalize(name, args)
    if got is None:
        got = _normalize(name, {k: v for k, v in obj.items() if k != "name"})
    return (name, got) if got else None


def lenient_command(block: str) -> tuple[str, str] | None:
    """Last-resort extraction of ``command`` from a block ``json.loads`` rejected.

    Takes everything from the opening quote of the ``"command"`` value to the last
    quote that closes a JSON value (one followed by ``,`` or ``}``), so embedded
    raw newlines and unescaped inner quotes survive intact — these are the
    ``Invalid control character`` / ``Invalid \\escape`` / ``Expecting ','
    delimiter`` failures. Assumes ``command`` is the block's last field, which is
    the shape Qwen3 emits; if it isn't, the command may capture trailing JSON and
    fail in bash, which still yields an error observation rather than a dead
    episode.
    """
    start = _COMMAND_FIELD_RE.search(block)
    if start is None:
        return None
    ends = [m.start() for m in _COMMAND_END_RE.finditer(block, start.end())]
    if not ends:
        return None
    name_m = _NAME_FIELD_RE.search(block)
    name = name_m.group(1) if name_m else tools.BASH.name
    return (name, unescape_json_string(block[start.end() : ends[-1]]))


def salvage(text: str) -> list[ToolCall]:
    """Recover the calls from ``<tool_call>`` blocks hermes dropped.

    Only *closed* blocks are salvaged: a block with no ``</tool_call>`` was cut
    off mid-generation, and running a truncated shell command is worse than
    reporting the format error.
    """
    calls: list[ToolCall] = []
    for block in _BLOCK_RE.findall(text):
        block = block.strip()
        try:
            got = from_obj(json.loads(block))
        except (json.JSONDecodeError, TypeError):
            got = None
        if got is None:
            # Python-dict syntax (single quotes, True/None) is not JSON but is a
            # safe literal; literal_eval evaluates no code, only literals.
            try:
                got = from_obj(ast.literal_eval(block))
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                got = None
        if got is None:
            # The regex rescue only knows bash's one field. A structured tool's
            # broken block is better reported than half-executed: its
            # ``"command"`` is a sub-command name, and the other fields carry the
            # raw newlines and quotes that broke the JSON in the first place.
            lenient = lenient_command(block)
            if lenient is not None and lenient[1] and _bash_like(lenient[0]):
                got = (lenient[0], {"command": lenient[1]})
        if got is not None:
            calls.append(got)
    return calls
