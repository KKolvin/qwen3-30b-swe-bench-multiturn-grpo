"""``str_replace_based_edit_tool``: view/create/str_replace/insert for the agent.

Why a second tool next to ``bash``. On run 20260903-002235 the model edited
files with ``sed -i`` 36k times: line-number surgery that breaks as soon as an
earlier edit shifts the file, so every edit was followed by a ``cat`` to check
it (``cat`` alone was 21.5% of tool time), and sed-with-embedded-newlines was
the single largest source of 60s timeouts. Exact-match ``old_str -> new_str``
is immune to line drift, refuses ambiguous edits instead of guessing, and
echoes the edited region back so the confirmation ``cat`` is unnecessary.

Why this schema and not our own. It is Anthropic's text-editor tool verbatim
(the same shape SWE-agent 1.0 and OpenHands expose), which is what the agent
SFT corpora Qwen3 was tuned on overwhelmingly use. Borrowing the prior is
free; inventing a name means learning it from an 11%-reward signal, and adds
one more thing to the tool-name confusion already seen (2,654 calls with the
shell command in the ``name`` field).

The file logic is pure and host-side; the container is touched only through
:func:`read_text` / :func:`write_text`, which move bytes as base64 through
``env.execute``. That sidesteps shell quoting entirely (the failure mode of the
sed calls this replaces) and keeps the submit-marker sniffing in
``DockerEnvironment._check_finished`` from ever seeing file contents.
"""

from __future__ import annotations

import base64
import posixpath
import re
import shlex
from typing import Any

EDIT_TOOL_NAME = "str_replace_based_edit_tool"
DEFAULT_CWD = "/testbed"

# A whole-file ``view`` past this many lines is cut here with a pointer to
# ``view_range``. The harness truncates observations at
# ``max_tool_response_length`` (2000 tokens on grpo_swebench.yaml) by removing
# the *middle*, which turns a long file into two disconnected fragments; a
# coherent head plus an explicit "N more lines" is the more useful thing to
# show. ~150 numbered source lines is about that token budget.
MAX_VIEW_LINES = 150
# Lines of context returned around a successful edit.
SNIPPET_CONTEXT = 4
# Line numbers listed when old_str is ambiguous. A one-word old_str in a real
# file matched 155 times; the model needs to know it is far off, not every line.
MAX_LISTED_MATCHES = 5
# ``docker exec`` passes the command as one argv element, and Linux caps a
# single argument at MAX_ARG_STRLEN = 128 KiB. Base64 inflates 4/3, so a file
# above ~96 KiB cannot be written in one ``printf``; append it in pieces.
WRITE_CHUNK_B64 = 64_000

EDIT_TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": EDIT_TOOL_NAME,
        "description": (
            "View, create and edit text files. Use this tool (not sed/cat/echo) to read and "
            "modify source files.\n"
            "* `view`: show a file with line numbers (optionally only `view_range` [start, end]), "
            "or list a directory.\n"
            "* `str_replace`: replace `old_str` with `new_str`. `old_str` must match the file text "
            "EXACTLY (including indentation and whitespace) and must occur exactly once; include "
            "enough surrounding lines to make it unique. The tool shows the edited region afterwards.\n"
            "* `insert`: insert `new_str` after line number `insert_line` (0 = top of file).\n"
            "* `create`: create a new file with `file_text`; fails if the file exists."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "str_replace", "insert"],
                    "description": "The operation to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file or directory, e.g. /testbed/src/module.py.",
                },
                "old_str": {
                    "type": "string",
                    "description": "str_replace: the exact existing text to replace (must be unique in the file).",
                },
                "new_str": {
                    "type": "string",
                    "description": "str_replace: the replacement text. insert: the text to insert.",
                },
                "file_text": {
                    "type": "string",
                    "description": "create: the full content of the new file.",
                },
                "insert_line": {
                    "type": "integer",
                    "description": "insert: line number after which to insert (0 inserts at the top).",
                },
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "view: [start_line, end_line] (1-indexed, inclusive; -1 = end of file).",
                },
            },
            "required": ["command", "path"],
        },
    },
}


class EditError(Exception):
    """A refused edit. The message is shown to the model verbatim, so it says what to do next."""


# --------------------------------------------------------------------------- #
# pure text operations
# --------------------------------------------------------------------------- #
def _numbered(lines: list[str], start: int) -> str:
    return "\n".join(f"{i:6d}\t{line}" for i, line in enumerate(lines, start))


def view_text(text: str, view_range: list[int] | None, *, max_lines: int = MAX_VIEW_LINES) -> str:
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]  # trailing newline is not an extra line
    n = len(lines)
    if view_range is None:
        if n <= max_lines:
            return _numbered(lines, 1)
        return (
            _numbered(lines[:max_lines], 1)
            + f"\n... ({n - max_lines} more lines; file has {n} lines. "
            f"Use view_range [start, end] to see a specific part.)"
        )
    if not (isinstance(view_range, list) and len(view_range) == 2 and all(isinstance(x, int) for x in view_range)):
        raise EditError("view_range must be [start_line, end_line], two integers.")
    start, end = view_range
    if end == -1:
        end = n
    if start < 1 or start > max(n, 1):
        raise EditError(f"view_range start {start} is out of bounds; the file has {n} lines.")
    if end < start:
        raise EditError(f"view_range end {end} is before start {start}.")
    end = min(end, n)
    return _numbered(lines[start - 1 : end], start)


def _snippet(text: str, center_start_line: int, center_end_line: int) -> str:
    """``SNIPPET_CONTEXT`` lines either side of the touched region, numbered."""
    lines = text.split("\n")
    lo = max(1, center_start_line - SNIPPET_CONTEXT)
    hi = min(len(lines), center_end_line + SNIPPET_CONTEXT)
    return _numbered(lines[lo - 1 : hi], lo)


def _whitespace_key(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _near_miss_hint(text: str, old_str: str) -> str:
    """Point at a whitespace-only mismatch, the way these edits usually fail.

    Compares each candidate window of the file against ``old_str`` with all runs
    of whitespace collapsed. A hit means the model has the right lines but the
    wrong indentation/tabs; telling it so beats a bare "not found".
    """
    want = _whitespace_key(old_str)
    if not want:
        return ""
    k = old_str.count("\n") + 1
    lines = text.split("\n")
    for i in range(0, max(0, len(lines) - k + 1)):
        if _whitespace_key("\n".join(lines[i : i + k])) == want:
            return (
                f" A match differing only in whitespace/indentation exists at line {i + 1}: "
                "copy old_str exactly as shown by `view`, including leading spaces and tabs."
            )
    return ""


def str_replace_text(text: str, old_str: str, new_str: str) -> tuple[str, str]:
    """Return ``(new_text, snippet)`` or raise :class:`EditError`.

    Refuses zero matches (with a whitespace hint when one applies) and multiple
    matches (with their line numbers), so an ambiguous edit never lands on the
    wrong occurrence.
    """
    if not old_str:
        raise EditError("old_str is empty; give the exact existing text to replace.")
    count = text.count(old_str)
    if count == 0:
        raise EditError("old_str was not found in the file, so nothing was changed." + _near_miss_hint(text, old_str))
    if count > 1:
        at = [text.count("\n", 0, m.start()) + 1 for m in re.finditer(re.escape(old_str), text)]
        shown = ", ".join(map(str, at[:MAX_LISTED_MATCHES]))
        if len(at) > MAX_LISTED_MATCHES:
            shown += f", ... and {len(at) - MAX_LISTED_MATCHES} more"
        raise EditError(
            f"old_str occurs {count} times (at lines {shown}); "
            "nothing was changed. Include more surrounding lines so it matches exactly once."
        )
    idx = text.index(old_str)
    new_text = text[:idx] + new_str + text[idx + len(old_str) :]
    first = text.count("\n", 0, idx) + 1
    last = first + new_str.count("\n")
    return new_text, _snippet(new_text, first, last)


def insert_text(text: str, insert_line: int, new_str: str) -> tuple[str, str]:
    lines = text.split("\n")
    trailing_nl = text.endswith("\n")
    if trailing_nl:
        lines = lines[:-1]
    n = len(lines)
    if not isinstance(insert_line, int) or insert_line < 0 or insert_line > n:
        raise EditError(f"insert_line must be between 0 and {n} (the file has {n} lines).")
    new_lines = new_str.split("\n")
    if new_lines and new_lines[-1] == "":
        new_lines = new_lines[:-1]
    lines = lines[:insert_line] + new_lines + lines[insert_line:]
    new_text = "\n".join(lines) + ("\n" if trailing_nl or not text else "")
    return new_text, _snippet(new_text, insert_line + 1, insert_line + len(new_lines))


# --------------------------------------------------------------------------- #
# container I/O (base64 both ways; no shell quoting of file contents)
# --------------------------------------------------------------------------- #
def resolve_path(path: Any, cwd: str = DEFAULT_CWD) -> str:
    if not isinstance(path, str) or not path.strip():
        raise EditError("path is required.")
    p = path.strip()
    if not p.startswith("/"):
        p = posixpath.join(cwd, p)
    return posixpath.normpath(p)


def _exec(env: Any, command: str) -> dict:
    return env.execute({"command": command})


def stat_path(env: Any, path: str) -> str:
    """``"file"``, ``"dir"`` or ``"missing"``."""
    q = shlex.quote(path)
    out = _exec(env, f"if [ -d {q} ]; then echo dir; elif [ -e {q} ]; then echo file; else echo missing; fi")
    kind = (out.get("output") or "").strip().splitlines()
    return kind[-1] if kind and kind[-1] in ("file", "dir", "missing") else "missing"


def _decode_b64_text(path: str, b64: str) -> str:
    try:
        raw = base64.b64decode(b64.strip(), validate=True)
    except (ValueError, TypeError) as exc:
        raise EditError(f"Could not read {path}: unexpected output from the container ({exc}).")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        raise EditError(f"{path} is not a UTF-8 text file; this tool only edits text files.")


def read_entry(env: Any, path: str) -> tuple[str, str | None]:
    """``(kind, text)`` in ONE exec: kind is ``"file"``/``"dir"``/``"missing"``.

    Each ``docker exec`` is ~0.25s (with the image's BASH_ENV conda activation),
    which is most of what a call costs; a separate existence check doubled it.
    The first output line is the tag, the rest is the file as base64.
    """
    q = shlex.quote(path)
    out = _exec(
        env,
        f"if [ -d {q} ]; then echo DIR; elif [ -f {q} ]; then echo FILE; base64 -w0 -- {q}; else echo MISSING; fi",
    )
    lines = (out.get("output") or "").split("\n", 1)
    tag = lines[0].strip()
    if tag == "DIR":
        return "dir", None
    if tag == "MISSING":
        return "missing", None
    if tag != "FILE" or out.get("returncode") != 0:
        raise EditError(f"Could not read {path}: {(out.get('output') or '').strip()[:300]}")
    return "file", _decode_b64_text(path, lines[1] if len(lines) > 1 else "")


def read_text(env: Any, path: str) -> str:
    kind, text = read_entry(env, path)
    if kind != "file":
        raise EditError(f"{path} is a directory." if kind == "dir" else f"{path} does not exist.")
    return text  # type: ignore[return-value]


def write_text(env: Any, path: str, text: str, *, chunk: int | None = None) -> None:
    chunk = chunk or WRITE_CHUNK_B64
    b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
    q = shlex.quote(path)
    # ``mkdir -p`` so ``create`` can make a file in a new directory; ``>`` on the
    # first chunk truncates, ``>>`` appends the rest.
    pieces = [b64[i : i + chunk] for i in range(0, len(b64), chunk)] or [""]
    for i, piece in enumerate(pieces):
        redir = ">" if i == 0 else ">>"
        prefix = f"mkdir -p {shlex.quote(posixpath.dirname(path) or '/')} && " if i == 0 else ""
        out = _exec(env, f"{prefix}printf %s {shlex.quote(piece)} | base64 -d {redir} {q}")
        if out.get("returncode") != 0:
            raise EditError(f"Could not write {path}: {(out.get('output') or '').strip()[:300]}")


def list_dir(env: Any, path: str) -> str:
    out = _exec(
        env,
        f"find {shlex.quote(path)} -maxdepth 2 -not -path '*/.*' | sort | head -200",
    )
    return (out.get("output") or "").rstrip()


# --------------------------------------------------------------------------- #
# dispatcher
# --------------------------------------------------------------------------- #
_MUTATING = ("create", "str_replace", "insert")


def normalize_args(args: Any) -> dict | None:
    """The editor validates its own fields; only a dict with a sub-command is worth passing on."""
    return args if isinstance(args, dict) and "command" in args else None


def run_edit_tool(env: Any, args: dict, *, cwd: str = DEFAULT_CWD) -> dict:
    """Execute one call; return a bash-shaped observation ``{"returncode", "output", "edit"}``.

    ``returncode`` is 0 on success and 1 on a refused/failed call, so the same
    observation template renders both. ``edit`` names the sub-command when the
    file was actually changed, for metrics; it is absent otherwise.
    """
    if not isinstance(args, dict):
        return {"returncode": 1, "output": "Arguments must be an object with `command` and `path`."}
    cmd = args.get("command")
    try:
        if cmd not in ("view", "create", "str_replace", "insert"):
            raise EditError(
                f"Unknown command {cmd!r}. Use one of: view, create, str_replace, insert."
            )
        path = resolve_path(args.get("path"), cwd)

        if cmd == "create":
            if stat_path(env, path) != "missing":
                raise EditError(f"{path} already exists; use str_replace or insert to change it.")
            file_text = args.get("file_text")
            if not isinstance(file_text, str):
                raise EditError("create requires `file_text` (the full file content).")
            write_text(env, path, file_text)
            return {"returncode": 0, "output": f"Created {path} ({file_text.count(chr(10))} lines).", "edit": cmd}

        kind, text = read_entry(env, path)
        if cmd == "view":
            if kind == "dir":
                return {"returncode": 0, "output": list_dir(env, path)}
            if kind == "missing":
                raise EditError(f"{path} does not exist.")
            return {"returncode": 0, "output": view_text(text or "", args.get("view_range"))}
        if kind == "dir":
            raise EditError(f"{path} is a directory.")
        if kind == "missing":
            raise EditError(f"{path} does not exist. Use `create` for a new file.")
        assert text is not None

        if cmd == "str_replace":
            old_str, new_str = args.get("old_str"), args.get("new_str")
            if not isinstance(old_str, str):
                raise EditError("str_replace requires `old_str`.")
            if not isinstance(new_str, str):
                new_str = ""  # deleting text is a legitimate replacement
            new_text, snippet = str_replace_text(text, old_str, new_str)
            write_text(env, path, new_text)
            return {"returncode": 0, "output": f"Edited {path}. The region now reads:\n{snippet}", "edit": cmd}

        # insert
        new_str = args.get("new_str")
        if not isinstance(new_str, str):
            raise EditError("insert requires `new_str`.")
        line = args.get("insert_line")
        if isinstance(line, str) and line.strip().lstrip("-").isdigit():
            line = int(line)
        if not isinstance(line, int) or isinstance(line, bool):
            raise EditError("insert requires an integer `insert_line`.")
        new_text, snippet = insert_text(text, line, new_str)
        write_text(env, path, new_text)
        return {"returncode": 0, "output": f"Inserted into {path}. The region now reads:\n{snippet}", "edit": cmd}
    except EditError as exc:
        return {"returncode": 1, "output": str(exc)}


def summarize_call(args: Any) -> str:
    """One line for the timeline/dump: ``str_replace /testbed/x.py``."""
    if not isinstance(args, dict):
        return "<bad args>"
    return f"{args.get('command', '?')} {args.get('path', '')}".strip()
