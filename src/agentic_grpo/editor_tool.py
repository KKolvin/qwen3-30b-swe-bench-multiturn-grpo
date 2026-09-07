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
import difflib
import posixpath
import re
import shlex
from typing import Any

from agentic_grpo.config import int_env

EDIT_TOOL_NAME = "str_replace_based_edit_tool"
DEFAULT_CWD = "/testbed"

# The editor sizes its own observations and marks them ``bounded`` so the loop
# does not cut them again. The loop's cap (``max_tool_response_length``, 2000
# CHARS on grpo_swebench.yaml, ~500 tokens) is sized for a bash observation; a
# blind middle cut on a ``view`` turned a 150-line read into 12 head + 12 tail
# lines with no idea what was between (measured 2026-09-05). Reading code is
# the point of this tool, so it gets its own, larger budget: ~6000 chars is
# ~1500 tokens, about 75-100 numbered source lines, and a whole-file ``view``
# or a too-wide ``view_range`` is cut at a line boundary with the remaining
# line count and a pointer to ``view_range``. ``AGENTIC_EDIT_VIEW_CHARS``
# overrides it; watch tokens/mean_observation and traj/truncation_rate if raising it.
MAX_VIEW_CHARS = int_env("AGENTIC_EDIT_VIEW_CHARS", 6000)
# Lines of context returned around a successful edit.
SNIPPET_CONTEXT = 4
# Most lines quoted back in a failure message: a bounded coherent excerpt the
# model can copy old_str from, kept well inside MAX_VIEW_CHARS.
MAX_ERROR_CONTEXT_LINES = 40
# Occurrences shown with context when old_str is ambiguous, and the context each gets.
MAX_SHOWN_MATCHES = 3
AMBIGUOUS_CONTEXT = 2
# Minimum difflib ratio for a fuzzy single-line anchor when no line matches exactly.
FUZZY_ANCHOR_RATIO = 0.6
# ``view`` output prefix: right-aligned number then a tab. Stripped from old_str
# when every line carries it (see strip_line_numbers).
_LINE_NUMBER_PREFIX = re.compile(r"^[ \t]*\d+\t")
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
            "enough surrounding lines to make it unique; do NOT include the line-number prefixes that "
            "`view` shows. The tool shows the edited region afterwards, and on a failed match it "
            "quotes the file's actual text at the likely spot so you can retry without another view.\n"
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


def _numbered_within(lines: list[str], start: int, max_chars: int, max_lines: int | None) -> tuple[str, int]:
    """Number ``lines`` from ``start``, keeping whole lines while under both caps.

    Returns ``(text, shown)``. At least one line is always shown so a single
    very long line still produces output rather than an empty view.
    """
    limit = len(lines) if max_lines is None else min(len(lines), max_lines)
    shown, used = 0, 0
    while shown < limit:
        cost = len(lines[shown]) + 8  # 6-wide number + tab + newline
        if shown and used + cost > max_chars:
            break
        used += cost
        shown += 1
    return _numbered(lines[:shown], start), shown


def view_text(
    text: str,
    view_range: list[int] | None,
    *,
    max_chars: int | None = None,
    max_lines: int | None = None,
) -> str:
    """Numbered lines of ``text``, whole-file or ``view_range``, within the view budget.

    A cut always falls on a line boundary and ends with how many lines remain
    and how to see them, so the model pages instead of re-reading. The default
    budget is :data:`MAX_VIEW_CHARS`; ``max_lines`` is an extra line cap for
    callers that want one.
    """
    max_chars = MAX_VIEW_CHARS if max_chars is None else max_chars
    lines = _file_lines(text)
    n = len(lines)
    if view_range is None:
        out, shown = _numbered_within(lines, 1, max_chars, max_lines)
        if shown >= n:
            return out
        return (
            out + f"\n... ({n - shown} more lines; file has {n} lines. "
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
    out, shown = _numbered_within(lines[start - 1 : end], start, max_chars, max_lines)
    if shown >= end - start + 1:
        return out
    last = start + shown - 1
    return (
        out + f"\n... ({end - last} more lines of the requested range not shown; showing {start}-{last} "
        f"of {start}-{end}. Ask for a smaller view_range, e.g. [{last + 1}, {min(end, last + shown)}].)"
    )


def _file_lines(text: str) -> list[str]:
    """Lines of ``text``; a trailing newline does not count as an extra empty line."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return lines


def _snippet(text: str, center_start_line: int, center_end_line: int) -> str:
    """``SNIPPET_CONTEXT`` lines either side of the touched region, numbered."""
    lines = _file_lines(text)
    lo = max(1, center_start_line - SNIPPET_CONTEXT)
    hi = min(len(lines), center_end_line + SNIPPET_CONTEXT)
    return _numbered(lines[lo - 1 : hi], lo)


def _whitespace_key(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _window(text: str, first: int, last: int, *, context: int = SNIPPET_CONTEXT) -> str:
    """Numbered lines ``first..last`` (1-indexed, inclusive) plus ``context`` either side, capped.

    Failure messages embed this so the model can fix ``old_str`` in the same
    turn instead of spending a ``view`` call on it. Capped at
    ``MAX_ERROR_CONTEXT_LINES`` because the harness truncates long observations
    by cutting out the *middle*, which would leave the shown text incoherent.
    """
    lines = _file_lines(text)
    lo = max(1, first - context)
    hi = min(len(lines), last + context)
    shown = lines[lo - 1 : hi]
    tail = ""
    if len(shown) > MAX_ERROR_CONTEXT_LINES:
        shown = shown[:MAX_ERROR_CONTEXT_LINES]
        tail = f"\n... ({hi - lo + 1 - MAX_ERROR_CONTEXT_LINES} more lines not shown)"
    return _numbered(shown, lo) + tail


def strip_line_numbers(s: str) -> str | None:
    """Remove ``view``-style ``NNN<TAB>`` prefixes when *every* non-blank line has one, else ``None``.

    The prefix exists only in ``view`` output, but a model that copies from
    there pastes it into ``old_str``, and the whitespace hint cannot catch that
    (digits survive whitespace collapsing). Requiring the prefix on every line
    keeps a real ``"1\tfoo"`` in source from being mangled; a partly numbered
    ``old_str`` is treated as literal text.
    """
    lines = s.split("\n")
    content = [ln for ln in lines if ln.strip()]
    if not content or not all(_LINE_NUMBER_PREFIX.match(ln) for ln in content):
        return None
    return "\n".join(_LINE_NUMBER_PREFIX.sub("", ln, count=1) if ln.strip() else ln for ln in lines)


def _match_lines(text: str, old_str: str) -> list[int]:
    """1-indexed first line of every occurrence of ``old_str`` in ``text``."""
    return [text.count("\n", 0, m.start()) + 1 for m in re.finditer(re.escape(old_str), text)]


def _closest_region(text: str, old_str: str) -> tuple[int, int] | None:
    """Best guess at the lines the model meant, as ``(first, last)`` 1-indexed, or ``None``.

    Anchors on the longest non-blank line of ``old_str``: an exact
    whitespace-stripped hit in the file if there is one, else the most similar
    line above a 0.6 ratio. Among several hits, the one whose surrounding window
    best resembles the whole ``old_str`` wins. Single-line comparisons keep this
    linear in the file length; a window-by-window diff of a 5k-line file is not.
    """
    want = old_str.split("\n")
    if want and want[-1] == "":
        want = want[:-1]
    k = len(want)
    anchors = sorted(((ln.strip(), j) for j, ln in enumerate(want) if ln.strip()), key=lambda t: -len(t[0]))
    if not anchors:
        return None
    lines = text.split("\n")
    stripped = [ln.strip() for ln in lines]

    hits: list[tuple[int, int]] = []  # (file line index, offset of anchor inside old_str)
    for anchor, j in anchors[:3]:
        hits = [(i, j) for i, s in enumerate(stripped) if s == anchor]
        if hits:
            break
    if not hits:
        anchor, j = anchors[0]
        sm = difflib.SequenceMatcher(None, "", anchor, autojunk=False)
        best, best_ratio = -1, FUZZY_ANCHOR_RATIO
        for i, s in enumerate(stripped):
            if not s:
                continue
            sm.set_seq1(s)
            if sm.real_quick_ratio() < best_ratio or sm.quick_ratio() < best_ratio:
                continue
            r = sm.ratio()
            if r > best_ratio:
                best, best_ratio = i, r
        if best < 0:
            return None
        hits = [(best, j)]

    def region(hit: tuple[int, int]) -> tuple[int, int]:
        first = max(0, hit[0] - hit[1])
        last = min(len(lines) - 1, first + k - 1)
        return first, last

    if len(hits) > 1:
        want_key = _whitespace_key(old_str)
        hits.sort(
            key=lambda h: -difflib.SequenceMatcher(
                None, _whitespace_key("\n".join(lines[region(h)[0] : region(h)[1] + 1])), want_key, autojunk=False
            ).ratio()
        )
    first, last = region(hits[0])
    return first + 1, last + 1


def _not_found_detail(text: str, old_str: str, *, had_line_numbers: bool) -> str:
    """Why ``old_str`` missed, with the file's actual text at the likely spot.

    Three cases, most specific first: a whitespace-only mismatch (the common
    one: tabs/indentation copied wrong), a region that resembles ``old_str``,
    or nothing similar at all. In every case the model gets what it needs to
    retry without a ``view`` round trip.
    """
    parts = []
    if had_line_numbers:
        parts.append(
            "old_str contained `view` line-number prefixes (\"NNN<TAB>\"); those are not part of the file "
            "and were stripped before matching, but the text still did not match."
        )
    want = _whitespace_key(old_str)
    k = old_str.count("\n") + 1
    if want:
        lines = text.split("\n")
        for i in range(0, max(0, len(lines) - k + 1)):
            if _whitespace_key("\n".join(lines[i : i + k])) == want:
                parts.append(
                    f"Lines {i + 1}-{i + k} differ from old_str only in whitespace/indentation. They read:\n"
                    f"{_window(text, i + 1, i + k)}\n"
                    "Copy old_str exactly from this (without the line-number prefixes), including leading spaces and tabs."
                )
                return " ".join(parts)
    region = _closest_region(text, old_str)
    if region is not None:
        first, last = region
        parts.append(
            f"The closest text in the file is at lines {first}-{last}:\n{_window(text, first, last)}\n"
            "Copy old_str exactly from this (without the line-number prefixes) if it is the region you meant."
        )
    else:
        parts.append("Nothing resembling old_str is in the file; use `view` to check its current content.")
    return " ".join(parts)


def _ambiguous_detail(text: str, old_str: str, at: list[int]) -> str:
    """Each of the first occurrences with its surrounding lines, so ``old_str`` can be extended in one go."""
    k = old_str.count("\n") + 1
    blocks = [
        f"occurrence at line {ln}:\n{_window(text, ln, ln + k - 1, context=AMBIGUOUS_CONTEXT)}"
        for ln in at[:MAX_SHOWN_MATCHES]
    ]
    return "\n".join(blocks)


def str_replace_text(text: str, old_str: str, new_str: str) -> tuple[str, str, str]:
    """Return ``(new_text, snippet, note)`` or raise :class:`EditError`.

    Refuses zero matches and multiple matches, in both cases quoting the file's
    actual text at the relevant place so the retry needs no ``view``. ``note`` is
    non-empty only when ``view`` line-number prefixes had to be stripped from
    the arguments to make the edit match.
    """
    if not old_str:
        raise EditError("old_str is empty; give the exact existing text to replace.")
    note = ""
    count = text.count(old_str)
    if count == 0:
        bare = strip_line_numbers(old_str)
        if bare is not None and bare.strip():
            if text.count(bare) == 0:
                raise EditError(
                    "old_str was not found in the file, so nothing was changed. "
                    + _not_found_detail(text, bare, had_line_numbers=True)
                )
            old_str, count = bare, text.count(bare)
            bare_new = strip_line_numbers(new_str)
            if bare_new is not None:
                new_str = bare_new
            note = (
                "Note: `view` line-number prefixes were stripped from old_str"
                + ("/new_str" if bare_new is not None else "")
                + " before applying; they are not part of the file. Do not include them."
            )
        else:
            raise EditError(
                "old_str was not found in the file, so nothing was changed. "
                + _not_found_detail(text, old_str, had_line_numbers=False)
            )
    if count > 1:
        at = _match_lines(text, old_str)
        shown = ", ".join(map(str, at[:MAX_LISTED_MATCHES]))
        if len(at) > MAX_LISTED_MATCHES:
            shown += f", ... and {len(at) - MAX_LISTED_MATCHES} more"
        raise EditError(
            f"old_str occurs {count} times (at lines {shown}); "
            "nothing was changed. Include more surrounding lines so it matches exactly once. "
            f"The first {min(len(at), MAX_SHOWN_MATCHES)} occurrences with their context:\n"
            + _ambiguous_detail(text, old_str, at)
        )
    idx = text.index(old_str)
    new_text = text[:idx] + new_str + text[idx + len(old_str) :]
    first = text.count("\n", 0, idx) + 1
    last = first + new_str.count("\n")
    return new_text, _snippet(new_text, first, last), note


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


def _bound(text: str) -> str:
    """Last-resort cap so no editor observation exceeds the view budget.

    Everything the tool emits is already sized (``view`` by
    :func:`_numbered_within`, failure windows by ``MAX_ERROR_CONTEXT_LINES``),
    so this only triggers on pathological input such as a directory listing
    of very long paths; it cuts on a line boundary and says so.
    """
    if len(text) <= MAX_VIEW_CHARS:
        return text
    head = text[:MAX_VIEW_CHARS]
    nl = head.rfind("\n")
    if nl > 0:
        head = head[:nl]
    return head + f"\n... ({text[len(head):].count(chr(10))} more lines not shown)"


def run_edit_tool(env: Any, args: dict, *, cwd: str = DEFAULT_CWD) -> dict:
    """Execute one call; return a bash-shaped observation ``{"returncode", "output", "edit", "bounded"}``.

    ``returncode`` is 0 on success and 1 on a refused/failed call, so the same
    observation template renders both. ``edit`` names the sub-command when the
    file was actually changed, for metrics; it is absent otherwise. ``bounded``
    tells the loop the output is already within this tool's budget, so it must
    not apply the (smaller, middle-cutting) bash observation cap on top.
    """
    obs = _run_edit_tool(env, args, cwd=cwd)
    obs["output"] = _bound(obs.get("output", "") or "")
    obs["bounded"] = True
    return obs


def _run_edit_tool(env: Any, args: dict, *, cwd: str) -> dict:
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
            new_text, snippet, note = str_replace_text(text, old_str, new_str)
            write_text(env, path, new_text)
            head = f"Edited {path}. The region now reads:\n{snippet}"
            return {"returncode": 0, "output": head + (f"\n{note}" if note else ""), "edit": cmd}

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
