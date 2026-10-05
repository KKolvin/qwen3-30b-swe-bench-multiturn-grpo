"""Adapter for the conversation dump: ``messages-<pid>.jsonl``, text and nothing else.

This is the bottom rung of the degradation ladder in `SIMULATOR.md` §15: a trace
that holds only what was said. No token counts, no durations, no timestamps.
Everything the IR needs is either **derived** from the text by a rule that can be
written down, or **estimated** from a prior the caller supplied:

- Token counts are derived. The dump's header carries the chat template and the
  tool schemas the harness rendered with, so the first prompt is the template
  applied to the system and user messages, each assistant turn is its text
  re-encoded, and each observation is the template applied to the tool messages
  with the template's own system prefix stripped, which is what the harness did
  (``agent_loop.apply_chat_template(..., remove_system_prompt=True)``). Every
  rendered prompt is then left-truncated to ``prompt_length`` the way verl does.
  Measured on run 20261005-113503 (2,208 episodes): completions 97.9% exact,
  observations 100%; the rest is the model sampling a non-canonical token split,
  which re-encoding cannot reproduce and which drifts a few tokens per episode.
- Structure is derived. Turn k+1's prompt is turn k's whole sequence plus the
  observation, so ``shared_prefix_tokens`` follows; the eight rollouts of one
  problem share the first prompt, so they share a :class:`~simulator.ir.Prefix`.
  A turn's tool calls are counted from what followed it: one tool message per
  call that ran, or a nudge when none did (context, not work). An observation is
  recognisable by the harness's template, so anything else is a nudge.
  Only the last turn, which nothing follows, is read from its ``<tool_call>``
  blocks and the exit status. The blocks alone would undercount: on run
  20261005-113503 the model wrote 372 calls with raw tabs, bad escapes or no
  closing tag that strict JSON rejects and the harness's salvage ran anyway.
  The harness runs a turn's calls one after another, so the tasks chain.
- Durations are estimated. Container start, each tool call, patch recovery (when
  the episode did not submit), cleanup and grading each hang an
  :class:`~simulator.ir.Unknown` under the knob :data:`~simulator.adapters.base.KNOB_BY_KIND`
  names, and the caller's :class:`~simulator.adapters.base.Priors` must supply
  all of them or the conversion is refused. Nothing here is a number.

What text cannot say, and the adapter says so in its notes: which training step
an episode belonged to (no timestamps, so a run's steps come out as one
workload), the generation cap (``max_new_tokens`` stays unset for the config
bundle), why generation stopped, and whether a tool call timed out.

The observed half is empty by construction. That is the point of this adapter,
not a limitation: at use time there is no timeline, and the pipeline has to run
on this alone (`SIMULATOR.md` §13, rule 6).
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Protocol

from simulator.adapters.base import KNOB_BY_KIND, AdapterError, Conversion, Priors
from simulator.ir import Lease, Prefix, Request, Session, Task, Workload
from simulator.observed import Observations

NAME = "agentic-messages"
_FILE = "messages-*.jsonl"
_TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
#: How the harness's observation template starts (configs/agent.yaml,
#: ``observation_template``): a returncode, or an exception block before it. A
#: tool message that starts with neither is one of the nudges -- the format-error
#: one ("Tool call error:") or the one for a response cut off at the token limit.
_OBSERVATION_STARTS = ("<returncode>", "<exception>")
_NUDGE_STARTS = ("Tool call error", "Your previous response reached the output token limit")

Record = dict[str, Any]


class Tokenizer(Protocol):
    """The two calls this adapter makes. A HF tokenizer satisfies it; so does a fake."""

    chat_template: str | None

    def apply_chat_template(self, messages: list[dict], *, tools: list[dict] | None = None,
                            add_generation_prompt: bool = True, tokenize: bool = True) -> Any: ...

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


class AgenticMessagesAdapter:
    """See the module docstring.

    ``tokenizer`` is loaded from the header's ``model`` path when not given, with
    the header's template installed if the tokenizer's differs. ``prompt_length``
    overrides the header's; a header predating the field needs it passed.
    ``group_size`` splits an instance's rollouts into consecutive groups of that
    size in file order, which is right for this harness: a worker finishes every
    rollout of one step before the next step starts, and one problem's rollouts
    land in one worker. ``None`` leaves one group per instance per file.
    """

    name = NAME

    def __init__(self, tokenizer: Tokenizer | None = None, prompt_length: int | None = None,
                 group_size: int | None = None):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.group_size = group_size

    def claims(self, path: Path) -> bool:
        return bool(_files(path))

    def convert(self, path: Path, priors: Priors) -> Conversion:
        files = _files(path)
        meta, episodes = _read(files)
        tok = self._tokenizer(meta)
        cap = self.prompt_length if self.prompt_length is not None else meta.get("prompt_length")
        tools = meta.get("tool_schemas") or None
        render = _Renderer(tok, tools, cap)
        run = path.name if path.is_dir() else path.parent.name
        source = f"run:{run}/messages"
        tally = _Tally()
        if cap is None:
            tally.note("header has no prompt_length and none was passed: first prompts are not capped, "
                       "so a long problem statement comes out longer than the harness sent it")

        prefixes: list[Prefix] = []
        sessions: list[Session] = []
        for group, members in self._groups(episodes):
            firsts = [render.first_prompt(e["messages"]) for e in members
                      if _well_formed(e["messages"])]
            prefix = None
            if firsts:
                if len(set(firsts)) > 1:
                    tally.uneven_groups += 1
                prefix = Prefix(id=f"{group}/prompt", tokens=min(firsts), prov={"tokens": "derived"})
                prefixes.append(prefix)
            for e in members:
                if not _well_formed(e["messages"]):
                    tally.odd += 1
                    continue
                sessions.append(_session(e, group, prefix, render, priors, tally))

        meta_out = {"adapter": NAME, "run": run, "files": [f.name for f in files],
                    "model": meta.get("model"), "prompt_length": cap, "group_size": self.group_size}
        workload = Workload(sessions=tuple(sessions), source=source, prefixes=tuple(prefixes),
                            meta=meta_out)
        return Conversion(adapter=NAME, workload=workload, observations=Observations(source=source),
                          notes=tally.notes(len(sessions), self.group_size))

    def _tokenizer(self, meta: Record) -> Tokenizer:
        tok = self.tokenizer
        if tok is None:
            model = meta.get("model")
            if not model:
                raise AdapterError("the header names no model, and no tokenizer was given")
            try:
                from transformers import AutoTokenizer  # heavy; only when actually needed
            except ImportError as exc:  # pragma: no cover
                raise AdapterError("reading a conversation needs `transformers` for the tokenizer") from exc
            tok = AutoTokenizer.from_pretrained(model)
        template = meta.get("chat_template")
        if template and getattr(tok, "chat_template", None) != template:
            # The header's template is the one the prompts were rendered with.
            tok.chat_template = template
        return tok

    def _groups(self, episodes: list[tuple[str, Record]]) -> list[tuple[str, list[Record]]]:
        by_key: dict[tuple[str, str], list[Record]] = defaultdict(list)
        for fname, e in episodes:
            by_key[(fname, e.get("instance_id", "?"))].append(e)
        out: list[tuple[str, list[Record]]] = []
        for (fname, inst), members in by_key.items():
            # The worker is part of the name: one instance's rollouts can sit in
            # two workers' files (a validation batch of 160 splits 20 per worker,
            # which is not a multiple of 8), and a group id doubles as a prefix id.
            worker = fname.removeprefix("messages-").removesuffix(".jsonl")
            if self.group_size:
                chunks = [members[i:i + self.group_size] for i in range(0, len(members), self.group_size)]
            else:
                chunks = [members]
            for k, chunk in enumerate(chunks):
                suffix = f"/{k}" if len(chunks) > 1 else ""
                out.append((f"{inst}@{worker}{suffix}", chunk))
        return out


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def _files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.glob(_FILE))
    if path.is_file() and path.match(_FILE):
        return [path]
    return []


def _read(files: Iterable[Path]) -> tuple[Record, list[tuple[str, Record]]]:
    """The first header found, and every episode record tagged with its file name."""
    meta: Record | None = None
    episodes: list[tuple[str, Record]] = []
    for p in files:
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = r.get("kind")
                if kind == "meta":
                    meta = meta or r
                elif kind == "episode" and "messages" in r and "traj" in r:
                    episodes.append((p.name, r))
    if meta is None:
        raise AdapterError("no header record: the template and tool schemas the prompts were "
                           "rendered with are missing, and token counts cannot be derived without them")
    return meta, episodes


def _well_formed(msgs: list[dict]) -> bool:
    return (len(msgs) >= 3 and msgs[0].get("role") == "system" and msgs[1].get("role") == "user"
            and msgs[2].get("role") == "assistant")


# ---------------------------------------------------------------------------
# rendering: text -> token counts, exactly as the harness did
# ---------------------------------------------------------------------------
def _ids(out: Any) -> list[int]:
    """Transformers 4 returns a list, 5 may return a BatchEncoding."""
    if isinstance(out, dict) and "input_ids" in out:
        out = out["input_ids"]
    elif hasattr(out, "input_ids"):
        out = out.input_ids
    return list(out)


class _Renderer:
    def __init__(self, tok: Tokenizer, tools: list[dict] | None, cap: int | None):
        self.tok, self.tools, self.cap = tok, tools, cap
        # The template may insert a default system prompt; the harness strips it
        # from every observation render. Same derivation as verl's
        # initialize_system_prompt: render one empty user turn and two, and the
        # part that did not double is the prefix.
        one = _ids(tok.apply_chat_template([{"role": "user", "content": ""}],
                                           add_generation_prompt=False, tokenize=True))
        two = _ids(tok.apply_chat_template([{"role": "user", "content": ""}] * 2,
                                           add_generation_prompt=False, tokenize=True))
        self.sys_n = len(one) - (len(two) - len(one))

    def _capped(self, ids: list[int]) -> list[int]:
        return ids[-self.cap:] if self.cap and len(ids) > self.cap else ids

    def first_prompt(self, msgs: list[dict]) -> int:
        ids = _ids(self.tok.apply_chat_template(msgs[:2], tools=self.tools,
                                                add_generation_prompt=True, tokenize=True))
        return len(self._capped(ids))

    def observation(self, msgs: list[dict]) -> int:
        ids = _ids(self.tok.apply_chat_template(msgs, tools=None, add_generation_prompt=True,
                                                tokenize=True))[self.sys_n:]
        return len(self._capped(ids))

    def completion(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))


def _count_calls(text: str) -> int:
    """Parsable ``<tool_call>`` blocks. Only a fallback for a turn nothing follows."""
    n = 0
    for body in _TOOL_CALL.findall(text):
        try:
            call = json.loads(body.strip(), strict=False)  # the model's raw tabs are fine
        except json.JSONDecodeError:
            continue
        if isinstance(call, dict) and call.get("name"):
            n += 1
    return n


#: Exits in which the last turn's tool ran and ended the episode (submit, or the
#: repeat guard refusing a call), against exits in which no call ran at all.
_ENDED_BY_A_CALL = {"Submitted", "RepetitionLimit"}
_ENDED_WITHOUT_A_CALL = {"FormatErrorLimit", "NoToolCall", "ContextLimit"}


def _is_observation(msg: dict) -> bool:
    return (msg.get("content") or "").startswith(_OBSERVATION_STARTS)


def _calls_that_ran(text: str, obs: list[dict], exit_status: str | None) -> tuple[int, bool]:
    """How many tools a turn ran, and whether what followed was a nudge.

    With tool messages: one call per observation, or none if it was a nudge.
    Without any (the episode ended on this turn): the exit says whether a call
    ended it, and the blocks say how many there were, floored at one.
    """
    if obs:
        if not _is_observation(obs[0]):
            return 0, True
        return len(obs), False
    parsed = _count_calls(text)
    if exit_status in _ENDED_BY_A_CALL:
        return max(1, parsed), False
    if exit_status in _ENDED_WITHOUT_A_CALL:
        return 0, False
    return parsed, False


# ---------------------------------------------------------------------------
# one session
# ---------------------------------------------------------------------------
def _session(e: Record, group: str, prefix: Prefix | None, render: _Renderer, priors: Priors,
             tally: "_Tally") -> Session:
    sid = e["traj"]
    msgs = e["messages"]
    tasks: list[Task] = []
    requests: list[Request] = []

    def task(tid: str, kind: str, after: tuple[str, ...], pool: str | None = None) -> Task:
        u = priors.require(KNOB_BY_KIND[kind], missing=f"{kind} durations (a conversation has none)")
        t = Task(id=tid, session=sid, duration_s=0.0, pool=pool, kind=kind, after=after,
                 prov={"duration_s": "estimated"}, unknown={"duration_s": u})
        tasks.append(t)
        tally.estimated[u.knob] += 1
        return t

    start = task(f"{sid}/start", "container_start", (), pool="container")
    after: tuple[str, ...] = (start.id,)
    prev: Request | None = None
    prompt = render.first_prompt(msgs)
    if render.cap and prompt == render.cap:
        tally.capped += 1
    i, turn = 2, 0
    while i < len(msgs):
        if msgs[i].get("role") != "assistant":
            tally.odd_turns += 1
            break
        turn += 1
        text = msgs[i].get("content") or ""
        completion = render.completion(text)
        rid = f"{sid}/gen{turn}"
        if prev is None:
            parent = prefix.id if prefix is not None else None
            shared = prefix.tokens if prefix is not None else 0
        else:
            parent, shared = prev.id, prev.prompt_tokens + prev.completion_tokens
        req = Request(id=rid, session=sid, prompt_tokens=prompt, prefix_parent=parent,
                      shared_prefix_tokens=shared, completion_tokens=completion, after=after,
                      prov={"prompt_tokens": "derived", "completion_tokens": "derived",
                            "shared_prefix_tokens": "derived"})
        requests.append(req)
        tally.requests += 1
        after = (rid,)
        i += 1
        obs = []
        while i < len(msgs) and msgs[i].get("role") != "assistant":
            obs.append(msgs[i])
            i += 1
        calls, nudged = _calls_that_ran(text, obs, e.get("exit_status"))
        for c in range(calls):
            t = task(f"{sid}/tool{turn}.{c}", "tool", after)
            after = (t.id,)
        if nudged:
            tally.nudges += 1
            if not (obs[0].get("content") or "").startswith(_NUDGE_STARTS):
                tally.unknown_nudges += 1
        if obs:
            prompt = prompt + completion + render.observation(obs)
        prev = req

    if e.get("exit_status") not in (None, "Submitted"):
        t = task(f"{sid}/patch_recover", "patch_recover", after)
        after = (t.id,)
    cleanup = task(f"{sid}/cleanup", "cleanup", after)
    task(f"{sid}/eval", "eval", (cleanup.id,), pool="eval")
    return Session(id=sid, requests=tuple(requests), tasks=tuple(tasks), group=group,
                   leases=(Lease(pool="container", acquire_before=start.id,
                                 release_after=cleanup.id),))


class _Tally:
    def __init__(self) -> None:
        self.requests = 0
        self.capped = 0
        self.nudges = 0
        self.unknown_nudges = 0
        self.odd = 0
        self.odd_turns = 0
        self.uneven_groups = 0
        self.estimated: Counter[str] = Counter()
        self._extra: list[str] = []

    def note(self, s: str) -> None:
        self._extra.append(s)

    def notes(self, n_sessions: int, group_size: int | None) -> tuple[str, ...]:
        out = [
            f"{self.requests} requests over {n_sessions} sessions: every token count is derived from "
            f"the text with the header's chat template and tool schemas; expect a drift of a few "
            f"tokens per episode where the model sampled a non-canonical token split",
            f"max_new_tokens left unset for the config bundle; stop is 'unknown'; no tool call can be "
            f"marked as a timeout: a conversation records none of these",
            "no timestamps, so the run's training steps are not separable: all episodes in one "
            "workload" + (f", instances split into consecutive groups of {group_size} in file order"
                          if group_size else ", one group per instance per file"),
        ]
        if self.capped:
            out.append(f"{self.capped} first prompts hit the prompt_length cap and were left-truncated, "
                       f"as the harness did")
        if self.nudges:
            out.append(f"{self.nudges} turns followed by a nudge instead of an observation: context, no task")
        if self.unknown_nudges:
            out.append(f"{self.unknown_nudges} of those nudges start with neither known nudge text; check the "
                       f"harness templates, they were treated as nudges because they are not observations")
        if self.uneven_groups:
            out.append(f"{self.uneven_groups} groups whose first prompts differ in length; the prefix is the shortest")
        if self.odd or self.odd_turns:
            out.append(f"{self.odd} episodes and {self.odd_turns} turns skipped for not reading as "
                       f"system, user, then alternating assistant and tool messages")
        for knob, n in sorted(self.estimated.items()):
            out.append(f"{n} durations estimated from prior {knob!r}")
        return tuple(out + self._extra)
