"""verl AgentLoop integration (SKILL 1: actor-rollout decoupling).

verl's agentic RL path hands each :class:`AgentLoopBase` a ``server_manager``
(an ``LLMServerClient``) and drives generation **token-in / token-out** over
Ray, exactly like verl's own ``single_turn_agent_loop`` / ``tool_agent_loop``:

    prompt_ids = await self.apply_chat_template(messages, tools=[...])
    out = await self.server_manager.generate(request_id, prompt_ids, sampling_params)
    # out.token_ids are the *exact* tokens the (weight-synced) engine sampled

We follow that contract so the rollout engine stays under verl's synchronous
weight control (``policy_lag == 0``) — the SGLang server verl manages in HYBRID
mode is resharded from the actor before every step. We do **not** re-plumb
generation over an OpenAI HTTP URL: verl never hands the agent loop a server
address, and the tokens returned here are the ground truth used for training (no
re-tokenization, no boundary drift).

What we reuse rather than reinvent:

* **verl** — ``apply_chat_template``, ``server_manager.generate``, the ``hermes``
  ``ToolParser`` (parses ``<tool_call>{...}</tool_call>`` out of the sampled
  tokens), and ``AgentLoopOutput`` (its ``reward_score`` field drops our binary
  reward onto the last token as ``rm_scores``).
* **mini-swe-agent** — ``get_sb_environment`` (the per-instance SWE-bench docker
  container), the ``bash`` tool schema (``BASH_TOOL``), and the prompt/observation
  templates from ``configs/agent.yaml``. ``env.execute`` raises ``Submitted`` when
  the agent runs the submit marker, carrying the final patch.
* **ours** — ``compute_reward`` (SWE-bench harness) and ``TrajectoryMetrics``.

The blocking bits (docker container create / ``env.execute`` / the harness) are
run in the loop's default executor so verl's event loop keeps serving the other
concurrent rollouts.

The standalone reference loop (``scripts/train_standalone.py``) still drives
mini-swe-agent's own ``DefaultAgent`` via ``env_wrapper`` + ``build_rollout_model``
against a plain SGLang server; that path is unchanged and independent of this one.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import re
import time
from typing import Any
from uuid import uuid4

from agentic_grpo.config import AgentConfig
from agentic_grpo.metrics import TrajectoryMetrics
from agentic_grpo.reward import apply_to_metrics, compute_reward

logger = logging.getLogger("agentic_grpo.agent_loop")

# Fallback bash tool schema, identical to mini-swe-agent's BASH_TOOL, used if the
# import isn't available (e.g. unit tests without minisweagent installed).
_BASH_TOOL_FALLBACK = {
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

try:  # verl >= 0.5
    from verl.experimental.agent_loop.agent_loop import (  # type: ignore
        AgentLoopBase,
        AgentLoopMetrics,
        AgentLoopOutput,
        register,
    )

    _HAS_VERL = True
except Exception:  # pragma: no cover - verl not installed
    _HAS_VERL = False

    def register(_name):  # type: ignore
        def _decorator(cls):
            return cls

        return _decorator

    class AgentLoopBase:  # type: ignore
        """Stub so the file imports without verl present."""

        def __init__(self, *args, **kwargs):
            self.server_manager = kwargs.get("server_manager")
            self.tokenizer = kwargs.get("tokenizer")
            self.config = kwargs.get("config")

    class AgentLoopMetrics:  # type: ignore
        def __init__(self, generate_sequences=0.0, tool_calls=0.0, compute_score=0.0):
            self.generate_sequences = generate_sequences
            self.tool_calls = tool_calls
            self.compute_score = compute_score

    AgentLoopOutput = dict  # type: ignore


# ---------------------------------------------------------------------------
# Pure trajectory bookkeeping (unit-testable, no verl/tokenizer dependency)
# ---------------------------------------------------------------------------
class _Trajectory:
    """Accumulate the flat token sequence + response mask for one episode.

    Mirrors verl's ``tool_agent_loop`` layout: ``prompt_ids`` is the full running
    sequence (initial prompt + every generated turn + every tool observation),
    and ``response_mask`` covers only the post-prompt region — ``1`` for tokens
    the policy generated (trained on), ``0`` for tool-observation tokens.
    """

    def __init__(self, prompt_ids: list[int]):
        self._all: list[int] = list(prompt_ids)
        self._prompt_len = len(prompt_ids)
        self._mask: list[int] = []

    def current_ids(self) -> list[int]:
        """Full sequence so far — the prompt for the next ``generate`` call."""
        return self._all

    def add_generated(self, ids: list[int]) -> None:
        self._all.extend(ids)
        self._mask.extend([1] * len(ids))

    def add_tool(self, ids: list[int]) -> None:
        self._all.extend(ids)
        self._mask.extend([0] * len(ids))

    def response_len(self) -> int:
        return len(self._mask)

    def finalize(self, response_length: int, *, pad_token_id: int) -> tuple[list[int], list[int], list[int]]:
        """Return ``(prompt_ids, response_ids, response_mask)`` for AgentLoopOutput.

        ``response_ids``/``response_mask`` are right-clipped to ``response_length``.
        If the episode produced no response tokens at all (e.g. the container
        failed before the first generation), emit a single padding token so verl's
        ``_pad_token_ids`` never sees an empty list — an empty response is what
        crashed the previous HTTP-based path.
        """
        prompt = self._all[: self._prompt_len]
        response = self._all[self._prompt_len :]
        mask = self._mask
        if not response:
            return prompt, [pad_token_id], [1]
        return prompt, response[:response_length], mask[:response_length]


def _command_from_tool_call(tool_call: Any) -> tuple[str, str]:
    """Extract ``(name, command)`` from a verl ``FunctionCall``.

    Returns an empty command (never raises) so a malformed call becomes an error
    observation the agent can recover from rather than killing the rollout.
    """
    name = getattr(tool_call, "name", "") or ""
    raw_args = getattr(tool_call, "arguments", "") or "{}"
    try:
        args = json.loads(raw_args)
        command = args.get("command", "") if isinstance(args, dict) else ""
    except (json.JSONDecodeError, TypeError):
        command = ""
    return name, command


# --- recovering tool calls verl's hermes parser threw away -------------------
#
# ``HermesToolParser.extract_tool_calls`` logs ``Failed to decode tool call`` and
# *silently drops* any ``<tool_call>`` block whose JSON won't parse, returning an
# empty list — indistinguishable from "the model made no tool call at all". The
# first full-batch run died on exactly this: every drop ended the episode, so all
# rewards were 0 and GRPO advantages were identically zero. The observed causes
# are all recoverable, so we salvage first and only fall back to telling the model
# it erred.
_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_COMMAND_FIELD_RE = re.compile(r'"command"\s*:\s*"')
_COMMAND_END_RE = re.compile(r'"\s*(?=[,}])')
_NAME_FIELD_RE = re.compile(r'"name"\s*:\s*"([^"]*)"')
_JSON_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f"}


def _attempted_tool_call(text: str) -> bool:
    """True if the model tried to call a tool, even if the call is unusable.

    Only the opening marker is required: a response cut off mid-call has no
    ``</tool_call>``, and that is a format error to report, not a finished turn.
    """
    return _TOOL_CALL_OPEN in text


def _unescape_json_string(raw: str) -> str:
    """Apply the escapes the model got right, leaving unknown ones literal.

    A single left-to-right pass, so ``\\\\n`` becomes a literal backslash + ``n``
    rather than a newline, and a regex like ``\\d+`` inside a bash command
    survives untouched.
    """
    return re.sub(r"\\(.)", lambda m: _JSON_ESCAPES.get(m.group(1), m.group(0)), raw, flags=re.DOTALL)


def _command_from_obj(obj: Any) -> tuple[str, str] | None:
    """Pull ``(name, command)`` out of a decoded ``<tool_call>`` payload.

    Accepts the near-miss shapes Qwen3 emits alongside the canonical
    ``{"name": ..., "arguments": {"command": ...}}`` — above all the
    ``arguments``-less form, which is what makes hermes raise ``KeyError:
    'arguments'`` (the single most common failure in the first full-batch run).
    """
    if not isinstance(obj, dict):
        return None
    name = obj["name"] if isinstance(obj.get("name"), str) else "bash"
    args = obj.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            return (name, args)  # ``arguments`` held the bare command string
    if isinstance(args, dict) and isinstance(args.get("command"), str):
        return (name, args["command"])
    if isinstance(obj.get("command"), str):
        return (name, obj["command"])  # ``arguments`` key absent entirely
    return None


def _lenient_command(block: str) -> tuple[str, str] | None:
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
    name = name_m.group(1) if name_m else "bash"
    return (name, _unescape_json_string(block[start.end() : ends[-1]]))


def _salvage_tool_calls(text: str) -> list[tuple[str, str]]:
    """Recover ``(name, command)`` pairs from ``<tool_call>`` blocks hermes dropped.

    Only *closed* blocks are salvaged: a block with no ``</tool_call>`` was cut
    off mid-generation, and running a truncated shell command is worse than
    reporting the format error.
    """
    calls: list[tuple[str, str]] = []
    for block in _TOOL_CALL_BLOCK_RE.findall(text):
        block = block.strip()
        try:
            got = _command_from_obj(json.loads(block))
        except (json.JSONDecodeError, TypeError):
            got = None
        if got is None:
            got = _lenient_command(block)
        if got is not None and got[1]:
            calls.append(got)
    return calls


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    return int(raw) if raw.isdigit() else default


_CONTAINER_SEM: "asyncio.Semaphore | None" = None
_LIVE_CONTAINER_SEM: "asyncio.Semaphore | None" = None


def _container_semaphore() -> "asyncio.Semaphore":
    """Per-process cap on concurrent docker container *starts* (lazy, loop-bound).

    Sized by ``AGENTIC_MAX_CONCURRENT_CONTAINERS`` (default 8). Bounds the
    ``docker run`` burst rate only — it is released the moment the container is
    up, so it does **not** bound how many containers are alive. See
    :func:`_live_container_semaphore` for that.
    """
    global _CONTAINER_SEM
    if _CONTAINER_SEM is None:
        _CONTAINER_SEM = asyncio.Semaphore(max(1, _int_env("AGENTIC_MAX_CONCURRENT_CONTAINERS", 8)))
    return _CONTAINER_SEM


def _live_container_semaphore() -> "asyncio.Semaphore":
    """Per-process cap on containers alive at once (lazy, loop-bound).

    The start semaphore alone is not enough: it frees as soon as ``docker run``
    returns, so live containers accumulate with in-flight rollouts. The first
    full-batch run (256x8) reached **193 concurrent** containers and the rootless
    daemon began returning ``exit status 125``, degrading rollouts to reward-0
    samples that are indistinguishable from genuine task failures.

    Held for the container's whole lifetime (create -> cleanup), so the ceiling is
    ``rollout.agent.num_workers * AGENTIC_MAX_LIVE_CONTAINERS``. The default of 8
    over 8 workers gives ~64 live containers. Raising it trades docker stability
    for rollout throughput; this is the knob to tune if a step is too slow.
    """
    global _LIVE_CONTAINER_SEM
    if _LIVE_CONTAINER_SEM is None:
        _LIVE_CONTAINER_SEM = asyncio.Semaphore(max(1, _int_env("AGENTIC_MAX_LIVE_CONTAINERS", 8)))
    return _LIVE_CONTAINER_SEM


def _dump_enabled() -> bool:
    return bool(os.environ.get("AGENTIC_TRAJECTORY_DUMP_DIR", ""))


def _dump_trajectory(instance_id: str, exit_status: str, reward: float, actions: list[dict]) -> None:
    """Append one JSON line describing an episode, if dumping is enabled.

    Opt-in via ``AGENTIC_TRAJECTORY_DUMP_DIR``. A binary reward tells you *that* a
    rollout scored 0, never *why* — this records the actual commands so you can
    see whether the agent worked productively and ran out of turns, or never
    attempted the submit marker at all. Best-effort: a dump failure must never
    disturb a rollout.
    """
    dump_dir = os.environ.get("AGENTIC_TRAJECTORY_DUMP_DIR", "")
    if not dump_dir:
        return
    try:
        os.makedirs(dump_dir, exist_ok=True)
        record = {
            "instance_id": instance_id,
            "exit_status": exit_status,
            "reward": reward,
            "num_actions": len(actions),
            "actions": actions,
        }
        # One file per process; each line is a complete episode.
        path = os.path.join(dump_dir, f"trajectories-{os.getpid()}.jsonl")
        with open(path, "a") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:  # pragma: no cover - diagnostics must not break training
        logger.warning("trajectory dump failed for %s", instance_id, exc_info=True)


def _truncate(text: str, max_len: int, side: str = "middle") -> str:
    if max_len <= 0 or len(text) <= max_len:
        return text
    if side == "left":
        return "(truncated)..." + text[-max_len:]
    if side == "right":
        return text[:max_len] + "...(truncated)"
    half = max_len // 2
    return text[:half] + "...(truncated)..." + text[-half:]


# ---------------------------------------------------------------------------
# The agent loop
# ---------------------------------------------------------------------------
@register("swebench_agent")
class SWEBenchAgentLoop(AgentLoopBase):
    """Native token-level multi-turn SWE-bench coding loop for verl."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mt = self.rollout_config.multi_turn
        self.response_length = self.rollout_config.response_length
        self.max_assistant_turns = mt.max_assistant_turns or AgentConfig().step_limit
        self.max_tool_response_length = getattr(mt, "max_tool_response_length", 8192) or 8192
        self.tool_response_truncate_side = getattr(mt, "tool_response_truncate_side", "middle") or "middle"

        # verl's hermes parser turns the sampled <tool_call>{...}</tool_call>
        # tokens into FunctionCall objects — the same format the model generates.
        from verl.experimental.agent_loop.tool_parser import ToolParser  # type: ignore

        self.tool_parser = ToolParser.get_tool_parser(mt.format, self.tokenizer)

        self._agent_config_path = AgentConfig().agent_config_path
        (
            self._system_tmpl,
            self._instance_tmpl,
            self._obs_tmpl,
            self._error_tmpl,
            self.max_consecutive_format_errors,
        ) = _load_agent_templates(self._agent_config_path)
        self._bash_schema = _bash_tool_schema()
        self._pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> "AgentLoopOutput":  # type: ignore[override]
        instance = kwargs.get("extra_info") or kwargs.get("instance") or {}
        instance_id = instance.get("instance_id", "unknown")

        metrics = TrajectoryMetrics(instance_id=instance_id)
        gen_s = tool_s = 0.0
        submission = ""
        exit_status = "IncompleteRollout"
        assistant_turns = 0

        # Build the prompt FIRST — this is tokenizer-only (no docker), so
        # ``prompt_ids`` is always valid even if the container never starts. An
        # empty ``prompt_ids`` is exactly what crashes verl's ``_pad_token_ids``
        # (``'list' object has no attribute 'dim'``), so a failed rollout must
        # still carry a real prompt and degrade to a reward-0 sample rather than
        # taking down the whole step via ``asyncio.gather``.
        messages = self._initial_messages(instance)
        prompt_ids = await self.apply_chat_template(messages, tools=[self._bash_schema])
        traj = _Trajectory(prompt_ids)
        sp = self._with_stop_tokens(sampling_params)

        t_traj = time.perf_counter()
        env = None
        consecutive_format_errors = 0
        actions: list[dict] = []  # only populated when dumping is enabled
        try:
            # The live-container semaphore is held across the whole episode, not
            # just the start, so the daemon never accumulates more than
            # num_workers * cap containers.
            async with _live_container_semaphore():
                try:
                    env = await self._make_env_bounded(instance)

                    while assistant_turns < self.max_assistant_turns and traj.response_len() < self.response_length:
                        t0 = time.perf_counter()
                        out = await self.server_manager.generate(
                            request_id=uuid4().hex,
                            prompt_ids=traj.current_ids(),
                            sampling_params=sp,
                        )
                        gen_s += time.perf_counter() - t0
                        traj.add_generated(out.token_ids)
                        assistant_turns += 1

                        if traj.response_len() >= self.response_length:
                            exit_status = "ContextLimit"
                            metrics.truncated = True
                            break

                        _, parsed = await self.tool_parser.extract_tool_calls(out.token_ids, None)
                        calls = [_command_from_tool_call(tc) for tc in parsed]

                        if not calls:
                            raw = await self._decode(out.token_ids)
                            attempted = _attempted_tool_call(raw)
                            calls = _salvage_tool_calls(raw) if attempted else []
                            if calls:
                                metrics.salvaged_tool_calls += len(calls)
                            else:
                                # No usable action this turn. Both causes are
                                # recoverable and must NOT end the episode:
                                #   * no tool call at all — overwhelmingly the agent
                                #     stopping right after `git diff > patch.txt` +
                                #     `cat patch.txt`, i.e. two of the three submit
                                #     steps done, patch written, never submitted.
                                #     Terminating here threw that patch away.
                                #   * a <tool_call> block salvage could not repair.
                                # Nudge with the format-error template (which restates
                                # the submit command) and let the agent finish.
                                consecutive_format_errors += 1
                                metrics.format_errors += 1
                                if _dump_enabled():
                                    actions.append(
                                        {
                                            "turn": assistant_turns,
                                            "tool": "<format_error>" if attempted else "<no_tool_call>",
                                            "command": None,
                                            "raw_response": _truncate(raw, 600),
                                        }
                                    )
                                if consecutive_format_errors >= self.max_consecutive_format_errors:
                                    exit_status = "FormatErrorLimit" if attempted else "NoToolCall"
                                    break
                                err_ids = await self.apply_chat_template(
                                    [{"role": "tool", "content": self._format_error(raw, attempted=attempted)}],
                                    remove_system_prompt=True,
                                )
                                metrics.tool_obs_tokens.append(len(err_ids))
                                traj.add_tool(err_ids)
                                if traj.response_len() >= self.response_length:
                                    exit_status = "ContextLimit"
                                    metrics.truncated = True
                                    break
                                continue
                        consecutive_format_errors = 0

                        obs_messages: list[dict] = []
                        submitted = False
                        for name, command in calls:
                            metrics.tool_call_count += 1
                            t1 = time.perf_counter()
                            obs, sub = await self.loop.run_in_executor(None, self._run_bash, env, name, command)
                            tool_s += time.perf_counter() - t1
                            if _dump_enabled():
                                actions.append(
                                    {
                                        "turn": assistant_turns,
                                        "tool": name,
                                        "command": _truncate(command, 600),
                                        "returncode": obs.get("returncode"),
                                        "output": _truncate(obs.get("output", "") or "", 400),
                                        "submitted": sub is not None,
                                    }
                                )
                            if sub is not None:
                                submission, submitted, exit_status = sub, True, "Submitted"
                                break
                            obs_messages.append({"role": "tool", "content": self._format_observation(obs)})

                        if submitted:
                            break

                        tool_ids = await self.apply_chat_template(obs_messages, remove_system_prompt=True)
                        metrics.tool_obs_tokens.append(len(tool_ids))
                        traj.add_tool(tool_ids)
                        if traj.response_len() >= self.response_length:
                            exit_status = "ContextLimit"
                            metrics.truncated = True
                            break
                    else:
                        # Loop fell through its condition rather than breaking: the
                        # agent used every turn without ever submitting a patch.
                        if assistant_turns >= self.max_assistant_turns:
                            exit_status = "TurnLimit"
                finally:
                    # Free the container (and the slot) before scoring, which runs
                    # its own harness container.
                    if env is not None:
                        await self.loop.run_in_executor(None, _safe_cleanup, env)
                        env = None
        except Exception as exc:  # pragma: no cover - defensive; keep the batch alive
            logger.exception("SWEBench agent loop crashed for %s", instance_id)
            exit_status = f"Crashed:{type(exc).__name__}"
        metrics.total_trajectory_time = time.perf_counter() - t_traj
        metrics.total_tool_call_time = tool_s

        # Binary SWE-bench reward from the official harness (blocking docker run).
        t_score = time.perf_counter()
        rr = await self.loop.run_in_executor(None, compute_reward, instance, submission)
        score_s = time.perf_counter() - t_score

        prompt_ids, response_ids, response_mask = traj.finalize(
            self.response_length, pad_token_id=self._pad_token_id
        )
        metrics.num_turns = assistant_turns
        metrics.exit_status = exit_status
        metrics.prompt_tokens = len(prompt_ids)
        metrics.completion_tokens = sum(response_mask)
        metrics.response_tokens = sum(response_mask)
        metrics.total_trajectory_tokens = len(prompt_ids) + len(response_ids)
        apply_to_metrics(metrics, rr)
        _dump_trajectory(instance_id, exit_status, rr.reward, actions)

        return self._build_output(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            reward=rr.reward,
            num_turns=assistant_turns * 2 + 1,
            gen_s=gen_s,
            tool_s=tool_s,
            score_s=score_s,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # helpers (the blocking ones run inside loop.run_in_executor)
    # ------------------------------------------------------------------
    async def _make_env_bounded(self, instance: dict) -> Any:
        """Create the docker env under a concurrency cap, with retry on failure.

        Rootless docker (``dockerd-rootless`` + ``slirp4netns``) can't absorb
        hundreds of simultaneous ``docker run`` calls — the daemon returns exit
        125 or hangs under a stampede. A per-process semaphore keeps only a few
        container starts in flight, and we retry transient failures with backoff.
        The retry sleep happens *outside* the semaphore so we don't hold a slot
        while waiting. Raises the last error if every attempt fails (the caller
        turns that into a reward-0 sample, not a crash).
        """
        sem = _container_semaphore()
        attempts = _int_env("AGENTIC_CONTAINER_START_RETRIES", 3)
        last_exc: Exception | None = None
        for attempt in range(max(1, attempts)):
            async with sem:
                try:
                    return await self.loop.run_in_executor(None, self._make_env, instance)
                except Exception as exc:  # noqa: BLE001 - retry any start failure
                    last_exc = exc
                    logger.warning(
                        "container start failed for %s (attempt %d/%d): %s",
                        instance.get("instance_id", "?"), attempt + 1, attempts, exc,
                    )
            await asyncio.sleep(1.5 * (attempt + 1))
        assert last_exc is not None
        raise last_exc

    def _make_env(self, instance: dict) -> Any:
        """Build the per-instance SWE-bench docker environment (mini-swe-agent)."""
        import yaml

        from minisweagent.run.benchmarks.swebench import get_sb_environment  # type: ignore

        config: dict = {}
        if os.path.isfile(self._agent_config_path):
            config = yaml.safe_load(open(self._agent_config_path)) or {}
        env_cfg = config.setdefault("environment", {})
        env_cfg.setdefault("environment_class", "docker")
        return get_sb_environment(config, instance)

    def _run_bash(self, env: Any, name: str, command: str) -> tuple[dict, str | None]:
        """Run one bash tool call in the container.

        Returns ``(observation_dict, submission)``. ``submission`` is non-None only
        when the command triggered mini-swe-agent's submit marker (``env.execute``
        raises ``Submitted`` carrying the final patch).
        """
        from minisweagent.exceptions import Submitted  # type: ignore

        if name != "bash":
            return {"returncode": -1, "output": f"Unknown tool '{name}'. Only 'bash' is available."}, None
        if not command:
            return {"returncode": -1, "output": "Missing 'command' argument in bash tool call."}, None
        try:
            out = env.execute({"command": command})
            return out, None
        except Submitted as e:
            submission = e.messages[0].get("extra", {}).get("submission", "") if e.messages else ""
            return {"returncode": 0, "output": ""}, submission

    def _initial_messages(self, instance: dict) -> list[dict]:
        task = instance.get("problem_statement", "")
        tvars = {**instance, "task": task}
        return [
            {"role": "system", "content": _render(self._system_tmpl, **tvars)},
            {"role": "user", "content": _render(self._instance_tmpl, **tvars)},
        ]

    async def _decode(self, token_ids: list[int]) -> str:
        """Decode sampled tokens back to text (off-loop; the tokenizer is blocking).

        Needed because hermes discards the ``<tool_call>`` markers from the
        ``content`` it returns, so its output can't tell a malformed call from no
        call at all.
        """
        return await self.loop.run_in_executor(None, self.tokenizer.decode, token_ids)

    def _format_error(self, raw: str, *, attempted: bool = True) -> str:
        """Render the agent-config format-error message for a turn with no action.

        ``attempted`` distinguishes a broken ``<tool_call>`` block from a response
        with no tool call at all. The latter is usually the agent believing it has
        finished, so the nudge restates the exact submit command — most such
        episodes have already written ``patch.txt`` and only need the final step.

        An unclosed ``<tool_call>`` means generation was cut off, which the template
        handles differently, so ``finish_reason`` selects that branch.
        """
        if attempted:
            error = (
                "Could not parse a `bash` tool call from your response. The JSON inside "
                '<tool_call>...</tool_call> must be valid: it needs a "name" of "bash" and an '
                '"arguments" object holding "command", with every newline, quote and backslash '
                "properly escaped."
            )
        else:
            error = (
                "Your response contained no `bash` tool call, so nothing was executed. "
                "Every response must make at least one `bash` tool call.\n"
                "If you are still working, issue the next command. If you have finished editing "
                "and have already written and checked patch.txt, submit it now with EXACTLY "
                "this command:\n"
                "  echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt\n"
                "Your work is not recorded until you run that command."
            )
        tvars: dict[str, Any] = {"error": error}
        if _TOOL_CALL_OPEN in raw and "</tool_call>" not in raw:
            tvars["finish_reason"] = "length"
        return _render(self._error_tmpl, **tvars)

    def _format_observation(self, obs: dict) -> str:
        output = {
            "returncode": obs.get("returncode"),
            "output": _truncate(obs.get("output", "") or "", self.max_tool_response_length, self.tool_response_truncate_side),
            "exception_info": (obs.get("extra", {}) or {}).get("exception_info", "") or obs.get("exception_info", ""),
        }
        return _render(self._obs_tmpl, output=output)

    def _with_stop_tokens(self, sampling_params: dict[str, Any]) -> dict[str, Any]:
        stop = self.tool_parser.stop_token_ids
        if not stop:
            return sampling_params
        sp = dict(sampling_params)
        sp["stop_token_ids"] = list(set((sp.get("stop_token_ids") or []) + stop))
        return sp

    def _build_output(
        self,
        *,
        prompt_ids: list[int],
        response_ids: list[int],
        response_mask: list[int],
        reward: float,
        num_turns: int,
        gen_s: float,
        tool_s: float,
        score_s: float,
        metrics: TrajectoryMetrics,
    ) -> Any:
        # verl's _pad_token_ids chokes on an empty prompt or response; never emit
        # one (a fully-degenerate rollout still returns a single pad token).
        if not prompt_ids:
            prompt_ids = [self._pad_token_id]
        if not response_ids:
            response_ids, response_mask = [self._pad_token_id], [1]

        extra_fields = {
            "trajectory_metrics": metrics.to_dict(),
            "exit_status": metrics.exit_status,
        }
        loop_metrics = AgentLoopMetrics(generate_sequences=gen_s, tool_calls=tool_s, compute_score=score_s)
        if not _HAS_VERL:  # pragma: no cover - test/standalone stub
            return {
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "response_mask": response_mask,
                "reward_score": reward,
                "num_turns": num_turns,
                "extra_fields": extra_fields,
            }
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            reward_score=reward,
            num_turns=num_turns,
            metrics=loop_metrics,
            extra_fields=extra_fields,
        )


# ---------------------------------------------------------------------------
# template loading (mini-swe-agent's own prompts, single source of truth)
# ---------------------------------------------------------------------------
_DEFAULT_ERROR_TMPL = (
    "Tool call error:\n\n<error>\n{{error}}\n</error>\n\n"
    "Every response must call the 'bash' tool with a single JSON argument, "
    'e.g. {"command": "ls -la"}.'
)


def _load_agent_templates(path: str) -> tuple[str, str, str, str, int]:
    """Return the templates + format-error budget from the agent yaml.

    ``(system, instance, observation, format_error, max_consecutive_format_errors)``.
    The last two were previously read only by :mod:`agentic_grpo.rollout_backend`
    (the standalone litellm loop), leaving this loop with no way to tell the model
    it had emitted an unusable tool call.
    """
    import yaml

    cfg: dict = {}
    if os.path.isfile(path):
        cfg = yaml.safe_load(open(path)) or {}
    agent = cfg.get("agent", {}) or {}
    model = cfg.get("model", {}) or {}
    raw_limit = agent.get("max_consecutive_format_errors", 3)
    try:
        limit = max(1, int(raw_limit))
    except (TypeError, ValueError):
        limit = 3
    return (
        agent.get("system_template", "You are a helpful assistant that can interact with a computer shell."),
        agent.get("instance_template", "{{task}}"),
        model.get("observation_template", "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}\n</output>"),
        model.get("format_error_template", _DEFAULT_ERROR_TMPL),
        limit,
    )


def _render(template: str, **vars: Any) -> str:
    from jinja2 import Template  # lightweight; already a mini-swe-agent dependency

    return Template(template).render(**vars)


def _bash_tool_schema() -> dict:
    try:
        from minisweagent.models.utils.actions_toolcall import BASH_TOOL  # type: ignore

        return BASH_TOOL
    except Exception:
        return _BASH_TOOL_FALLBACK


def _safe_cleanup(env: Any) -> None:
    try:
        env.cleanup()
    except Exception:  # pragma: no cover
        pass


_METRICS_PATCHED = False


def _patch_verl_data_metrics() -> None:
    """Merge batch-level TrajectoryMetrics into verl's W&B metrics dict."""
    global _METRICS_PATCHED
    if _METRICS_PATCHED or not _HAS_VERL:
        return

    from agentic_grpo.metrics import TrajectoryMetrics
    from agentic_grpo.server_monitor import get_shared_monitor
    import verl.trainer.ppo.metric_utils as metric_utils
    import verl.trainer.ppo.ray_trainer as ray_trainer

    original = metric_utils.compute_data_metrics

    # Server running-batch capacity for the drain-phase start; set to match the
    # SGLang `--max-running-requests` launch flag. Absent -> observed peak.
    cap_env = os.environ.get("AGENTIC_MAX_RUNNING_REQUESTS")
    max_concurrency = int(cap_env) if cap_env and cap_env.isdigit() else None

    def compute_data_metrics(batch, use_critic: bool = True):
        out = original(batch, use_critic=use_critic)
        raw = batch.non_tensor_batch.get("trajectory_metrics")
        if raw is not None:
            objs = [TrajectoryMetrics(**item) if isinstance(item, dict) else item for item in raw]
            out.update(TrajectoryMetrics.aggregate(objs))
        # Server-side latency + drain breakdown from SGLang's own /metrics for
        # this step's rollout window. Started by the generate_sequences hook below,
        # so by now it has been sampling for the whole rollout.
        monitor = get_shared_monitor()
        if monitor is not None:
            out.update(monitor.summarize_since_last(max_concurrency))
        return out

    metric_utils.compute_data_metrics = compute_data_metrics
    ray_trainer.compute_data_metrics = compute_data_metrics

    # --- Start the drain monitor BEFORE the rollout it is supposed to measure ---
    # Creating it lazily in compute_data_metrics is too late: that runs at the END
    # of a step, so the monitor began polling an already-idle server and _drain()
    # saw no sample with running > 0 and returned {}. Result: srv/* silently
    # missing for step 1 (and for a 1-step run, missing entirely).
    #
    # AgentLoopManager is a plain object, not a Ray actor, so its
    # generate_sequences runs in THIS (trainer) process -- the same process whose
    # module-level singleton compute_data_metrics later reads. Wrapping it starts
    # the poller just before rollout begins. Idempotent: get_shared_monitor()
    # returns the existing instance once created.
    try:
        from verl.experimental.agent_loop.agent_loop import AgentLoopManager

        _orig_gen = AgentLoopManager.generate_sequences

        # MUST stay a SYNC def returning _orig_gen(...) untouched. verl decorates
        # generate_sequences with @auto_await, so the attribute is a sync wrapper
        # that returns EITHER a coroutine (when the caller awaits) or the finished
        # result (when called directly). An `async def` wrapper here would hand the
        # trainer a coroutine it never awaits and rollout would silently produce
        # nothing. Passing the return value straight through preserves both paths.
        @functools.wraps(_orig_gen)
        def generate_sequences(self, *args, **kwargs):
            try:
                get_shared_monitor()
            except Exception:  # noqa: BLE001 - diagnostics must never break rollout
                logger.warning("server_monitor: pre-rollout start failed", exc_info=True)
            return _orig_gen(self, *args, **kwargs)

        AgentLoopManager.generate_sequences = generate_sequences
    except Exception:  # noqa: BLE001 - older/newer verl may move this class
        logger.warning(
            "server_monitor: could not hook AgentLoopManager.generate_sequences; "
            "srv/* drain metrics will start one step late.",
            exc_info=True,
        )

    _METRICS_PATCHED = True


_patch_verl_data_metrics()
