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
  container), the ``bash`` tool schema (``BASH_TOOL``, via :mod:`agentic_grpo.bash_tool`),
  and the prompt/observation templates from ``configs/agent.yaml``. ``env.execute`` raises ``Submitted`` when
  the agent runs the submit marker, carrying the final patch.
* **ours** — ``compute_reward`` (SWE-bench harness), ``TrajectoryMetrics``, the
  tool registry (:mod:`agentic_grpo.tools`, one adapter each over
  :mod:`agentic_grpo.bash_tool` and :mod:`agentic_grpo.editor_tool`), and the
  tool-call parse/salvage layer (:mod:`agentic_grpo.tool_calls`).

The blocking bits (docker container create / ``env.execute`` / the harness) are
run in the loop's default executor so verl's event loop keeps serving the other
concurrent rollouts.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import re
import time
import yaml
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from agentic_grpo.config import AgentConfig, resolve_container_env
from agentic_grpo import tool_calls, tools
from agentic_grpo.episode import Episode, truncate
from agentic_grpo.metrics import TrajectoryMetrics
from agentic_grpo.reward import apply_to_metrics, compute_reward
from agentic_grpo.timeline import patch_trainer_timeline

logger = logging.getLogger("agentic_grpo.agent_loop")

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

    # Attribute-carrying stand-ins for verl's pydantic models, with the same
    # field names and defaults. They must be *objects*, not dicts: ``run()``
    # builds one output object on both paths, and the tests read it back as
    # ``out.extra_fields[...]``. A dict here would make the whole suite
    # verl-only, which is the thing this fallback exists to avoid.
    @dataclass
    class AgentLoopMetrics:  # type: ignore
        generate_sequences: float = 0.0
        tool_calls: float = 0.0
        compute_score: float = 0.0
        num_preempted: int = -1

    @dataclass
    class AgentLoopOutput:  # type: ignore
        prompt_ids: list = field(default_factory=list)
        response_ids: list = field(default_factory=list)
        response_mask: list = field(default_factory=list)
        metrics: "AgentLoopMetrics" = field(default_factory=AgentLoopMetrics)
        response_logprobs: Any = None
        routed_experts: Any = None
        multi_modal_data: Any = None
        reward_score: float | None = None
        num_turns: int = 0
        extra_fields: dict = field(default_factory=dict)
        mm_processor_kwargs: Any = None


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


_EVAL_SEM: "asyncio.Semaphore | None" = None
_EVAL_POOL: "ThreadPoolExecutor | None" = None


def _eval_semaphore() -> "asyncio.Semaphore":
    """Per-process cap on concurrent reward-eval containers (lazy, loop-bound).

    Grading runs its own docker container, and it runs *after* the rollout has
    released its live-container slot -- so nothing bounded it. That was harmless
    only while the harness cache made grading a 20 ms file read; with the cache
    keyed on the patch (see :mod:`agentic_grpo.reward`) every submitted patch now
    starts a real container, and an unbounded 2048-per-step fan-out would put the
    daemon straight back into ``exit status 125``.

    Size it so ``AGENTIC_MAX_LIVE_CONTAINERS + AGENTIC_MAX_EVAL_CONTAINERS``,
    times ``rollout.agent.num_workers``, stays under the daemon's ceiling.
    """
    global _EVAL_SEM
    if _EVAL_SEM is None:
        _EVAL_SEM = asyncio.Semaphore(max(1, _int_env("AGENTIC_MAX_EVAL_CONTAINERS", 4)))
    return _EVAL_SEM


def _eval_pool() -> ThreadPoolExecutor:
    """Threads reserved for the blocking harness call.

    ``run_in_executor(None, ...)`` would put a 900 s test run into the loop's
    default pool, the same one container creation and cleanup use; a batch of
    slow evals would then starve the rollouts still in flight.
    """
    global _EVAL_POOL
    if _EVAL_POOL is None:
        _EVAL_POOL = ThreadPoolExecutor(
            max_workers=max(1, _int_env("AGENTIC_MAX_EVAL_CONTAINERS", 4)),
            thread_name_prefix="reward-eval",
        )
    return _EVAL_POOL


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
        self._tool_schemas = tools.schemas()
        self._pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> "AgentLoopOutput":  # type: ignore[override]
        """One episode: rollout in a container, then grade. Never raises into verl."""
        instance = kwargs.get("extra_info") or kwargs.get("instance") or {}
        instance_id = instance.get("instance_id", "unknown")

        # Build the prompt FIRST — this is tokenizer-only (no docker), so
        # ``prompt_ids`` is always valid even if the container never starts. An
        # empty ``prompt_ids`` is exactly what crashes verl's ``_pad_token_ids``
        # (``'list' object has no attribute 'dim'``), so a failed rollout must
        # still carry a real prompt and degrade to a reward-0 sample rather than
        # taking down the whole step via ``asyncio.gather``.
        prompt_ids = await self.apply_chat_template(self._initial_messages(instance), tools=self._tool_schemas)
        ep = Episode(instance_id, prompt_ids)
        sp = self._with_stop_tokens(sampling_params)

        try:
            await self._rollout(ep, instance, sp)
        except Exception as exc:  # pragma: no cover - defensive; keep the batch alive
            logger.exception("SWEBench agent loop crashed for %s", instance_id)
            ep.crashed(exc)
        ep.close()

        rr, score_s = await self._grade(ep, instance)

        prompt_ids, response_ids, response_mask = ep.finalize(self.response_length, pad_token_id=self._pad_token_id)
        apply_to_metrics(ep.metrics, rr)
        ep.flush(rr.reward)
        return self._build_output(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            reward=rr.reward,
            num_turns=ep.assistant_turns * 2 + 1,
            gen_s=ep.gen_s,
            tool_s=ep.tool_s,
            score_s=score_s,
            metrics=ep.metrics,
        )

    # ------------------------------------------------------------------
    # phases
    # ------------------------------------------------------------------
    async def _rollout(self, ep: Episode, instance: dict, sp: dict[str, Any]) -> None:
        """Admission -> container -> turns until an exit -> patch fallback -> cleanup."""
        # The live-container semaphore is held across the whole episode, not
        # just the start, so the daemon never accumulates more than
        # num_workers * cap containers.
        t_slot = time.time()
        async with _live_container_semaphore():
            ep.admitted(t_slot)
            env = None
            try:
                with ep.span("container_start"):
                    env = await self._make_env_bounded(instance)
                while ep.can_continue(self.max_assistant_turns, self.response_length):
                    if not await self._step(ep, env, sp):
                        break
                else:
                    # Ran out of turns without any other exit: the agent used
                    # every turn and never submitted a patch.
                    if ep.assistant_turns >= self.max_assistant_turns:
                        ep.stop("TurnLimit")
            finally:
                # Free the container (and the slot) before scoring, which runs
                # its own harness container.
                if env is not None:
                    await self._release(ep, env, instance)

    async def _step(self, ep: Episode, env: Any, sp: dict[str, Any]) -> bool:
        """One assistant turn: generate, act, observe. False when the episode is over."""
        out = await self._generate(ep, sp)
        if ep.context_full(self.response_length):
            return ep.stop("ContextLimit", truncated=True)

        calls, raw, attempted = await self._actions(ep, out)
        if not calls:
            return await self._nudge(ep, raw, attempted)
        ep.consecutive_format_errors = 0

        obs_messages = await self._execute(ep, env, calls)
        if obs_messages is None:  # submitted
            return False
        ids = await self.apply_chat_template(obs_messages, remove_system_prompt=True)
        return self._observe(ep, ids)

    async def _generate(self, ep: Episode, sp: dict[str, Any]) -> Any:
        t0 = time.perf_counter()
        call_start = time.time()
        out = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=ep.traj.current_ids(),
            sampling_params=sp,
        )
        ep.generated(out, call_start, time.time(), time.perf_counter() - t0)
        return out

    async def _actions(self, ep: Episode, out: Any) -> tuple[list[tool_calls.ToolCall], str, bool]:
        """Parsed calls for this turn, falling back to salvage of what hermes dropped.

        Returns ``(calls, raw_text, attempted)``; ``raw_text`` is only decoded
        (and ``attempted`` only meaningful) when the strict path found nothing.
        """
        _, parsed = await self.tool_parser.extract_tool_calls(out.token_ids, None)
        calls = [tool_calls.from_function_call(tc) for tc in parsed]
        if calls:
            return calls, "", True
        raw = await self._decode(out.token_ids)
        attempted = tool_calls.attempted(raw)
        calls = tool_calls.salvage(raw) if attempted else []
        if calls:
            ep.salvaged(len(calls))
        return calls, raw, attempted

    async def _nudge(self, ep: Episode, raw: str, attempted: bool) -> bool:
        """No usable action this turn. Tell the model and continue, up to a budget.

        Both causes are recoverable and must NOT end the episode:
          * no tool call at all — overwhelmingly the agent stopping right after
            `git diff > patch.txt` + `cat patch.txt`, i.e. two of the three
            submit steps done, patch written, never submitted. Terminating here
            threw that patch away.
          * a <tool_call> block salvage could not repair.
        The format-error template restates the submit command.
        """
        ep.format_error(attempted, raw)
        if ep.consecutive_format_errors >= self.max_consecutive_format_errors:
            return ep.stop("FormatErrorLimit" if attempted else "NoToolCall")
        ids = await self.apply_chat_template(
            [{"role": "tool", "content": self._format_error(raw, attempted=attempted)}],
            remove_system_prompt=True,
        )
        return self._observe(ep, ids)

    async def _execute(self, ep: Episode, env: Any, calls: list[tool_calls.ToolCall]) -> list[dict] | None:
        """Run the turn's calls in order. None once one of them submitted."""
        ep.tool_phase(len(calls))
        obs_messages: list[dict] = []
        for name, args in calls:
            t1 = time.perf_counter()
            call_t = time.time()
            obs, sub = await self.loop.run_in_executor(None, tools.run, env, name, args)
            ep.tool_called(name, args, obs, sub, call_t, time.perf_counter() - t1)
            if sub is not None:
                ep.submission = sub
                ep.stop("Submitted")
                return None
            obs_messages.append({"role": "tool", "content": self._format_observation(obs)})
        return obs_messages

    def _observe(self, ep: Episode, ids: list[int]) -> bool:
        """Append a tool/nudge observation; False if that filled the context."""
        ep.add_observation(ids)
        if ep.context_full(self.response_length):
            return ep.stop("ContextLimit", truncated=True)
        return True

    async def _release(self, ep: Episode, env: Any, instance: dict) -> None:
        """Patch fallback, then cleanup. Runs in ``finally``: the container must go."""
        # Last chance to salvage a patch: every exit but "Submitted" leaves
        # `submission` empty even when the agent's edits are sitting in
        # /testbed. See _recover_patch. Must happen before cleanup -- the
        # container is gone right after.
        if not ep.submission and _patch_fallback_enabled():
            with ep.span("patch_recover"):
                ep.submission = await self.loop.run_in_executor(None, _recover_patch, env, instance)
            ep.metrics.patch_recovered = bool(ep.submission)
        with ep.span("cleanup"):
            await self.loop.run_in_executor(None, _safe_cleanup, env)

    async def _grade(self, ep: Episode, instance: dict) -> tuple[Any, float]:
        """Binary SWE-bench reward from the official harness (its own docker run)."""
        t_eval_slot = time.time()
        async with _eval_semaphore():
            t_score = time.perf_counter()
            ep.scoring(t_eval_slot)
            rr = await self.loop.run_in_executor(_eval_pool(), compute_reward, instance, ep.submission)
            ep.scored()
            return rr, time.perf_counter() - t_score

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
        from minisweagent.run.benchmarks.swebench import get_sb_environment  # type: ignore

        config: dict = {}
        if os.path.isfile(self._agent_config_path):
            config = yaml.safe_load(open(self._agent_config_path)) or {}
        env_cfg = config.setdefault("environment", {})
        env_cfg.setdefault("environment_class", "docker")
        # The yaml's `environment.env` carries ${VAR} references (the egress
        # proxy's host-specific IP); mini-swe-agent does no substitution.
        env_cfg["env"] = resolve_container_env(env_cfg.get("env", {}))
        return get_sb_environment(config, instance)

    def _initial_messages(self, instance: dict) -> list[dict]:
        task = instance.get("problem_statement", "")
        # ``edit_tool`` selects the two-tool or bash-only wording in agent.yaml, so
        # the AGENTIC_EDIT_TOOL=0 arm is not told about a tool it cannot call.
        tvars = {**instance, "task": task, "edit_tool": tools.edit_tool_enabled()}
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

        Tool names and argument shapes come from :mod:`agentic_grpo.tools`, so a
        bash-only run (``AGENTIC_EDIT_TOOL=0``) is never pointed at the editor.
        """
        if attempted:
            error = (
                "Could not parse a tool call from your response. The JSON inside "
                f"<tool_call>...</tool_call> must be valid: {tools.call_shapes()}, with every "
                "newline, quote and backslash properly escaped."
            )
        else:
            error = (
                "Your response contained no tool call, so nothing was executed. "
                f"Every response must make at least one {tools.names()} call.\n"
                "If you are still working, issue the next command. If you have finished editing "
                "and have already written and checked patch.txt, submit it now with EXACTLY "
                "this command:\n"
                "  echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt\n"
                "Your work is not recorded until you run that command."
            )
        tvars: dict[str, Any] = {"error": error, "edit_tool": tools.edit_tool_enabled()}
        if tool_calls.unclosed(raw):
            tvars["finish_reason"] = "length"
        return _render(self._error_tmpl, **tvars)

    def _format_observation(self, obs: dict) -> str:
        output = {
            "returncode": obs.get("returncode"),
            "output": truncate(obs.get("output", "") or "", self.max_tool_response_length, self.tool_response_truncate_side),
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
            # include_turns=False: the per-turn timeline is for the dump file, not
            # for the batch. It rides in non_tensor_batch all the way to the
            # trainer, and 80 turns x a dozen floats x 2048 trajectories of
            # Ray-serialised object arrays buys nothing that summarize_turns()
            # has not already reduced to a scalar.
            "trajectory_metrics": metrics.to_dict(include_turns=False),
            "exit_status": metrics.exit_status,
        }
        loop_metrics = AgentLoopMetrics(generate_sequences=gen_s, tool_calls=tool_s, compute_score=score_s)
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
    "Every response must call a tool: 'bash' with a single JSON argument, "
    'e.g. {"command": "ls -la"}{% if edit_tool %}, or the file editor{% endif %}.'
)


def _load_agent_templates(path: str) -> tuple[str, str, str, str, int]:
    """Return the templates + format-error budget from the agent yaml.

    ``(system, instance, observation, format_error, max_consecutive_format_errors)``.
    The last two were once read only by the retired standalone litellm loop,
    leaving this loop with no way to tell the model it had emitted an unusable
    tool call.
    """
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


_BASE_COMMIT_RE = re.compile(r"\A[0-9a-fA-F]{7,40}\Z")


def _patch_fallback_enabled() -> bool:
    return os.environ.get("AGENTIC_PATCH_FALLBACK", "1") != "0"


def _recover_patch(env: Any, instance: dict) -> str:
    """Read the working tree's diff out of the container, as a last-resort submission.

    ``submission`` normally arrives only through mini-swe-agent's submit marker
    -- the agent echoing ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` as the first
    line of a bash call that exits 0. Every other ending (TurnLimit,
    ContextLimit, NoToolCall, FormatErrorLimit) leaves it empty and the episode
    grades as ``empty_patch`` no matter what the agent actually wrote into
    /testbed. On run 20260828-052125 that was 2113 of 5282 episodes, and 951 of
    them had already written patch.txt: real work, thrown away because the model
    forgot one echo. Their resolve rate is 0.0% by construction while submitted
    episodes score 16.5%, so the loss is also a large block of all-zero GRPO
    groups that carry no advantage.

    So ask git what changed before the container goes away. ``diff
    <base_commit>`` rather than a bare ``diff``, so the patch survives an agent
    that staged or committed its work; ``core.fileMode=false`` mirrors the
    harness's own eval script. Untracked files (repro scripts, patch.txt itself)
    are excluded by construction -- which is what the submission instructions in
    ``configs/agent.yaml`` ask for anyway.

    Best effort throughout: anything unexpected returns "" and the episode keeps
    the reward 0 it already had.
    """
    base = str(instance.get("base_commit") or "")
    ref = base if _BASE_COMMIT_RE.match(base) else ""
    try:
        out = env.execute({"command": f"git -c core.fileMode=false diff {ref}".rstrip()})
    except Exception as exc:  # noqa: BLE001 - the container may already be gone
        logger.warning("patch recovery failed for %s: %s", instance.get("instance_id", "?"), exc)
        return ""
    if out.get("returncode") != 0:
        return ""
    patch = out.get("output") or ""
    if not patch.strip():
        return ""
    # A cut diff cannot apply, so an oversized one is dropped whole rather than
    # truncated: both grade 0, but only the truncated one would masquerade as a
    # real submission in patch_applied_rate.
    if len(patch) > _int_env("AGENTIC_PATCH_FALLBACK_MAX_BYTES", 1_000_000):
        logger.warning(
            "patch recovery discarded %d bytes for %s (over cap)",
            len(patch), instance.get("instance_id", "?"),
        )
        return ""
    return patch


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

    from agentic_grpo.metrics import TrajectoryMetrics, guard_infra_failures
    from agentic_grpo.server_monitor import get_shared_monitor
    from agentic_grpo.sglang_timing import patch_verl_rollout_timing
    import verl.trainer.ppo.metric_utils as metric_utils
    import verl.trainer.ppo.ray_trainer as ray_trainer

    # Per-request SGLang timestamps on every generate response. Must happen here,
    # in the trainer process: SGLangReplica.__init__ (which picks the server actor
    # class) is called from RayPPOTrainer.init_workers, i.e. this process. Doing it
    # in the AgentLoopWorker would be too late and in the wrong process.
    patch_verl_rollout_timing()

    # Absolute timestamps for every training phase (gen, reward, update_actor,
    # ...). Same reasoning as above: fit() runs in THIS process, so the
    # marked_timer rebinding has to happen here.
    patch_trainer_timeline()

    # Hermes logs ``Failed to decode tool call`` at ERROR for every block it
    # drops -- 4769 lines in a 3-step run -- but we salvage 84% of those (see
    # tool_calls.salvage) and count the rest as ``format_errors`` with the
    # timeline's ``attempted`` flag. The log line therefore reports a handled
    # condition, and at ERROR it swamps the run log and reads like a fault.
    # Silence it and trust traj/mean_salvaged_calls + traj/mean_format_errors.
    try:
        from verl.experimental.agent_loop import tool_parser as _verl_tool_parser
        _verl_tool_parser.logger.setLevel(logging.CRITICAL)
    except Exception:  # noqa: BLE001 - never let logging config break a run
        logger.debug("could not quiet verl tool_parser logger", exc_info=True)

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
            # Stop a run whose batches are infrastructure failure rather than the
            # agent failing the task -- the two are the same reward 0.0 and only
            # this exit-status breakdown can tell them apart. Deliberately allowed
            # to propagate: it is the one way to halt verl's fit() loop.
            guard_infra_failures(out, step=batch.meta_info.get("global_steps"))
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
