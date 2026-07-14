"""Agent wrapper & environment hook (SKILL 2.1).

We do **not** write a custom environment from scratch. Instead we reuse
mini-swe-agent's ``DefaultAgent`` control loop and its SWE-bench dockerized
environment, subclassing the agent only to *intercept* the two interesting
boundaries:

* ``query()``           -> model generation (an action / tool call)
* ``execute_actions()`` -> environment observation (tool output)

Around each boundary we accumulate :class:`TrajectoryMetrics` via
:class:`TrajectoryTimer`, so the rollout produces both the trajectory and its
profiling data in one pass. After the loop terminates we parse the final
mini-swe-agent exit message to recover the ``submission`` (unified diff) which
is handed to the SWE-bench grader for the binary reward.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agentic_grpo.config import AgentConfig
from agentic_grpo.metrics import TrajectoryMetrics, TrajectoryTimer

logger = logging.getLogger("agentic_grpo.env_wrapper")

# Action strings that count as code edits (used for `edit_count` profiling).
_EDIT_MARKERS = ("apply_patch", "edit ", "str_replace", ">>>>>>> REPLACE", "<<<<<<< SEARCH")


class RolloutResult:
    """Everything a single agent rollout produces."""

    __slots__ = ("instance_id", "messages", "submission", "exit_status", "metrics")

    def __init__(
        self,
        instance_id: str,
        messages: list[dict],
        submission: str,
        exit_status: str,
        metrics: TrajectoryMetrics,
    ):
        self.instance_id = instance_id
        self.messages = messages
        self.submission = submission
        self.exit_status = exit_status
        self.metrics = metrics


class SWEBenchAgentWrapper:
    """Wrap mini-swe-agent's loop for one SWE-bench instance + one rollout.

    Parameters
    ----------
    model:
        A mini-swe-agent ``Model`` instance. In this pipeline it is built by
        :func:`agentic_grpo.rollout_backend.build_rollout_model`, i.e. the
        stock ``LitellmModel`` pointed at the SGLang OpenAI server.
    instance:
        A SWE-bench dataset row.
    config:
        Agent loop configuration.
    """

    def __init__(self, model: Any, instance: dict, config: AgentConfig):
        self.model = model
        self.instance = instance
        self.config = config
        self.instance_id = instance["instance_id"]
        self.metrics = TrajectoryMetrics(instance_id=self.instance_id)
        self.timer = TrajectoryTimer(self.metrics)
        self._agent = self._build_agent()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _build_agent(self):
        """Instantiate a DefaultAgent over a SWE-bench docker environment.

        Both pieces are reused from mini-swe-agent:

        * ``get_sb_environment`` builds the correct per-instance container
          (image-name resolution, env startup command, environment class) - we
          don't reconstruct any of that.
        * ``DefaultAgent`` is configured purely via the yaml kwargs
          (``system_template``, ``step_limit``, ...) exactly like the upstream
          ``mini-extra swebench`` runner.

        We only subclass the agent to *time* the loop for profiling.
        """
        from minisweagent.run.benchmarks.swebench import get_sb_environment  # type: ignore

        config = self._load_config()
        env = get_sb_environment(config, self.instance)
        return _build_instrumented_agent(self.model, env, self, config.get("agent", {}))

    def _load_config(self) -> dict:
        """Load mini-swe-agent's yaml config and apply our minimal overrides."""
        import yaml

        path = Path(self.config.agent_config_path)
        config: dict = yaml.safe_load(path.read_text()) if path.is_file() else {}
        config.setdefault("agent", {}).setdefault("step_limit", self.config.step_limit)
        env_cfg = config.setdefault("environment", {})
        env_cfg.setdefault("environment_class", "docker")
        env_cfg.setdefault("timeout", self.config.command_timeout)
        return config

    # ------------------------------------------------------------------
    # rollout
    # ------------------------------------------------------------------
    def rollout(self) -> RolloutResult:
        """Run the full multi-turn episode and return trajectory + metrics."""
        # The problem statement is the task, same as upstream mini-swe-agent.
        task = self.instance.get("problem_statement", "")
        with self.timer.trajectory():
            try:
                exit_extra = self._agent.run(task=task)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Rollout crashed for %s", self.instance_id)
                exit_extra = {"exit_status": f"Crashed:{type(exc).__name__}", "submission": ""}

        self.metrics.num_turns = self._agent.n_calls
        self.metrics.exit_status = exit_extra.get("exit_status", "")
        submission = exit_extra.get("submission", "") or ""

        try:
            self._agent.env.cleanup()  # release container if env supports it
        except Exception:
            pass

        return RolloutResult(
            instance_id=self.instance_id,
            messages=list(self._agent.messages),
            submission=submission,
            exit_status=self.metrics.exit_status,
            metrics=self.metrics,
        )


def _build_instrumented_agent(model: Any, env: Any, wrapper: "SWEBenchAgentWrapper", agent_kwargs: dict):
    """Build a DefaultAgent that times its own loop for profiling.

    This is the *only* thing we add on top of mini-swe-agent: hooks around
    ``query`` (generation) and ``execute_actions`` (tool/env) so we can fill in
    the latency/token breakdown in :class:`TrajectoryMetrics`. The agent's
    behaviour is otherwise 100% upstream. Built lazily so the package imports
    without mini-swe-agent installed (unit tests).
    """
    from minisweagent.agents.default import DefaultAgent  # type: ignore

    class _InstrumentedAgent(DefaultAgent):
        def query(self):
            message = super().query()
            extra = message.get("extra", {})
            # Token usage: mini-swe-agent's LitellmModel persists the raw
            # response under extra["response"]; read the server-reported usage
            # from it (falls back to any explicit extra fields). Generation
            # latency itself comes from the server's /metrics, not timed here.
            usage = (extra.get("response") or {}).get("usage") or {}
            prompt_toks = usage.get("prompt_tokens", extra.get("prompt_tokens"))
            if prompt_toks is not None:
                # context grows each turn; keep the peak (largest prefill).
                wrapper.metrics.prompt_tokens = max(wrapper.metrics.prompt_tokens, int(prompt_toks))
            comp_toks = usage.get("completion_tokens", extra.get("completion_tokens"))
            if comp_toks is not None:
                wrapper.metrics.completion_tokens += int(comp_toks)
            # Prefix-cache hits (server ground truth): cached prompt tokens skip
            # prefill, so this directly explains prefill cost across turns.
            details = usage.get("prompt_tokens_details") or {}
            cached = usage.get("cached_tokens", details.get("cached_tokens"))
            if cached is not None:
                wrapper.metrics.cached_prompt_tokens += int(cached)
            return message

        def execute_actions(self, message):
            actions = message.get("extra", {}).get("actions", [])
            wrapper.metrics.tool_call_count += len(actions)
            for action in actions:
                cmd = action.get("command", "") if isinstance(action, dict) else str(action)
                if any(marker in cmd for marker in _EDIT_MARKERS):
                    wrapper.metrics.edit_count += 1
            with wrapper.timer.tool_call():
                obs_messages = super().execute_actions(message)
            # NB: observation token counts are filled exactly by TrajectoryAdaptor
            # (real tokenizer), not guessed from char length here.
            return obs_messages

    return _InstrumentedAgent(model, env, **agent_kwargs)
