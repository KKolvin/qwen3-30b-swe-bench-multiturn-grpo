"""verl AgentLoop integration (SKILL 1: actor-rollout decoupling).

verl's agentic RL path separates the inference *server* (SGLang ``AsyncServer``)
from the *agent client*. To plug our SWE-bench coding agent in, we implement a
custom ``AgentLoop`` whose ``run`` method:

1. builds a :class:`SWEBenchAgentWrapper` driven by a thin Model adaptor that
   routes generations through verl's ``server_manager.generate`` (so the engine
   stays under verl's synchronous weight control, policy_lag=0);
2. runs the multi-turn episode against the dockerized SWE-bench env;
3. grades the final patch -> binary reward;
4. flattens the trajectory into masked token tensors via
   :class:`TrajectoryAdaptor` and returns them as an ``AgentLoopOutput``.

Register this loop in the agent-loop config and point the dataset rows at it via
the ``agent_name`` field (see ``configs/grpo_swebench.yaml``).

NOTE: verl's agent-loop API surface moves quickly across releases. The import of
the base class is done defensively; if it isn't available the module still
imports so the standalone trainer and unit tests work.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from agentic_grpo.config import AgentConfig, RolloutConfig
from agentic_grpo.env_wrapper import SWEBenchAgentWrapper
from agentic_grpo.reward import apply_to_metrics, compute_reward
from agentic_grpo.rollout_backend import build_rollout_model
from agentic_grpo.trajectory_adaptor import AdaptedTrajectory, TrajectoryAdaptor

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

    class AgentLoopMetrics:  # type: ignore
        def __init__(self, generate_sequences=0.0, tool_calls=0.0, compute_score=0.0):
            self.generate_sequences = generate_sequences
            self.tool_calls = tool_calls
            self.compute_score = compute_score

    AgentLoopOutput = dict  # type: ignore


def _rollout_base_url(server_manager: Any) -> str:
    """OpenAI endpoint of verl's async rollout server for this worker.

    verl's server-based rollout serves an OpenAI-compatible API; mini-swe-agent
    talks to it directly. The address is exposed by the server manager (or via
    ``VERL_ROLLOUT_BASE_URL`` as an override) so we never re-plumb generation.
    """
    if "VERL_ROLLOUT_BASE_URL" in os.environ:
        return os.environ["VERL_ROLLOUT_BASE_URL"]
    address = getattr(server_manager, "server_address", None) or getattr(
        server_manager, "address", "localhost:30000"
    )
    return f"http://{address}/v1"


@register("swebench_agent")
class SWEBenchAgentLoop(AgentLoopBase):
    """Custom verl agent loop for multi-turn SWE-bench coding rollouts."""

    async def run(self, sampling_params: dict, **kwargs) -> "AgentLoopOutput":
        instance = kwargs["instance"] if "instance" in kwargs else kwargs.get("extra_info", {})
        rollout_cfg = RolloutConfig(
            model_path=self.config.actor_rollout_ref.model.path,
            base_url=_rollout_base_url(self.server_manager),
            context_length=self.config.actor_rollout_ref.rollout.max_model_len,
            group_size=self.config.actor_rollout_ref.rollout.n,
        )
        agent_cfg = AgentConfig()

        model = build_rollout_model(rollout_cfg)
        wrapper = SWEBenchAgentWrapper(model=model, instance=instance, config=agent_cfg)
        result = wrapper.rollout()

        t0 = time.perf_counter()
        rr = compute_reward(instance, result.submission)
        compute_score_s = time.perf_counter() - t0

        apply_to_metrics(result.metrics, rr)

        adaptor = TrajectoryAdaptor(self.tokenizer, max_length=rollout_cfg.context_length)
        adapted = adaptor.adapt(result.messages, reward=rr.reward, metrics=result.metrics)

        return _build_output(adapted, compute_score_s=compute_score_s)


def _build_output(adapted: AdaptedTrajectory, *, compute_score_s: float) -> Any:
    """Pack an AdaptedTrajectory into verl's AgentLoopOutput (or a dict stub)."""
    metrics = adapted.metrics
    prompt_ids = [tid for tid, mask in zip(adapted.input_ids, adapted.response_mask) if mask == 0]
    response_ids = [tid for tid, mask in zip(adapted.input_ids, adapted.response_mask) if mask == 1]

    loop_metrics = AgentLoopMetrics(
        generate_sequences=metrics.generation_time(),  # derived (traj - tool)
        tool_calls=metrics.total_tool_call_time,
        compute_score=compute_score_s,
    )
    extra_fields = {
        "trajectory_metrics": metrics.to_dict(),
        "tool_call_counts": metrics.tool_call_count,
    }

    if not _HAS_VERL:
        return {
            "prompt_ids": prompt_ids,
            "response_ids": response_ids,
            "response_mask": adapted.response_mask,
            "reward_score": adapted.reward,
            "num_turns": metrics.num_turns,
            "metrics": {
                "generate_sequences": loop_metrics.generate_sequences,
                "tool_calls": loop_metrics.tool_calls,
                "compute_score": loop_metrics.compute_score,
            },
            "extra_fields": extra_fields,
        }

    return AgentLoopOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=adapted.response_mask,
        reward_score=adapted.reward,
        num_turns=metrics.num_turns,
        metrics=loop_metrics,
        extra_fields=extra_fields,
    )


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
        # this step's rollout window (no-op unless AGENTIC_SGLANG_METRICS_URL set).
        monitor = get_shared_monitor()
        if monitor is not None:
            out.update(monitor.summarize_since_last(max_concurrency))
        return out

    metric_utils.compute_data_metrics = compute_data_metrics
    ray_trainer.compute_data_metrics = compute_data_metrics
    _METRICS_PATCHED = True


_patch_verl_data_metrics()
