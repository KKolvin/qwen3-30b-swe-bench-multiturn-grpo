"""Multi-turn Agentic GRPO RL training pipeline for SWE-bench.

Public surface mirrors the components described in SKILL.md:

* :mod:`agentic_grpo.env_wrapper`        - SWEBenchAgentWrapper (mini-swe-agent hook)
* :mod:`agentic_grpo.rollout_backend`    - SGLang synchronous rollout backend (policy_lag=0)
* :mod:`agentic_grpo.metrics`            - per-trajectory metrics (client-side)
* :mod:`agentic_grpo.server_monitor`     - server-side latency / drain from SGLang /metrics
* :mod:`agentic_grpo.trajectory_adaptor` - verl tensor adaptor (masked GRPO tensors)
* :mod:`agentic_grpo.reward`             - binary SWE-bench outcome reward
* :mod:`agentic_grpo.agent_loop`         - verl AgentLoop integration
"""

from agentic_grpo.metrics import TrajectoryMetrics

__all__ = ["TrajectoryMetrics"]
__version__ = "0.1.0"
