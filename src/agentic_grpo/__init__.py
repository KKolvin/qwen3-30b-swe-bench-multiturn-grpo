"""Multi-turn Agentic GRPO RL training pipeline for SWE-bench.

Everything runs inside verl's agentic rollout path:

* :mod:`agentic_grpo.agent_loop`     - the verl AgentLoop: admit -> (generate -> act -> observe)* -> release -> grade
* :mod:`agentic_grpo.episode`        - one rollout's state and its record (tokens, turn timings, metrics, timeline, dump)
* :mod:`agentic_grpo.tools`          - the tool registry: which tools are active, one ``Tool`` adapter each
* :mod:`agentic_grpo.bash_tool`      - the bash tool (mini-swe-agent schema + container exec, Qwen3 argument aliases, command policy)
* :mod:`agentic_grpo.editor_tool`    - the edit tool (view/create/str_replace/insert)
* :mod:`agentic_grpo.submit_tool`    - the submit tool: git diff against base_commit ends the episode (and is the fallback)
* :mod:`agentic_grpo.tool_calls`     - sampled text -> (tool, args): strict parse + salvage of dropped blocks
* :mod:`agentic_grpo.reward`         - binary SWE-bench outcome reward (official harness, patch-keyed cache)
* :mod:`agentic_grpo.metrics`        - per-trajectory metrics and their batch aggregation
* :mod:`agentic_grpo.timeline`       - run timeline: every rollout + training event, timestamped
* :mod:`agentic_grpo.sglang_timing`  - per-request server timestamps (queue / prefill / decode)
* :mod:`agentic_grpo.server_monitor` - SGLang /metrics scrape (srv/*)
* :mod:`agentic_grpo.config`         - AgentConfig + container env resolution
"""

from agentic_grpo.metrics import TrajectoryMetrics

__all__ = ["TrajectoryMetrics"]
__version__ = "0.1.0"
