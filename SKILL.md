# Skill: Multi-turn Agentic GRPO RL Training Pipeline

## 1. Objective & Architecture Overview
Build a synchronous GRPO (Group Relative Policy Optimization) training pipeline for a multi-turn coding agent evaluating on SWE-bench.

**Infrastructure & Hardware:**
* **Hardware:** 8x NVIDIA B200
* **Model:** Qwen3-30B-A3B
* **Dataset:** SWE-bench (Subset: 100~500 issues)
* **Framework integration:** Expecting integration with an RL framework (e.g., `verl` or `OpenRLHF` style actor-rollout decoupling).

**Core Requirements:**
1. **Rollout Backend:** `sglang` in synchronous mode (0 updates of policy_lag). The actor weights must strictly match the rollout engine weights per GRPO step.
2. **Environment:** Heavily reuse `mini-swe-agent` and its underlying SWE-bench dockerized environment.
3. **Reward Function:** Binary outcome. `1.0` if `tests_pass` (resolved) else `0.0`.
4. **Metrics Profiling:** High-resolution trajectory statistics (token counts and latency breakdowns) for multi-turn bottlenecks.

---

## 2. Component Implementation Guide

### 2.1 Agent Wrapper & Environment Hook (`env_wrapper.py`)
Do not write a custom environment from scratch. Import and wrap `mini-swe-agent`'s core loop.

* **Task:** Create `SWEBenchAgentWrapper`.
* **Behavior:** * Initialize the `mini-swe-agent` state machine.
    * Intercept the `step()` function to intercept actions (model generations) and observations (tool outputs).
    * Parse the final evaluation output from SWE-bench to extract the boolean `resolved` status for the reward.

### 2.2 SGLang Rollout Backend (`rollout_backend.py`)
Implement the generation interface using SGLang's engine APIs.

* **Synchronization:** Ensure `policy_lag = 0`. The rollout must block the training loop, generate trajectories, and immediately pass them to the critic/reference models for GRPO advantage estimation before the actor updates.
* **Context Management:** Enforce `context_length = 16384`. Implement truncation strategies for tool outputs if the context window is breached during multi-turn rollouts.

### 2.3 Trajectory Adaptor & Metrics Tracker (`trajectory_adaptor.py`)
This is the most critical monitoring component. You must instrument the multi-turn interaction loop to capture granular profiling data.

Define a `TrajectoryMetrics` dataclass and populate it during the rollout:

```python
from dataclasses import dataclass
import time

@dataclass
class TrajectoryMetrics:
    # Token Statistics
    prompt_tokens: int = 0
    total_trajectory_tokens: int = 0
    tool_obs_tokens: list[int] = None # Tokens per tool observation
    
    # Latency Breakdown (seconds)
    prefill_time: float = 0.0
    total_trajectory_time: float = 0.0
    total_tool_call_time: float = 0.0
    total_rollout_time: float = 0.0 # Includes SGLang overhead + Env overhead
    
    # Action/Event Counts
    tool_call_count: int = 0
    edit_count: int = 0 # Specific parsing of 'edit' tool calls
    
    # Artifacts
    pytest_output_length: int = 0 # Character or token length of final test run