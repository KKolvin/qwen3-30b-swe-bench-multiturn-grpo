# Multi-turn Agentic GRPO RL Training Pipeline

Synchronous **GRPO** (Group Relative Policy Optimization) training for a
multi-turn coding agent evaluated on **SWE-bench**.

- **Model:** Qwen3-30B-A3B-Instruct-2507 (`/data0/shared/Qwen3-30B-A3B-Instruct-2507`)
- **Dataset:** SWE-bench_Verified (`SWE-bench/SWE-bench_Verified`), 500 instances, 3 epochs
- **Hardware:** 8× NVIDIA B200
- **RL framework:** [verl](https://github.com/verl-project/verl) (actor / ref / GRPO)
- **Rollout engine:** [SGLang](https://github.com/sgl-project/sglang), synchronous (`policy_lag = 0`)
- **Agent + environment:** [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) + dockerized SWE-bench env
- **Reward:** binary outcome — `1.0` if resolved, else `0.0`
- **Batch size:** 256 | **Context length:** 16384



## Setup

```bash
pip install "verl[sglang]>=0.5.0"
pip install -e .
```

Docker is required for SWE-bench instance images.

## Run

```bash
# 1. Prepare a gradable dataset from Hugging Face (SWE-bench_Verified by default).
python scripts/prepare_swebench_hf.py

# 2. Launch GRPO training (3 epochs, auto-computes total_steps).
bash scripts/run_grpo.sh
```


## Architecture

```
   verl trainer (Ray) ──── weights, policy_lag=0 ────▶  SGLang rollout server
          │                                                      ▲
          │ one AgentLoop.run() per sampled prompt               │ token-in / token-out
          ▼                                                      │
   ┌─────────────────────────────────────────────────────────────┴──────┐
   │ agent_loop.py   SWEBenchAgentLoop                                  │
   │   generate ─▶ parse <tool_call> ─▶ run tool in docker ─▶ append obs│
   │   tools: bash | str_replace_based_edit_tool (editor_tool.py)       │
   │   ends on submit marker / turn limit / context limit ─▶ patch      │
   └───────────────┬────────────────────────────────────┬───────────────┘
                   │ patch                              │ TrajectoryMetrics + timeline events
      ┌────────────▼─────────────┐         ┌────────────▼──────────────────────────┐
      │ reward.py                │         │ metrics.py · timeline.py              │
      │ swebench harness → 0/1   │         │ sglang_timing.py · server_monitor.py  │
      └────────────┬─────────────┘         └───────────────────────────────────────┘
                   ▼
      reward on last token ─▶ verl GRPO advantage ─▶ actor update ─▶ weight sync ↺
```

## Layout


| Path                                     | Role                                  |
| ---------------------------------------- | ------------------------------------- |
| `src/agentic_grpo/config.py`             | AgentConfig + container env resolution |
| `src/agentic_grpo/agent_loop.py`         | verl AgentLoop: the multi-turn rollout |
| `src/agentic_grpo/tools.py`              | Tool registry: schema + arg shaping + runner per tool |
| `src/agentic_grpo/tool_calls.py`         | Sampled text → (tool, args): strict parse + salvage |
| `src/agentic_grpo/editor_tool.py`        | str_replace_based_edit_tool (next to bash) |
| `src/agentic_grpo/metrics.py`            | Per-trajectory metrics (client-side)  |
| `src/agentic_grpo/server_monitor.py`     | Server-side latency/drain (SGLang `/metrics`) |
| `src/agentic_grpo/sglang_timing.py`      | Per-request queue/prefill/decode timestamps |
| `src/agentic_grpo/timeline.py`           | Run timeline: every rollout + training event, timestamped |
| `src/agentic_grpo/reward.py`             | Binary SWE-bench reward               |
| `configs/grpo_swebench.yaml`             | verl trainer config                   |
| `configs/agent.yaml`                     | mini-swe-agent loop config            |
| `scripts/prepare_swebench_hf.py`         | Download + convert a gradable SWE-bench set to parquet |
| `scripts/prepare_one.py`                 | 1-instance parquet + docker image name (smoke test) |
| `scripts/run_grpo.sh`                    | Launch training                       |
| `scripts/build_timeline.py`              | Merge a run's timeline shards into one `timeline.json` |




## Tests

```bash
pytest -q
```

The suite has no verl/sglang/CUDA dependency — `agent_loop.py` falls back to
attribute-compatible stubs for verl's `AgentLoopBase`/`AgentLoopOutput` when the
import fails, so the unit tests run on a bare laptop checkout.

