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
# sglang (0.5.8) pins transformers==4.57.1, while verl 0.9.0 *declares*
# transformers>=5.5.3 -- a floor its code does not actually need on this path
# (it still carries 4.5x version gates). So install verl without its deps:
git clone https://github.com/volcengine/verl /data0/shared/$USER/verl
git -C /data0/shared/$USER/verl checkout release/v0.9.0
pip install "sglang[all]==0.5.8"
pip install -e /data0/shared/$USER/verl --no-deps   # verl 0.9.0, editable
pip install -e .
```

verl is pinned to the **`release/v0.9.0`** branch, not the 0.8.0 wheel: 0.8.0
OOMs at the SGLang weight sync (`resume_memory_occupation` ->
`cu_mem_create: out of memory`), which killed the 4-GPU run on 2026-07-30 and
the 8-GPU run 20260908-052218 at step 2. 0.9.0 carries both fixes —
`aggressive_empty_cache()` before `rollout.resume(tags=["weights"])`
(commit d908005d) and `enable_weights_cpu_backup` for HYBRID rollout mode —
and `trainer.use_v1=false` in the config keeps the run on the legacy
`RayPPOTrainer` path this repo hooks.

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
   │   tools registry (tools.py): bash_tool.py | editor_tool.py         │
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
| `src/agentic_grpo/agent_loop.py`         | verl AgentLoop: the rollout as phases (admit → step* → release → grade) |
| `src/agentic_grpo/episode.py`            | One rollout's state + record: tokens, turn timings, metrics, timeline, dump |
| `src/agentic_grpo/tools.py`              | Tool registry: which tools are active, one `Tool` adapter each; `AGENTIC_EDIT_TOOL=0` drops the editor from schemas, prompts and nudges alike |
| `src/agentic_grpo/bash_tool.py`          | `bash`: mini-swe-agent's schema + container exec, Qwen3 argument aliases, command policy (installs/network, servers, whole-repo lint refused), timeout wording |
| `src/agentic_grpo/editor_tool.py`        | `str_replace_based_edit_tool`: view/create/str_replace/insert over base64 file I/O |
| `src/agentic_grpo/submit_tool.py`        | `submit`: ends the episode with `git diff <base_commit>` as the patch (also the end-of-episode fallback); `AGENTIC_SUBMIT_TOOL=0` restores the marker protocol |
| `src/agentic_grpo/tool_calls.py`         | Sampled text → (tool, args): strict parse + salvage |
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

