# Multi-turn Agentic GRPO RL Training Pipeline

Synchronous **GRPO** (Group Relative Policy Optimization) training for a
multi-turn coding agent evaluated on **SWE-bench**.

- **Model:** Qwen3-30B-A3B-Instruct-2507 (`/data0/shared/Qwen3-30B-A3B-Instruct-2507`)
- **Dataset:** SWE-bench train (`/data1/shared/swe_bench_train_hf`), full set, 3 epochs
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

Standalone loop (for auditing without a full verl cluster):

```bash
python -m sglang.launch_server --model-path /data0/shared/Qwen3-30B-A3B-Instruct-2507 \
    --tp 8 --context-length 16384 --port 30000 &

python scripts/train_standalone.py --train data/swebench/train.parquet
```



## Architecture

```
                         ┌────────────────────────────────────────────┐
 SWE-bench issue ──▶     │  SWEBenchAgentWrapper  (env_wrapper.py)    │
                         │  wraps mini-swe-agent DefaultAgent loop    │
                         │   ├─ query()           → SGLang generation │◀─┐ weights
                         │   └─ execute_actions() → docker tool obs   │  │ (policy_lag=0)
                         └───────────────┬───────────────────────────┬┘  │
                                         │ messages + submission     │   │
                            ┌────────────▼─────────────┐   ┌─────────▼───┴────────┐
                            │ reward.py                │   │ rollout_backend.py   │
                            │ swebench harness → 0/1   │   │ SGLangRolloutModel   │
                            └────────────┬─────────────┘   │ + sync_weights()     │
                                         │                 └──────────────────────┘
                            ┌────────────▼──────────────────────────────────────┐
                            │ trajectory_adaptor.py                             │
                            │ TrajectoryMetrics (token/latency profiling)       │
                            │ TrajectoryAdaptor → input_ids + response_mask     │
                            └────────────┬──────────────────────────────────────┘
                                         │ group rewards
                            ┌────────────▼───────────────┐
                            │ grpo.py                    │
                            │ group-normalized advantage │ → verl actor update → sync_weights ↺
                            └────────────────────────────┘
```



## Layout


| Path                                     | Role                                  |
| ---------------------------------------- | ------------------------------------- |
| `src/agentic_grpo/config.py`             | Hardcoded paths & hyperparameters     |
| `src/agentic_grpo/env_wrapper.py`        | Agent wrapper (reuses mini-swe-agent) |
| `src/agentic_grpo/rollout_backend.py`    | SGLang glue + on-policy weight sync   |
| `src/agentic_grpo/trajectory_adaptor.py` | Trajectory → masked GRPO tensors      |
| `src/agentic_grpo/metrics.py`            | Per-trajectory metrics (client-side)  |
| `src/agentic_grpo/server_monitor.py`     | Server-side latency/drain (SGLang `/metrics`) |
| `src/agentic_grpo/reward.py`             | Binary SWE-bench reward               |
| `src/agentic_grpo/grpo.py`               | Group-relative advantage estimation   |
| `src/agentic_grpo/agent_loop.py`         | verl AgentLoop integration            |
| `configs/grpo_swebench.yaml`             | verl trainer config                   |
| `configs/agent.yaml`                     | mini-swe-agent loop config            |
| `scripts/prepare_swebench_hf.py`         | Download + convert a gradable SWE-bench set to parquet |
| `scripts/prepare_one.py`                 | 1-instance parquet + docker image pull recipe |
| `scripts/run_grpo.sh`                    | Launch training                       |
| `scripts/train_standalone.py`            | Standalone synchronous loop           |




## Tests

```bash
pytest -q
```

