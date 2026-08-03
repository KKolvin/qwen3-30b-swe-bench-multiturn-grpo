#!/usr/bin/env bash
# Launch synchronous GRPO training (3 epochs over SWE-bench_Verified train split).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export NCCL_DEBUG=WARN
export VLLM_USE_V1=1

# Let the caching allocator hand physical pages back. Without this it keeps
# reserved-but-free blocks in fixed segments, and torch_memory_saver's
# cu_mem_create -- which needs PHYSICAL pages -- cannot reclaim them. That is
# what killed the 2026-07-30 run at the weight sync (cards at 162GB/183GB while
# resume_memory_occupation failed 3x and raised). Must reach the Ray WORKERS,
# where FSDP and SGLang actually allocate, so it is also pushed into
# runtime_env.env_vars below; exporting it here alone would only affect the driver.
# PYTORCH_ALLOC_CONF, not PYTORCH_CUDA_ALLOC_CONF: torch 2.9.1 still honours the
# old name but warns "PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF".
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

# --- GPU selection (default: the whole node, 8x B200) ---
# The config in configs/grpo_swebench.yaml is the 8-GPU baseline and should stay
# that way. AGENTIC_GPUS is the escape hatch for the days when a card is not
# ours — e.g. another user's vLLM server sitting on GPU 0:
#
#   AGENTIC_GPUS=1,2,3,4 bash scripts/run_grpo.sh
#
# Unset (the normal case) this block is a no-op: no CUDA_VISIBLE_DEVICES is set
# and no overrides are appended, so the 8-GPU config applies verbatim.
GPU_OVERRIDES=()
if [ -n "${AGENTIC_GPUS:-}" ]; then
  # Ray reads CUDA_VISIBLE_DEVICES at ray.init to size the cluster, then rewrites
  # it per actor — so set it HERE and never push it into runtime_env.env_vars,
  # which would clobber Ray's per-actor mapping and land every worker on one card.
  export CUDA_VISIBLE_DEVICES="${AGENTIC_GPUS}"
  N_GPUS=$(awk -F, '{print NF}' <<<"${AGENTIC_GPUS}")

  # TP = N keeps a single rollout replica, matching the 8-GPU baseline's TP=8.
  #
  # gpu_memory_utilization must come down as N shrinks, because FSDP shards bf16
  # params + grads (30.5B x 2 bytes x 2 = 122GB) across N cards. These values are
  # MEASURED, not derived -- the earlier derivation (0.75 - shard delta, giving
  # 0.66 at N=4) OOMed at the weight sync on 2026-07-30 and killed a 3h38m run:
  #
  #   RuntimeError: Failed to complete async request to resume_memory_occupation
  #
  # The derivation assumed non-pool residency was just the 30.5GB shard, but the
  # cards sat at 162GB/183GB: PyTorch's caching allocator holds reserved-but-free
  # blocks that torch_memory_saver's cu_mem_create cannot take, since it needs
  # PHYSICAL pages. Real non-pool residency was ~41GB, ~1.35x the shard. So:
  #   N=8 -> 0.75  (proven baseline, unchanged)
  #   N=4 -> 0.55  (0.66 is known to fail; this leaves ~82GB for non-pool)
  #   other N -> shard math with the measured 1.35x allocator overhead, floored.
  # Losing KV cache is close to free here: step-1 profiling put inference MFU at
  # 3.6% because rollout is bound by docker container slots (~128 live), not the
  # GPU. AGENTIC_GPU_MEM_UTIL overrides this outright if you want to tune it.
  case "${N_GPUS}" in
    8) GPU_MEM_UTIL="0.75" ;;
    4) GPU_MEM_UTIL="0.55" ;;
    *) GPU_MEM_UTIL=$(python3 -c "
n = ${N_GPUS}
# 1.35x accounts for allocator blocks cu_mem_create cannot reclaim (measured).
util = 0.75 - 1.35 * (122.0/n - 122.0/8) / 183.0
print(f'{max(0.30, (util * 100 // 1) / 100):.2f}')") ;;
  esac
  GPU_MEM_UTIL="${AGENTIC_GPU_MEM_UTIL:-${GPU_MEM_UTIL}}"

  # Park the actor's params on CPU too. optimizer_offload is already true for the
  # same reason (see the config); leaving param_offload false pinned another
  # 122/2/N GB (15.2GB at N=4) on the card across the weight sync, which is
  # exactly the memory resume_memory_occupation could not get back. The cost is a
  # CPU<->GPU param transfer per step, cheap next to a dead run. Only applied on
  # the reduced-GPU path -- the 8-GPU baseline is proven with it false.
  echo "AGENTIC_GPUS=${AGENTIC_GPUS} -> ${N_GPUS} GPUs, TP=${N_GPUS}, gpu_memory_utilization=${GPU_MEM_UTIL}, actor param_offload=True (baseline is 8 GPUs / TP=8 / 0.75 / param_offload=False)"
  GPU_OVERRIDES=(
    trainer.n_gpus_per_node="${N_GPUS}"
    actor_rollout_ref.rollout.tensor_model_parallel_size="${N_GPUS}"
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}"
    actor_rollout_ref.actor.fsdp_config.param_offload=True
  )
fi

# --- Server-side rollout drain profiling (SGLang /metrics) ---
# ON by default. verl runs its SGLang replicas on EPHEMERAL per-replica ports, so
# there is no static URL to configure -- which is why every run before this
# silently recorded zero srv/* metrics. server_monitor.discover_metrics_url() now
# resolves the address at runtime from verl's named Ray actor
# (sglang_server_<replica>_<node>.get_server_address), the same handle the rollout
# worker uses. Requires rollout.engine_kwargs.sglang.enable_metrics=true (it is,
# in configs/grpo_swebench.yaml) or /metrics returns nothing.
#   AGENTIC_SGLANG_METRICS_URL - pin an explicit URL, skipping discovery (needed
#     for the standalone loop, whose server is on a known port)
#   AGENTIC_SGLANG_METRICS=0   - disable the monitor entirely
# Capacity C mirrors rollout.max_num_seqs and sets the saturation threshold used
# to locate the drain start; polling is ~1 req/s, negligible against a rollout.
export AGENTIC_MAX_RUNNING_REQUESTS="${AGENTIC_MAX_RUNNING_REQUESTS:-256}"
export AGENTIC_METRICS_POLL_INTERVAL="${AGENTIC_METRICS_POLL_INTERVAL:-1.0}"

# --- Docker container concurrency (rootless daemon) ---
# Two separate ceilings, both PER AgentLoopWorker process (there are
# rollout.agent.num_workers of them, 8 by default), so multiply by that for the
# cluster total:
#   MAX_CONCURRENT_CONTAINERS - bounds the `docker run` burst rate; released as
#     soon as the container is up.
#   MAX_LIVE_CONTAINERS       - bounds containers ALIVE at once; held for the
#     whole episode. Without this the first full-batch run (256x8) accumulated
#     193 live containers and the rootless daemon started returning exit 125,
#     silently degrading rollouts to reward-0 samples. 8 x 8 workers = ~64 live.
# Raise MAX_LIVE_CONTAINERS to buy rollout throughput at the cost of docker
# stability; this is the first knob to try if a step is too slow.
# 16 x 8 workers = ~128 live. Profiling step 1 showed rollout was bound by this,
# not by the GPU: trajectories averaged 2148s wall but only 121s of generation and
# 13s of tool calls, i.e. ~2000s waiting for a slot, and 2048 rollouts / 64 slots =
# 32 waves x ~146s = 4672s matched timing_s/gen (4682s) almost exactly. Only 64
# concurrent requests against max_num_seqs=256 also left inference MFU at 3.6%.
# 193 live containers is the level that broke the daemon (exit 125) — but that was
# before MAX_CONCURRENT_CONTAINERS capped the start rate, which is the thing the
# daemon actually chokes on.
export AGENTIC_MAX_CONCURRENT_CONTAINERS="${AGENTIC_MAX_CONCURRENT_CONTAINERS:-8}"
export AGENTIC_MAX_LIVE_CONTAINERS="${AGENTIC_MAX_LIVE_CONTAINERS:-16}"

# Point every docker client at the ROOTLESS daemon (store on /data1). The docker
# CLI resolves this from its `default` context, so mini-swe-agent's subprocess
# calls work without it — but the SWE-bench harness grades via docker-py's
# `from_env()`, which reads DOCKER_HOST and otherwise falls back to the ROOTFUL
# socket and dies with PermissionError. That failure was swallowed into
# reward=0.0, making every rollout look unresolved regardless of its patch.
export DOCKER_HOST="${DOCKER_HOST:-unix:///run/user/$(id -u)/docker.sock}"

EXPERIMENT_NAME="$(date +'%Y%m%d-%H%M%S')"

# Checkpoints go on /data0 (multi-TB), NOT the repo disk on / (~885G, near full).
# A 30B FSDP checkpoint is ~60GB+; writing it under the repo fills / and the save
# fails mid-write ("PytorchStreamWriter ... file write failed"). Override the
# checkpoint root; AGENTIC_CKPT_ROOT lets you relocate it.
CKPT_ROOT="${AGENTIC_CKPT_ROOT:-/data0/shared/${USER:-nobody}/agentic-checkpoints}"
mkdir -p "${CKPT_ROOT}"

# --- Resume (default: fresh run) ---
# trainer.resume_mode is already 'auto', meaning "resume from the latest
# checkpoint in default_local_dir" -- but that is a no-op here, because
# EXPERIMENT_NAME above is a fresh timestamp on every launch, so default_local_dir
# points at a brand-new EMPTY directory and 'auto' silently finds nothing. Pass
# the previous run's experiment name to point back at its checkpoints:
#
#   AGENTIC_RESUME=20260731-025759 bash scripts/run_grpo.sh
#
# Training picks up at the step AFTER the newest global_step_* in that directory,
# so the checkpoint interval (save_freq, 3) bounds how much work a stop can lose.
LOG_SUFFIX=""
if [ -n "${AGENTIC_RESUME:-}" ]; then
  EXPERIMENT_NAME="${AGENTIC_RESUME}"
  RESUME_DIR="${CKPT_ROOT}/${EXPERIMENT_NAME}"
  # Fail loudly rather than let resume_mode=auto find nothing and silently
  # restart from scratch -- that looks like a resume until the reward resets.
  if [ ! -d "${RESUME_DIR}" ]; then
    echo "AGENTIC_RESUME=${EXPERIMENT_NAME} but ${RESUME_DIR} does not exist." >&2
    echo "Available: $(ls "${CKPT_ROOT}" 2>/dev/null | tr '\n' ' ')" >&2
    exit 1
  fi
  # `|| true` is load-bearing: under `set -euo pipefail` a no-match grep fails the
  # whole pipeline, and the failing command substitution would kill the script
  # right here -- silently, with exit 1 and none of the diagnostics below.
  LATEST_STEP=$(ls "${RESUME_DIR}" 2>/dev/null | grep -oE 'global_step_[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1 || true)
  if [ -z "${LATEST_STEP}" ]; then
    echo "AGENTIC_RESUME=${EXPERIMENT_NAME}: ${RESUME_DIR} holds no global_step_* checkpoint." >&2
    echo "Contents: $(ls "${RESUME_DIR}" 2>/dev/null | tr '\n' ' ')" >&2
    exit 1
  fi
  # Reusing EXPERIMENT_NAME reuses the log path too, and the `tee` below
  # TRUNCATES -- which would destroy the original run's log. Suffix the resume's
  # log so both survive in the same run directory.
  LOG_SUFFIX="-resume-$(date +'%H%M%S')"
  echo "Resuming ${EXPERIMENT_NAME} from global_step_${LATEST_STEP} (${RESUME_DIR})"
fi

# --- Run artifacts: one directory per experiment under analysis/ ---
# The console log lives next to that run's analysis (metrics.csv, README.md)
# rather than in a flat logs/std/, so everything about a run is in one place.
# This has to come AFTER the resume block, since AGENTIC_RESUME rewrites
# EXPERIMENT_NAME and a resume must land in the ORIGINAL run's directory.
# Note this is the repo disk (/ is ~99% full) -- logs are a few MB per run, but
# checkpoints deliberately go to /data0 instead (see CKPT_ROOT above).
RUN_DIR="${REPO_ROOT}/analysis/${EXPERIMENT_NAME}"
mkdir -p "${RUN_DIR}"
LOG_FILE="${RUN_DIR}/run${LOG_SUFFIX}.log"
echo "Run artifacts: ${RUN_DIR} (log -> ${LOG_FILE})"

# Ray's session dir goes on /data0 for the same reason as the checkpoints: it
# defaults to /tmp/ray, i.e. the root volume, which sits at 874/885GB (12GB free).
# The raylet's file_system_monitor logs every 10s at that level --
#   "... is over 95% full ... Object creation will fail if spilling is required"
# -- and an object store spill onto a full disk takes the run down mid-step. The
# session dir also holds the object-spilling directory by default, so relocating
# _temp_dir moves both. Keep this path SHORT: Ray puts unix domain sockets under
# <temp>/sockets/, and those cap at ~107 chars once the session_<timestamp>_<pid>
# component is appended (this base leaves ~20 chars of margin).
RAY_TMP_ROOT="${AGENTIC_RAY_TMPDIR:-/data0/shared/${USER:-nobody}/ray}"
mkdir -p "${RAY_TMP_ROOT}"

# --- Rollout tracing (default: off) ---
# verl wraps each `agent_loop.run(...)` call in `rollout_trace_attr(...)` (see
# verl/experimental/agent_loop/agent_loop.py), so turning a backend on captures
# ONE span per trajectory, tagged step/sample_index/rollout_n/validate. That is
# the level at which you can read what an episode actually did.
#
#   AGENTIC_TRACE=weave AGENTIC_GPUS=1,2,3,4 bash scripts/run_grpo.sh
#
# CAVEAT: the finer-grained per-turn spans come from `@rollout_trace_op` on
# verl's own tool-calling components (tool_agent_loop / tool_parser / base_tool).
# SWEBenchAgentLoop does not use those, so per-generate and per-tool-call spans
# will NOT appear until its methods are decorated. Trajectory-level only for now.
#
# Backends: weave (reuses the wandb creds in ~/.netrc and the trainer
# project_name), mlflow, trackio. weave/mlflow/trackio are NOT declared
# dependencies -- install the one you name or the run dies at rollout init.
#
# Volume: traces/step = AGENTIC_TRACE_SAMPLES x rollout.agent.num_workers (8) x
# rollout.n (8). The default of 2 gives 2x8x8 = 128 traces/step; leaving it empty
# means "all samples", i.e. 256 x 8 = 2048 traces/step, which is rarely what you
# want. token2text is on so spans carry readable text instead of token ids.
TRACE_OVERRIDES=()
if [ -n "${AGENTIC_TRACE:-}" ]; then
  TRACE_SAMPLES="${AGENTIC_TRACE_SAMPLES:-2}"
  if ! ./.venv/bin/python -c "import ${AGENTIC_TRACE}" 2>/dev/null; then
    echo "AGENTIC_TRACE=${AGENTIC_TRACE} but that package is not importable in .venv; install it or unset AGENTIC_TRACE." >&2
    exit 1
  fi
  echo "Rollout tracing: backend=${AGENTIC_TRACE}, ~$((TRACE_SAMPLES * 8 * 8)) traces/step (samples_per_worker=${TRACE_SAMPLES})"
  TRACE_OVERRIDES=(
    actor_rollout_ref.rollout.trace.backend="${AGENTIC_TRACE}"
    actor_rollout_ref.rollout.trace.token2text=True
    actor_rollout_ref.rollout.trace.max_samples_per_step_per_worker="${TRACE_SAMPLES}"
  )
fi

# Compute total_steps for 3 epochs from the prepared dataset.
TRAIN_FILE="${REPO_ROOT}/data/swebench_verified/train.parquet"
if [ ! -f "$TRAIN_FILE" ]; then
  echo "Run 'python scripts/prepare_swebench_hf.py' first." >&2
  exit 1
fi
N_TRAIN=$(python3 -c "import pyarrow.parquet as pq; print(pq.read_metadata('${TRAIN_FILE}').num_rows)")
TOTAL_STEPS=$(python3 -c "import math; print(math.ceil(${N_TRAIN} / 256) * 3)")
echo "Training: ${N_TRAIN} instances, batch_size=256, 3 epochs -> ${TOTAL_STEPS} steps (wandb run: ${EXPERIMENT_NAME})"

# NB: launched via scripts/verl_entry.py (not `-m verl.trainer.main_ppo`) so our
# compute_data_metrics patch is applied inside the TaskRunner actor where fit()
# runs — otherwise the custom W&B metrics never get logged. config_path/name are
# baked into verl_entry.py's @hydra.main; only overrides are passed here.
python3 "${REPO_ROOT}/scripts/verl_entry.py" \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${REPO_ROOT}/data/swebench_verified/val.parquet" \
  trainer.total_training_steps="${TOTAL_STEPS}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${CKPT_ROOT}/${EXPERIMENT_NAME}" \
  +ray_kwargs.ray_init.runtime_env.env_vars.PATH="${REPO_ROOT}/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  +ray_kwargs.ray_init.runtime_env.env_vars.AGENTIC_MAX_CONCURRENT_CONTAINERS=\"${AGENTIC_MAX_CONCURRENT_CONTAINERS}\" \
  +ray_kwargs.ray_init.runtime_env.env_vars.AGENTIC_MAX_LIVE_CONTAINERS=\"${AGENTIC_MAX_LIVE_CONTAINERS}\" \
  +ray_kwargs.ray_init.runtime_env.env_vars.AGENTIC_CONTAINER_START_RETRIES=\"${AGENTIC_CONTAINER_START_RETRIES:-3}\" \
  +ray_kwargs.ray_init.runtime_env.env_vars.DOCKER_HOST=\"${DOCKER_HOST}\" \
  +ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_ALLOC_CONF=\"${PYTORCH_ALLOC_CONF}\" \
  +ray_kwargs.ray_init._temp_dir="${RAY_TMP_ROOT}" \
  ${GPU_OVERRIDES[@]+"${GPU_OVERRIDES[@]}"} \
  ${TRACE_OVERRIDES[@]+"${TRACE_OVERRIDES[@]}"} \
  "$@" 2>&1 | tee "${LOG_FILE}"
