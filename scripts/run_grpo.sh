#!/usr/bin/env bash
# Launch synchronous GRPO training (3 epochs over full SWE-bench train set).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export NCCL_DEBUG=WARN
export VLLM_USE_V1=1

# --- Server-side rollout drain profiling (SGLang /metrics) ---
# The config enables SGLang metrics (rollout.engine_kwargs.sglang.enable_metrics)
# and sets capacity C (rollout.max_num_seqs). Point the poller at the rollout
# server's /metrics and tell it C so the drain-phase start is read first-hand.
# SGLang serves /metrics on the same host:port as the OpenAI API.
export AGENTIC_SGLANG_METRICS_URL="${AGENTIC_SGLANG_METRICS_URL:-http://localhost:30000/metrics}"
export AGENTIC_MAX_RUNNING_REQUESTS="${AGENTIC_MAX_RUNNING_REQUESTS:-256}"
export AGENTIC_METRICS_POLL_INTERVAL="${AGENTIC_METRICS_POLL_INTERVAL:-1.0}"

EXPERIMENT_NAME="$(date +'%Y%m%d-%H%M%S')"
LOG_DIR="${REPO_ROOT}/logs/std"
mkdir -p "${LOG_DIR}"

# Compute total_steps for 3 epochs from the prepared dataset.
TRAIN_FILE="${REPO_ROOT}/data/swebench/train.parquet"
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
  data.val_files="${REPO_ROOT}/data/swebench/val.parquet" \
  trainer.total_training_steps="${TOTAL_STEPS}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  "$@" 2>&1 | tee "${LOG_DIR}/${EXPERIMENT_NAME}"
