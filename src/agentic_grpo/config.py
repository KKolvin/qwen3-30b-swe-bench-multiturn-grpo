"""Typed configuration objects shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

MODEL_PATH = "/data0/shared/Qwen3-30B-A3B-Instruct-2507"
DATASET_PATH = "/data1/shared/swe_bench_train_hf"
TRAIN_BATCH_SIZE = 256
CONTEXT_LENGTH = 16384
NUM_EPOCHS = 3


@dataclass
class RolloutConfig:
    model_path: str = MODEL_PATH
    base_url: str = "http://localhost:30000/v1"
    policy_lag: int = 0
    context_length: int = CONTEXT_LENGTH
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 2048
    group_size: int = 8
    tensor_parallel_size: int = 8
    return_logprobs: bool = True
    # Server running-batch capacity: the GPU is saturated while at least this
    # many generation requests are in flight. Must match the SGLang launch flag
    # (`--max-running-requests`). Used to locate the drain-phase start (when
    # in-flight requests fall below this and the GPU begins to idle). None ->
    # fall back to the batch's observed peak concurrency.
    max_running_requests: int | None = None

    def __post_init__(self) -> None:
        if self.policy_lag != 0:
            raise ValueError(f"policy_lag must be 0, got {self.policy_lag}")


@dataclass
class AgentConfig:
    step_limit: int = 40
    command_timeout: int = 120
    agent_config_path: str = "configs/agent.yaml"
    max_obs_tokens: int = 2048


@dataclass
class DataConfig:
    dataset_path: str = DATASET_PATH
    split: str = "train"
    output_dir: str = "data/swebench"
    seed: int = 0


@dataclass
class GRPOConfig:
    train_batch_size: int = TRAIN_BATCH_SIZE
    ppo_mini_batch_size: int = TRAIN_BATCH_SIZE
    ppo_micro_batch_size_per_gpu: int = 1
    ppo_epochs: int = 1
    learning_rate: float = 1e-6
    clip_ratio: float = 0.2
    kl_loss_coef: float = 0.001
    use_kl_loss: bool = True
    loss_agg_mode: str = "token-mean"
    num_epochs: int = NUM_EPOCHS
    total_steps: int = 0  # computed from dataset size
    adv_estimator: str = "grpo"


@dataclass
class PipelineConfig:
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    project_name: str = "agentic-grpo-swebench"
    experiment_name: str = "qwen3-30b-a3b-grpo"
