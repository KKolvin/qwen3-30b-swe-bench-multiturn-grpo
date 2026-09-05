"""Agent-side configuration: the turn-cap fallback, dataset paths, and container env resolution."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

logger = logging.getLogger("agentic_grpo.config")

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

DATASET_PATH = "/data1/shared/swe_bench_train_hf"


def resolve_container_env(env: dict) -> dict:
    """Expand ``${VAR}`` references in a mini-swe-agent ``environment.env`` block.

    mini-swe-agent passes this dict straight to ``docker exec -e KEY=VALUE`` on
    every command, and does no substitution of its own, so anything host-specific
    (the egress proxy's IP, see ``configs/agent.yaml``) has to be resolved here.

    A key referencing an **unset** variable is dropped rather than forwarded
    literally. That distinction matters: ``-e http_proxy=${AGENTIC_CONTAINER_PROXY}``
    is not "no proxy", it is a proxy whose hostname is the literal string, and
    curl/pip/urllib would then fail *every* request instead of falling back to
    direct egress.
    """
    resolved: dict = {}
    dropped: list[str] = []
    for key, value in env.items():
        if not isinstance(value, str):
            resolved[key] = value
            continue
        missing = [m.group(1) for m in _ENV_REF.finditer(value) if os.getenv(m.group(1)) is None]
        if missing:
            dropped.append(key)
            continue
        resolved[key] = _ENV_REF.sub(lambda m: os.environ[m.group(1)], value)
    if dropped:
        logger.warning(
            "container env: dropped %s (unset ${...} reference); the container falls "
            "back to direct egress",
            ", ".join(sorted(dropped)),
        )
    return resolved


def int_env(name: str, default: int) -> int:
    """An integer knob from the environment; anything but digits means the default."""
    raw = os.environ.get(name, "")
    return int(raw) if raw.isdigit() else default


@dataclass
class AgentConfig:
    # Fallback turn cap when verl's multi_turn.max_assistant_turns is unset. The
    # bash timeout and observation cap live in configs/agent.yaml and
    # configs/grpo_swebench.yaml respectively.
    step_limit: int = 40
    agent_config_path: str = "configs/agent.yaml"


@dataclass
class DataConfig:
    dataset_path: str = DATASET_PATH
    split: str = "train"
    output_dir: str = "data/swebench_verified"
    seed: int = 0
