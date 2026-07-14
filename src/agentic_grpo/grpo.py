"""GRPO advantage estimation (critic-less, group-normalized).

GRPO replaces the PPO value network with a *group baseline*: for each prompt we
sample ``group_size`` trajectories, then normalize each trajectory's scalar
reward by the mean/std of its group. With a binary reward this is exactly the
"how much better than my siblings was this attempt" signal.

The normalized advantage is broadcast onto every response token of the
trajectory (token-level), and prompt/observation tokens are masked out via the
``response_mask`` produced by :class:`TrajectoryAdaptor`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GroupAdvantage:
    instance_id: str
    advantages: list[float]  # one scalar per trajectory in the group
    group_mean: float
    group_std: float


def compute_group_advantages(
    rewards_by_group: dict[str, list[float]],
    *,
    eps: float = 1e-6,
    std_normalize: bool = True,
) -> dict[str, GroupAdvantage]:
    """Group-relative advantage for each prompt's sampled trajectories.

    ``rewards_by_group`` maps ``instance_id -> [reward_0, ..., reward_{n-1}]``.
    """
    out: dict[str, GroupAdvantage] = {}
    for instance_id, rewards in rewards_by_group.items():
        arr = np.asarray(rewards, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std())
        if std_normalize:
            adv = (arr - mean) / (std + eps)
        else:
            adv = arr - mean
        out[instance_id] = GroupAdvantage(
            instance_id=instance_id,
            advantages=adv.tolist(),
            group_mean=mean,
            group_std=std,
        )
    return out


def broadcast_token_advantages(
    advantage: float, response_mask: list[int]
) -> list[float]:
    """Spread a trajectory-level advantage over its response tokens."""
    return [advantage if m == 1 else 0.0 for m in response_mask]


def grpo_metrics(rewards_by_group: dict[str, list[float]]) -> dict[str, float]:
    """Quick scalars describing reward signal health for logging."""
    all_rewards = [r for rs in rewards_by_group.values() for r in rs]
    if not all_rewards:
        return {}
    arr = np.asarray(all_rewards)
    # Fraction of groups with non-zero variance == groups that actually produce
    # a learning signal under GRPO (all-correct / all-wrong groups give 0 adv).
    informative = sum(
        1 for rs in rewards_by_group.values() if len(set(rs)) > 1
    ) / len(rewards_by_group)
    return {
        "grpo/mean_reward": float(arr.mean()),
        "grpo/resolve_rate": float((arr >= 1.0).mean()),
        "grpo/informative_group_frac": informative,
        "grpo/num_groups": float(len(rewards_by_group)),
    }
