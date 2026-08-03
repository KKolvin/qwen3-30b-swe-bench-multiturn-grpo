#!/usr/bin/env python
"""Standalone synchronous GRPO loop (framework-light reference implementation).

This is the readable, self-contained counterpart to the full verl launch
(``scripts/run_grpo.sh``). It makes the synchronous, policy_lag=0 contract from
the SKILL spec explicit and easy to audit:

    for step in range(total_steps):
        weight_sync.sync(actor.named_parameters())      # rollout == actor
        weight_sync.assert_on_policy(actor.version)     # guard policy_lag==0
        groups = rollout(batch, group_size)             # blocks training
        rewards = grade(groups)                         # binary SWE-bench reward
        advantages = compute_group_advantages(rewards)  # critic-less baseline
        actor.update(groups, advantages)                # GRPO actor step
        # next step re-syncs weights -> never generates with stale weights

The actual actor backward pass is delegated to ``actor.update`` which you wire
to FSDP/Megatron (left as an integration point). Everything that defines the
*pipeline* - synchronization, rollout, reward, advantage, metrics - is here and
runnable end-to-end given the engine + actor are provided.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentic_grpo.config import PipelineConfig  # noqa: E402
from agentic_grpo.env_wrapper import SWEBenchAgentWrapper  # noqa: E402
from agentic_grpo.grpo import (  # noqa: E402
    broadcast_token_advantages,
    compute_group_advantages,
    grpo_metrics,
)
from agentic_grpo.metrics import TrajectoryMetrics  # noqa: E402
from agentic_grpo.reward import apply_to_metrics, compute_reward  # noqa: E402
from agentic_grpo.rollout_backend import SGLangWeightSync, build_rollout_model  # noqa: E402
from agentic_grpo.trajectory_adaptor import TrajectoryAdaptor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("train_standalone")


def load_instances(parquet_path: str) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    return [r["extra_info"] for r in rows]


def rollout_group(
    model,
    instance: dict,
    cfg: PipelineConfig,
) -> list:
    """Sample `group_size` independent trajectories for one issue."""
    results = []
    for _ in range(cfg.rollout.group_size):
        wrapper = SWEBenchAgentWrapper(model=model, instance=instance, config=cfg.agent)
        results.append(wrapper.rollout())
    return results


def _dump_trajectory(sink, step: int, group_idx: int, instance: dict, r, rr) -> None:
    """Append one per-trajectory record (JSONL) for offline inspection.

    Captures everything a single rollout produced — the full message trace, the
    submitted diff, the binary reward, and the complete TrajectoryMetrics
    (token/latency/turn breakdown + harness grading) — so a smoke run can be
    audited without a trainer/wandb.
    """
    rec = {
        "step": step,
        "group_idx": group_idx,
        "instance_id": instance["instance_id"],
        "reward": rr.reward,
        "resolved": rr.resolved,
        "empty_patch": rr.empty_patch,
        "eval_error": rr.eval_error,
        "exit_status": r.exit_status,
        "num_turns": r.metrics.num_turns,
        "submission": r.submission,
        "metrics": r.metrics.to_dict(),
        "messages": r.messages,
    }
    sink.write(json.dumps(rec, default=str) + "\n")
    sink.flush()


def train(cfg: PipelineConfig, train_path: str, actor=None, engine=None, traj_out: str | None = None) -> None:
    from transformers import AutoTokenizer

    # mini-swe-agent's own model handles generation/parsing/formatting; we only
    # add the on-policy weight sync. The tokenizer is just for the GRPO adaptor.
    model = build_rollout_model(cfg.rollout)
    tokenizer = AutoTokenizer.from_pretrained(cfg.rollout.model_path, trust_remote_code=True)
    weight_sync = SGLangWeightSync(engine=engine)
    adaptor = TrajectoryAdaptor(tokenizer, max_length=cfg.rollout.context_length)
    instances = load_instances(train_path)
    logger.info("Loaded %d training instances.", len(instances))

    traj_sink = None
    if traj_out:
        os.makedirs(os.path.dirname(traj_out) or ".", exist_ok=True)
        traj_sink = open(traj_out, "w")
        logger.info("Writing per-trajectory records -> %s", traj_out)

    rng_cursor = 0
    for step in range(cfg.grpo.total_steps):
        # ---- 1. strict on-policy weight sync (policy_lag == 0) ----
        actor_weights = actor.named_parameters() if actor is not None else None
        weight_sync.sync(actor_weights)
        if actor is not None:
            weight_sync.assert_on_policy(actor.version)

        # ---- 2. batch of issues ----
        batch = []
        for _ in range(cfg.grpo.train_batch_size):
            batch.append(instances[rng_cursor % len(instances)])
            rng_cursor += 1

        rewards_by_group: dict[str, list[float]] = defaultdict(list)
        adapted_by_group: dict[str, list] = defaultdict(list)
        all_metrics: list[TrajectoryMetrics] = []

        # ---- 3. blocking rollout (trainer idle) then binary reward grading ----
        t_rollout = time.perf_counter()
        rollouts: list[tuple[dict, object]] = []
        for instance in batch:
            for r in rollout_group(model, instance, cfg):
                rollouts.append((instance, r))
        for instance, r in rollouts:
            rr = compute_reward(instance, r.submission)
            apply_to_metrics(r.metrics, rr)
            adapted = adaptor.adapt(r.messages, reward=rr.reward, metrics=r.metrics)
            iid = instance["instance_id"]
            if traj_sink is not None:
                _dump_trajectory(traj_sink, step, len(rewards_by_group[iid]), instance, r, rr)
            rewards_by_group[iid].append(rr.reward)
            adapted_by_group[iid].append(adapted)
            all_metrics.append(r.metrics)
        rollout_s = time.perf_counter() - t_rollout

        # ---- 4. group-relative advantages (critic-less) ----
        group_adv = compute_group_advantages(rewards_by_group)
        training_samples = []
        for instance_id, adv in group_adv.items():
            for adapted, a in zip(adapted_by_group[instance_id], adv.advantages):
                training_samples.append(
                    {
                        "input_ids": adapted.input_ids,
                        "attention_mask": adapted.attention_mask,
                        "position_ids": adapted.position_ids,
                        "response_mask": adapted.response_mask,
                        "token_advantages": broadcast_token_advantages(a, adapted.response_mask),
                    }
                )

        # ---- 5. actor update (GRPO policy gradient) — the only training phase ----
        t_update = time.perf_counter()
        if actor is not None:
            actor.update(
                training_samples,
                ppo_epochs=cfg.grpo.ppo_epochs,
                mini_batch_size=cfg.grpo.ppo_mini_batch_size,
                clip_ratio=cfg.grpo.clip_ratio,
                kl_coef=cfg.grpo.kl_loss_coef if cfg.grpo.use_kl_loss else 0.0,
                loss_agg_mode=cfg.grpo.loss_agg_mode,
            )
        update_s = time.perf_counter() - t_update

        weight_sync.end()

        # ---- 6. metrics ----
        metrics = {
            **grpo_metrics(rewards_by_group),
            **TrajectoryMetrics.aggregate(all_metrics),
            "time/rollout_s": rollout_s,
            "time/update_s": update_s,
            "time/rollout_fraction": rollout_s / max(rollout_s + update_s, 1e-9),
        }
        logger.info(
            "step=%d rollout=%.1fs update=%.1fs rollout_frac=%.2f %s",
            step, rollout_s, update_s, metrics["time/rollout_fraction"],
            {k: round(v, 4) for k, v in metrics.items()},
        )
        _maybe_log_wandb(metrics, step, cfg)

    if traj_sink is not None:
        traj_sink.close()


def _maybe_log_wandb(metrics: dict, step: int, cfg: PipelineConfig) -> None:
    if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true"):
        return
    try:
        import wandb

        if wandb.run is None:
            wandb.init(project=cfg.project_name, name=cfg.experiment_name)
        wandb.log(metrics, step=step)
    except Exception:
        pass


def main() -> None:
    import math

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/swebench_verified/train.parquet")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override grpo.train_batch_size (e.g. 1 for a smoke test).")
    parser.add_argument("--group-size", type=int, default=None,
                        help="Override rollout.group_size (rollouts per issue).")
    parser.add_argument("--total-steps", type=int, default=None,
                        help="Override grpo.total_steps (e.g. 1 for a single-step verify).")
    parser.add_argument("--traj-out", default=None,
                        help="Write per-trajectory JSONL here (messages, submission, reward, metrics).")
    args = parser.parse_args()

    cfg = PipelineConfig()

    # Smoke-test overrides: shrink the batch/group/steps so a single step is cheap.
    if args.batch_size is not None:
        cfg.grpo.train_batch_size = args.batch_size
        cfg.grpo.ppo_mini_batch_size = args.batch_size
    if args.group_size is not None:
        cfg.rollout.group_size = args.group_size
    if args.total_steps is not None:
        cfg.grpo.total_steps = args.total_steps

    # Compute total_steps for 3 epochs if not already set.
    if cfg.grpo.total_steps == 0:
        instances = load_instances(args.train)
        cfg.grpo.total_steps = math.ceil(len(instances) / cfg.grpo.train_batch_size) * cfg.grpo.num_epochs
        logger.info("Computed total_steps=%d (%d instances, batch=%d, epochs=%d)",
                    cfg.grpo.total_steps, len(instances), cfg.grpo.train_batch_size, cfg.grpo.num_epochs)

    logger.warning(
        "Standalone loop: actor backward pass is an integration point "
        "(pass an FSDP/Megatron actor). Running rollout+reward+advantage only."
    )
    train(cfg, args.train, actor=None, traj_out=args.traj_out)


if __name__ == "__main__":
    main()
