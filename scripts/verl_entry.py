#!/usr/bin/env python
"""verl launch entrypoint that makes our metrics reach W&B.

Why this file exists
--------------------
verl runs the trainer inside a ``TaskRunner`` **Ray actor** (see
``verl.trainer.main_ppo.run_ppo``): ``RayPPOTrainer.fit()`` — and therefore
``compute_data_metrics`` and ``logger.log`` — executes in *that* actor process.

Our W&B metrics are injected by monkeypatching ``compute_data_metrics``, and that
patch is applied as an import side-effect of ``agentic_grpo.agent_loop``. But
verl only imports that module in the ``AgentLoopWorker`` actors (a *different*
process), so the patch never lands in the TaskRunner and the metrics silently
never get logged.

Fix: subclass ``TaskRunner`` so it imports ``agentic_grpo.agent_loop`` (thus
applies the patch) **inside the actor**, before ``fit()`` runs. Everything else
is stock verl.

Run via ``scripts/run_grpo.sh`` (which passes the usual hydra overrides).
"""

from __future__ import annotations

import hydra
import ray
from verl.trainer.main_ppo import TaskRunner, run_ppo


class _PatchedTaskRunner(TaskRunner):
    def run(self, config):
        # Import inside the actor process so agent_loop._patch_verl_data_metrics()
        # patches THIS process's verl.trainer.ppo.ray_trainer.compute_data_metrics,
        # the one fit() actually calls.
        import agentic_grpo.agent_loop  # noqa: F401  (import applies the patch)

        return super().run(config)


@hydra.main(config_path="../configs", config_name="grpo_swebench", version_base=None)
def main(config):
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(_PatchedTaskRunner))


if __name__ == "__main__":
    main()
