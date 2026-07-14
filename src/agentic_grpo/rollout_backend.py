"""SGLang rollout backend (SKILL 2.2) - thin glue, not a reimplementation.

We deliberately write as little code as possible here, because the two systems
we already depend on cover almost everything:

* **mini-swe-agent** already owns generation (querying through ``litellm``),
  action parsing (it registers a ``BASH_TOOL`` and uses tool-calling) and
  observation formatting (``observation_template``).
* **SGLang** already exposes an OpenAI-compatible server *and* native
  weight-update endpoints.

So this module is just:

1. :func:`build_rollout_model` - hand back mini-swe-agent's own ``LitellmModel``
   pointed at the local SGLang OpenAI endpoint. No custom chat templating,
   action regex, truncation, or observation formatting.
2. :class:`SGLangWeightSync` - the one RL-specific thing neither library does:
   push the freshly-updated actor weights into the running engine *before* each
   GRPO rollout so ``policy_lag`` stays exactly 0.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from agentic_grpo.config import RolloutConfig

logger = logging.getLogger("agentic_grpo.rollout_backend")


def build_rollout_model(cfg: RolloutConfig, extra_model_kwargs: dict | None = None) -> Any:
    """Return mini-swe-agent's model, talking to the SGLang OpenAI server.

    SGLang's server (``python -m sglang.launch_server ... --port 30000``) speaks
    the OpenAI API, so litellm reaches it with ``custom_llm_provider="openai"``
    and ``api_base``. mini-swe-agent then does all the agent-side work itself.

    We don't stream to measure TTFT client-side: the server publishes TTFT,
    inter-token and e2e latency on its ``/metrics`` endpoint (see
    :mod:`agentic_grpo.server_monitor`), which is ground truth and adds no
    rollout-hot-path overhead. Token usage is read from the response
    mini-swe-agent already builds.
    """
    from minisweagent.models import get_model  # type: ignore

    model_kwargs = {
        "api_base": cfg.base_url,
        "api_key": "EMPTY",
        "custom_llm_provider": "openai",
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "max_tokens": cfg.max_new_tokens,
    }
    if extra_model_kwargs:
        model_kwargs.update(extra_model_kwargs)

    return get_model(
        config={
            "model_name": f"openai/{cfg.model_path}",
            "model_kwargs": model_kwargs,
            # Self-hosted: don't fail rollouts on missing cost metadata.
            "cost_tracking": "ignore_errors",
        }
    )


class SGLangWeightSync:
    """Keep the rollout engine strictly on-policy (policy_lag == 0).

    Call :meth:`sync` at the start of every GRPO step, *before* any generation.

    * verl path: pass verl's ``sharding_manager``; entering it performs the
      (possibly resharded) actor -> SGLang weight transfer.
    * standalone path: pass a local ``sglang.Engine`` and stream named tensors
      via its ``update_weights_from_tensor`` API.
    """

    def __init__(self, engine: Any | None = None, sharding_manager: Any | None = None):
        self.engine = engine
        self.sharding_manager = sharding_manager
        self.version = 0  # bumped each sync; compared against the actor's version

    def sync(self, named_tensors: Iterable[tuple[str, Any]] | None = None) -> None:
        if self.sharding_manager is not None:
            self.sharding_manager.__enter__()
        elif self.engine is not None and named_tensors is not None:
            self.engine.update_weights_from_tensor(list(named_tensors))
        else:
            logger.debug("SGLangWeightSync.sync no-op (no engine / no manager).")
        self.version += 1

    def end(self) -> None:
        """Release the rollout engine back to the trainer (verl path)."""
        if self.sharding_manager is not None:
            self.sharding_manager.__exit__(None, None, None)

    def assert_on_policy(self, actor_version: int) -> None:
        if self.version != actor_version:
            raise RuntimeError(
                "policy_lag violation: rollout weight version "
                f"{self.version} != actor version {actor_version}. "
                "This pipeline requires strictly synchronous (policy_lag=0) training."
            )
