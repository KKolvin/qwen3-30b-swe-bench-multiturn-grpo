"""FSDP → SGLang weight sync for ``moe_runner_backend: flashinfer_trtllm``.

Why this exists
---------------
Qwen3-30B-A3B at TP=4 has ``moe_intermediate_size / tp = 192``. The
``flashinfer_trtllm`` kernel wants that dim divisible by 128, so SGLang pads
it to 256 and, for the bf16 path, permutes + block-layouts each expert in
``UnquantizedFusedMoEMethod.process_weights_after_loading``.

The first load from disk does both steps (``load_weights`` pads/swaps w1↔w3,
then postprocess permutes). verl's hybrid naive sync does not: every
``update_weights_from_tensor`` is ``model.load_weights`` only, and the live
Parameter is already in the kernel layout, so the FSDP tensor (192)
``copy_``-fails.

This module is the custom loader SGLang's TP workers call instead:

1. Reshape each MoE Parameter back to the padded logical layout (same
   storage, so this is free). ``FusedMoE.weight_loader`` can then pad 192→256
   and swap the w13 halves into it.
2. After the last bucket, re-run ``process_weights_after_loading`` so the
   kernel sees the permuted block layout again. That hook is installed on
   ``Scheduler.flush_cache``, which verl already calls once the bucket loop
   finishes. Intermediate buckets set ``flush_cache=False`` so we do not
   permute after a partial write (the permute is not idempotent).

Wire-up is two pieces, both required:

* ``engine_kwargs.sglang.custom_weight_loader`` registers this FQN so
  ``load_format`` resolves inside the TP workers.
* :func:`patch_sglang_weight_sync` (called from the trainer, same process
  that constructs ``SGLangReplica``) makes every naive
  ``update_weights_from_tensor`` request carry that FQN and suppress the
  per-bucket cache flush.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from typing import Any

import torch

logger = logging.getLogger("agentic_grpo.flashinfer_moe_loader")

LOADER_FQN = "agentic_grpo.flashinfer_moe_loader.load_weights"

_PENDING_ATTR = "_agentic_flashinfer_moe_pending"
_FLUSH_PATCHED = False
_LAUNCH_PATCHED = False


def load_weights(model: torch.nn.Module, named_tensors: Iterable[tuple[str, torch.Tensor]]) -> None:
    """SGLang custom-weight-loader entry: pad/swap FSDP tensors into logical MoE layout."""
    _patch_scheduler_flush_cache()
    _restore_logical_moe_layout(model)
    model.load_weights(named_tensors)
    setattr(model, _PENDING_ATTR, True)


def _logical_moe_shapes(layer: torch.nn.Module) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Padded ``create_weights`` shapes, before ``process_weights_after_loading``."""
    inter = int(layer.intermediate_size_per_partition)
    hidden = int(layer.hidden_size)
    n_experts = int(layer.num_local_experts)
    w13_n = 2 * inter if getattr(layer.moe_runner_config, "is_gated", True) else inter
    return (n_experts, w13_n, hidden), (n_experts, hidden, inter)


def _restore_logical_moe_layout(model: torch.nn.Module) -> None:
    """View kernel-layout MoE Parameters as the padded HF layout ``weight_loader`` writes."""
    for module in model.modules():
        if not getattr(module, "use_flashinfer_trtllm_moe", False):
            continue
        w13 = getattr(module, "w13_weight", None)
        w2 = getattr(module, "w2_weight", None)
        if w13 is None or w2 is None:
            continue
        w13_shape, w2_shape = _logical_moe_shapes(module)
        if tuple(w13.shape) != w13_shape:
            if w13.numel() != math.prod(w13_shape):
                raise RuntimeError(
                    f"flashinfer MoE w13 numel {w13.numel()} cannot restore {w13_shape} "
                    f"from {tuple(w13.shape)}"
                )
            w13.data = w13.data.reshape(*w13_shape)
        if tuple(w2.shape) != w2_shape:
            if w2.numel() != math.prod(w2_shape):
                raise RuntimeError(
                    f"flashinfer MoE w2 numel {w2.numel()} cannot restore {w2_shape} "
                    f"from {tuple(w2.shape)}"
                )
            w2.data = w2.data.reshape(*w2_shape)


def _finalize_moe_layout(model: torch.nn.Module) -> None:
    if not getattr(model, _PENDING_ATTR, False):
        return
    n_moe = 0
    for module in model.modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is None:
            continue
        quant_method.process_weights_after_loading(module)
        if getattr(module, "use_flashinfer_trtllm_moe", False):
            n_moe += 1
    setattr(model, _PENDING_ATTR, False)
    logger.warning(
        "flashinfer_moe_loader: re-packed %d MoE layers after FSDP weight sync",
        n_moe,
    )


def _patch_scheduler_flush_cache() -> None:
    """Run the permute/block-layout step once, on the post-sync ``flush_cache``."""
    global _FLUSH_PATCHED
    if _FLUSH_PATCHED:
        return
    from sglang.srt.managers.scheduler import Scheduler

    original = Scheduler.flush_cache

    def flush_cache(self, *args, **kwargs):
        worker = getattr(self, "draft_worker", None) or self.tp_worker
        model = getattr(getattr(worker, "model_runner", None), "model", None)
        if model is not None:
            _finalize_moe_layout(model)
        return original(self, *args, **kwargs)

    Scheduler.flush_cache = flush_cache
    _FLUSH_PATCHED = True


def patch_sglang_weight_sync() -> bool:
    """Ensure the pickled server class rewires naive FSDP→SGLang updates.

    ``TimedSGLangHttpServer.launch_server`` calls :func:`rewire_tokenizer_manager`
    — that override has to live on the actor class because Ray pickles it by
    value. This installs that class even when request timing is disabled.
    Harmless if :func:`agentic_grpo.sglang_timing.patch_verl_rollout_timing`
    already did the same swap.
    """
    global _LAUNCH_PATCHED
    if _LAUNCH_PATCHED:
        return True
    try:
        import ray
        from verl.workers.rollout.sglang_rollout import async_sglang_server as mod
        from agentic_grpo.sglang_timing import _build_timed_server_class

        timed_cls = _build_timed_server_class()
        original_init = mod.SGLangReplica.__init__

        def __init__(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.server_class = ray.remote(timed_cls)

        mod.SGLangReplica.__init__ = __init__
        _LAUNCH_PATCHED = True
        logger.info("flashinfer_moe_loader: SGLangReplica will rewire updates through %s", LOADER_FQN)
        return True
    except Exception:  # noqa: BLE001 - a missing verl must not kill import
        logger.warning("flashinfer_moe_loader: could not patch SGLangReplica", exc_info=True)
        return False


def rewire_tokenizer_manager(server: Any) -> None:
    tm = getattr(server, "tokenizer_manager", None)
    if tm is None:
        return
    loaders = list(getattr(getattr(tm, "server_args", None), "custom_weight_loader", None) or [])
    if LOADER_FQN not in loaders:
        return
    original = tm.update_weights_from_tensor

    async def update_weights_from_tensor(obj, request=None):
        # Per-bucket flush would permute after a partial write. verl already
        # calls flush_cache once the whole model has been streamed.
        obj.load_format = LOADER_FQN
        obj.flush_cache = False
        return await original(obj, request)

    tm.update_weights_from_tensor = update_weights_from_tensor
    logger.warning("flashinfer_moe_loader: tokenizer_manager rewired for padded MoE sync")
