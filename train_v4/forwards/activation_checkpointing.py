"""Activation recomputation for stateless packed training forwards only."""

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.utils.checkpoint import checkpoint


def _checkpoint_packed_forward(function, *args, **kwargs):
    # Native packed training always supplies an explicit attention mask.
    # Cached inference may mutate/read KV caches that are cleared before
    # backward, so those calls must never be replayed by checkpointing.
    if kwargs.get("attention_mask") is None or not torch.is_grad_enabled():
        return function(*args, **kwargs)
    return checkpoint(
        function, *args, use_reentrant=False, preserve_rng_state=True, **kwargs
    )


def apply_packed_activation_checkpointing(model):
    """Wrap online and EMA blocks identically before FSDP sharding.

    EMA runs without gradients and bypasses recomputation. Matching wrapper
    structure preserves the parameter names used by the existing EMA update;
    PyTorch's wrapper hooks preserve the original serialized weight names.
    """
    for index, block in enumerate(model.blocks):
        model.blocks[index] = checkpoint_wrapper(
            block, checkpoint_fn=_checkpoint_packed_forward
        )
    return model
