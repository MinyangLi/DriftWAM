"""Exponential moving average update for the complete target student."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _is_dtensor(tensor: Tensor) -> bool:
    return type(tensor).__name__ == "DTensor"


@torch.no_grad()
def update_ema(
    target_model: nn.Module,
    source_model: nn.Module,
    *,
    decay: float = 0.995,
) -> None:
    """Move every target parameter toward the matching online parameter.

    The target and source must use the same parameter names and the same
    regular-tensor or DTensor layout. Call this once after a successful joint
    optimizer step, never once per loss or gradient-accumulation microbatch.
    """

    decay = float(decay)
    if not math.isfinite(decay) or not 0 <= decay < 1:
        raise ValueError("EMA decay must be finite and lie in [0, 1)")

    target_parameters = dict(target_model.named_parameters())
    source_parameters = dict(source_model.named_parameters())
    if target_parameters.keys() != source_parameters.keys():
        raise ValueError("EMA target and source parameter names differ")
    if any(parameter.requires_grad for parameter in target_parameters.values()):
        raise ValueError("EMA target parameters must be frozen")

    source_weight = 1.0 - decay
    for name, target_parameter in target_parameters.items():
        source_parameter = source_parameters[name]
        target_value = target_parameter.data
        source_value = source_parameter.data
        if target_value.dtype != torch.float32 or source_value.dtype != torch.float32:
            raise ValueError(f"EMA parameter {name!r} requires FP32 source and target")
        target_is_dtensor = _is_dtensor(target_value)
        source_is_dtensor = _is_dtensor(source_value)
        if target_is_dtensor != source_is_dtensor:
            raise ValueError(
                f"EMA parameter {name!r} uses different distributed layouts"
            )

        if target_is_dtensor:
            target_local = target_value._local_tensor
            source_local = source_value._local_tensor
            if target_local.shape != source_local.shape:
                raise ValueError(f"EMA parameter {name!r} local shapes differ")
            target_local.lerp_(source_local.to(target_local), source_weight)
        else:
            if target_value.shape != source_value.shape:
                raise ValueError(f"EMA parameter {name!r} shapes differ")
            target_value.lerp_(source_value.to(target_value), source_weight)
