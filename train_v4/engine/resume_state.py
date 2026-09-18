"""Sharded optimizer and per-rank random-state checkpoint helpers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import DefaultLoadPlanner
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.optim import Optimizer
from torch.utils.data import DataLoader


RESUME_STATE_DIRNAME = "resume_state"
OPTIMIZER_DIRNAME = "optimizer"


def distributed_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def capture_rng_state(
    device: torch.device,
    loader: DataLoader[Any],
) -> dict[str, Any]:
    """Capture every RNG stream used by the v1 data and loss pipeline."""

    generator_state = None
    if loader.generator is not None:
        generator_state = loader.generator.get_state()
    cuda_state = None
    if device.type == "cuda":
        cuda_state = torch.cuda.get_rng_state(device)
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_state,
        "dataloader_generator": generator_state,
    }


def restore_rng_state(
    state: Mapping[str, Any],
    device: torch.device,
    loader: DataLoader[Any],
) -> None:
    """Restore per-rank RNG state immediately before the next real batch."""

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda" and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state(state["torch_cuda"], device)
    if loader.generator is not None and state.get("dataloader_generator") is not None:
        loader.generator.set_state(state["dataloader_generator"])


def save_resume_state(
    model: nn.Module,
    optimizer: Optimizer,
    checkpoint_root: Path,
    rng_state: Mapping[str, Any],
) -> None:
    """Collectively save sharded optimizer state and this rank's RNG state."""

    resume_root = checkpoint_root / RESUME_STATE_DIRNAME
    resume_root.mkdir(parents=True, exist_ok=True)
    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    optimizer_state = get_optimizer_state_dict(
        model,
        optimizer,
        options=options,
    )
    dcp.save(
        {"optimizer": optimizer_state},
        checkpoint_id=resume_root / OPTIMIZER_DIRNAME,
        no_dist=not (dist.is_available() and dist.is_initialized()),
    )
    torch.save(
        dict(rng_state),
        resume_root / f"rng_rank_{distributed_rank():05d}.pt",
    )


def _materialize_adamw_state(optimizer: Optimizer) -> None:
    """Allocate empty AdamW buffers so DCP has tensors to load into."""

    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("exact resume currently supports AdamW only")
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state[parameter]
            if state:
                continue
            step_device = (
                parameter.device
                if group.get("capturable") or group.get("fused")
                else torch.device("cpu")
            )
            state["step"] = torch.zeros(
                (),
                dtype=torch.float32,
                device=step_device,
            )
            state["exp_avg"] = torch.zeros_like(
                parameter,
                memory_format=torch.preserve_format,
            )
            state["exp_avg_sq"] = torch.zeros_like(
                parameter,
                memory_format=torch.preserve_format,
            )
            if group.get("amsgrad"):
                state["max_exp_avg_sq"] = torch.zeros_like(
                    parameter,
                    memory_format=torch.preserve_format,
                )


def load_resume_state(
    model: nn.Module,
    optimizer: Optimizer,
    checkpoint_root: Path,
) -> dict[str, Any]:
    """Collectively restore optimizer state and return this rank's RNG state."""

    resume_root = checkpoint_root / RESUME_STATE_DIRNAME
    optimizer_root = resume_root / OPTIMIZER_DIRNAME
    if not optimizer_root.is_dir():
        raise FileNotFoundError(
            f"optimizer resume state not found: {optimizer_root}"
        )
    rng_path = resume_root / f"rng_rank_{distributed_rank():05d}.pt"
    if not rng_path.is_file():
        raise FileNotFoundError(f"rank RNG state not found: {rng_path}")

    _materialize_adamw_state(optimizer)
    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    optimizer_state = get_optimizer_state_dict(
        model,
        optimizer,
        options=options,
    )
    container = {"optimizer": optimizer_state}
    # AdamW creates state lazily. Parameters unused before this checkpoint
    # are intentionally absent and retain the zero-valued materialized state.
    dcp.load(
        container,
        checkpoint_id=optimizer_root,
        planner=DefaultLoadPlanner(allow_partial_load=True),
        no_dist=not (dist.is_available() and dist.is_initialized()),
    )
    set_optimizer_state_dict(
        model,
        optimizer,
        container["optimizer"],
        options=options,
    )
    state = torch.load(rng_path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping):
        raise TypeError(f"rank RNG state must be a mapping: {rng_path}")
    return dict(state)
