"""Shared tensor helpers for causal LingBot-VA inference."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor


def repeat_candidates(tensor: Tensor, candidate_count: int) -> Tensor:
    expanded = tensor[:, None].expand(
        tensor.shape[0], candidate_count, *tensor.shape[1:]
    )
    return expanded.reshape(
        tensor.shape[0] * candidate_count, *tensor.shape[1:]
    ).contiguous()


def select_candidates(tensor: Tensor, indices: Tensor) -> Tensor:
    """Select one or more candidate-axis entries independently per batch row."""

    if tensor.ndim < 2:
        raise ValueError("candidate tensor must have shape [B,Q,...]")
    if indices.ndim not in (1, 2) or indices.shape[0] != tensor.shape[0]:
        raise ValueError("indices must have shape [B] or [B,S]")

    squeeze_selection = indices.ndim == 1
    if squeeze_selection:
        indices = indices.unsqueeze(1)
    indices = indices.to(device=tensor.device, dtype=torch.long)
    index = indices.view(
        *indices.shape,
        *([1] * (tensor.ndim - 2)),
    ).expand(
        *indices.shape,
        *tensor.shape[2:],
    )
    selected = torch.gather(tensor, dim=1, index=index)
    return selected[:, 0] if squeeze_selection else selected


def frame_positions(
    value: int | Tensor,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    if isinstance(value, int):
        return torch.full(
            (batch_size,), value, dtype=torch.long, device=device
        )
    positions = value.to(device=device, dtype=torch.long).reshape(-1)
    if positions.numel() == 1:
        return positions.expand(batch_size)
    if positions.numel() != batch_size:
        raise ValueError("frame positions must be scalar or have shape [B]")
    return positions


def build_grid_ids(
    latents: Tensor,
    frame_starts: Tensor,
    *,
    action_mode: bool,
    patch_size: Sequence[int],
    get_mesh_id: Callable[..., Tensor],
) -> Tensor:
    if action_mode:
        grid_shape = latents.shape[-3:]
    else:
        patch_f, patch_h, patch_w = tuple(patch_size)
        grid_shape = (
            latents.shape[-3] // patch_f,
            latents.shape[-2] // patch_h,
            latents.shape[-1] // patch_w,
        )
    grids = [
        get_mesh_id(
            *grid_shape,
            1 if action_mode else 0,
            f_w=1,
            f_shift=int(frame_start),
            action=action_mode,
        )
        for frame_start in frame_starts.tolist()
    ]
    return torch.stack(grids, dim=0).to(latents.device)
