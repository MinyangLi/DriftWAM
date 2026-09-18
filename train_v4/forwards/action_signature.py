"""Causal action signatures for generated video candidates."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ..config import DistillationConfig
from .inference_utils import build_grid_ids, frame_positions, repeat_candidates
from ..runtime import activate_lingbot_va


ROBOTWIN_ACTION_CHANNELS = tuple(range(14)) + (28, 29)


@dataclass(slots=True)
class ActionSignatureCondition:
    """Causally available context for the current generated chunk.

    ``frame_start`` is the latent-frame index of the first generated frame.
    History tensors contain only already observed video and executed actions.
    """

    text_emb: Tensor
    frame_start: int | Tensor = 0
    history_video: Tensor | None = None
    history_action: Tensor | None = None
    history_frame_start: int | Tensor = 0


class ActionSignatureGenerator:
    """Run the frozen LingBot-VA action teacher on supplied video candidates."""

    def __init__(
        self,
        teacher: Any,
        config: DistillationConfig,
        *,
        action_channel_ids: Sequence[int] = ROBOTWIN_ACTION_CHANNELS,
        cache_name: str = "action_signature",
    ) -> None:
        activate_lingbot_va(config.lingbot_va_root)
        from wan_va.utils import FlowMatchScheduler, get_mesh_id

        self.teacher = teacher
        self.config = config
        self.cache_name = cache_name
        self._get_mesh_id = get_mesh_id
        self._action_channel_ids = tuple(int(index) for index in action_channel_ids)
        if any(
            index < 0 or index >= config.action_dim
            for index in self._action_channel_ids
        ):
            raise ValueError("action_channel_ids contains an invalid channel")

        self.action_scheduler = FlowMatchScheduler(
            shift=config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )

    @torch.no_grad()
    def generate(
        self,
        video_candidates: Tensor,
        action_noise: Tensor,
        condition: ActionSignatureCondition,
    ) -> Tensor:
        """Generate one causal action signature for every video candidate.

        ``video_candidates`` has shape ``[B,Q,48,H*2,24,20]``. ``B`` is the
        condition count, ``Q`` is the candidate count, and ``H`` is the
        configured signature horizon in chunks. ``action_noise`` has shape
        ``[B,30,H*2,16,1]`` and is broadcast across ``Q`` so candidates under
        one condition share exactly the same noise realization.
        """

        self._validate_inputs(video_candidates, action_noise, condition)

        model_parameter = next(self.teacher.parameters())
        device = model_parameter.device
        dtype = getattr(torch, self.config.param_dtype)

        batch_size, candidate_count = video_candidates.shape[:2]
        flat_batch = batch_size * candidate_count
        frame_count = video_candidates.shape[3]
        latent_height, latent_width = video_candidates.shape[-2:]

        videos = video_candidates.to(device=device, dtype=dtype)
        videos = videos.reshape(
            flat_batch,
            videos.shape[2],
            frame_count,
            latent_height,
            latent_width,
        )
        shared_noise = repeat_candidates(
            action_noise.to(device=device, dtype=dtype),
            candidate_count,
        ).clone()
        text_emb = repeat_candidates(
            condition.text_emb.to(device=device, dtype=dtype),
            candidate_count,
        )

        frame_starts = frame_positions(
            condition.frame_start,
            batch_size,
            device,
        ).repeat_interleave(candidate_count)
        history_frame_starts = frame_positions(
            condition.history_frame_start,
            batch_size,
            device,
        ).repeat_interleave(candidate_count)

        history_video = None
        history_action = None
        if condition.history_video is not None:
            history_video = repeat_candidates(
                condition.history_video.to(device=device, dtype=dtype),
                candidate_count,
            )
            history_action = repeat_candidates(
                condition.history_action.to(device=device, dtype=dtype),
                candidate_count,
            )

        action_mask = torch.zeros(
            self.config.action_dim,
            dtype=torch.bool,
            device=device,
        )
        action_mask[list(self._action_channel_ids)] = True
        action_mask_5d = action_mask.view(1, -1, 1, 1, 1)

        self._create_cache(
            batch_size=flat_batch,
            latent_height=latent_height,
            latent_width=latent_width,
            device=device,
            dtype=dtype,
        )

        try:
            if history_video is not None:
                self._prime_history(
                    history_video,
                    history_action,
                    text_emb,
                    history_frame_starts,
                    action_mask_5d,
                )

            action_chunks = []
            chunk_size = self.config.frame_chunk_size
            for chunk_index in range(self.config.signature_horizon):
                frame_slice = slice(
                    chunk_index * chunk_size,
                    (chunk_index + 1) * chunk_size,
                )
                chunk_starts = frame_starts + chunk_index * chunk_size

                self._forward(
                    videos[:, :, frame_slice],
                    text_emb,
                    chunk_starts,
                    timestep=0.0,
                    action_mode=False,
                    update_cache=1,
                )

                actions = shared_noise[:, :, frame_slice].clone()
                actions.masked_fill_(~action_mask_5d, 0)
                initial_rows = chunk_starts.eq(0)
                if initial_rows.any():
                    actions[initial_rows, :, :1] = 0

                actions = self._denoise_action_chunk(
                    actions,
                    text_emb,
                    chunk_starts,
                    initial_rows,
                    action_mask_5d,
                    cache_clean_action=(
                        chunk_index + 1 < self.config.signature_horizon
                    ),
                )
                action_chunks.append(actions)

            signatures = torch.cat(action_chunks, dim=2)
            return signatures.reshape(
                batch_size,
                candidate_count,
                self.config.action_dim,
                frame_count,
                self.config.action_per_frame,
                1,
            )
        finally:
            self.teacher.clear_cache(self.cache_name)

    __call__ = generate

    def _validate_inputs(
        self,
        video_candidates: Tensor,
        action_noise: Tensor,
        condition: ActionSignatureCondition,
    ) -> None:
        expected_frames = (
            self.config.signature_horizon * self.config.frame_chunk_size
        )
        if video_candidates.ndim != 6:
            raise ValueError("video_candidates must have shape [B,Q,C,F,H,W]")
        if action_noise.ndim != 5:
            raise ValueError("action_noise must have shape [B,C,F,N,1]")
        if video_candidates.shape[0] != action_noise.shape[0]:
            raise ValueError("video_candidates and action_noise batch sizes differ")
        if video_candidates.shape[1] < 1:
            raise ValueError("video_candidates must contain at least one candidate")
        if video_candidates.shape[3] != expected_frames:
            raise ValueError(f"video_candidates must contain {expected_frames} frames")
        expected_action_shape = (
            video_candidates.shape[0],
            self.config.action_dim,
            expected_frames,
            self.config.action_per_frame,
            1,
        )
        if tuple(action_noise.shape) != expected_action_shape:
            raise ValueError(
                f"action_noise must have shape {expected_action_shape}, "
                f"got {tuple(action_noise.shape)}"
            )
        if condition.text_emb.shape[0] != video_candidates.shape[0]:
            raise ValueError("condition text batch size does not match videos")
        if (condition.history_video is None) != (condition.history_action is None):
            raise ValueError("history_video and history_action must be provided together")
        if condition.history_video is not None:
            if condition.history_video.shape[0] != video_candidates.shape[0]:
                raise ValueError("history batch size does not match videos")
            if condition.history_video.shape[2] != condition.history_action.shape[2]:
                raise ValueError("history video and action frame counts differ")

    def _create_cache(
        self,
        *,
        batch_size: int,
        latent_height: int,
        latent_width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        patch_f, patch_h, patch_w = tuple(self.teacher.patch_size)
        latent_tokens_per_chunk = (
            self.config.frame_chunk_size
            * latent_height
            * latent_width
            // math.prod((patch_f, patch_h, patch_w))
        )
        action_tokens_per_chunk = (
            self.config.frame_chunk_size * self.config.action_per_frame
        )
        self.teacher.clear_cache(self.cache_name)
        self.teacher.create_empty_cache(
            self.cache_name,
            self.config.attn_window,
            latent_tokens_per_chunk,
            action_tokens_per_chunk,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
        )

    def _prime_history(
        self,
        history_video: Tensor,
        history_action: Tensor,
        text_emb: Tensor,
        history_frame_starts: Tensor,
        action_mask: Tensor,
    ) -> None:
        chunk_size = self.config.frame_chunk_size
        history_frames = history_video.shape[2]
        for offset in range(0, history_frames, chunk_size):
            frame_slice = slice(offset, min(offset + chunk_size, history_frames))
            chunk_starts = history_frame_starts + offset
            self._forward(
                history_video[:, :, frame_slice],
                text_emb,
                chunk_starts,
                timestep=0.0,
                action_mode=False,
                update_cache=2,
            )
            action_chunk = history_action[:, :, frame_slice]
            action_chunk = action_chunk.masked_fill(~action_mask, 0)
            self._forward(
                action_chunk,
                text_emb,
                chunk_starts,
                timestep=0.0,
                action_mode=True,
                update_cache=2,
            )

    def _denoise_action_chunk(
        self,
        actions: Tensor,
        text_emb: Tensor,
        frame_starts: Tensor,
        initial_rows: Tensor,
        action_mask: Tensor,
        *,
        cache_clean_action: bool,
    ) -> Tensor:
        self.action_scheduler.set_timesteps(
            self.config.action_teacher_num_inference_steps
        )
        for timestep in self.action_scheduler.timesteps:
            per_frame_timestep = torch.full(
                (actions.shape[0], actions.shape[2]),
                float(timestep),
                dtype=torch.float32,
                device=actions.device,
            )
            if initial_rows.any():
                per_frame_timestep[initial_rows, 0] = 0

            prediction = self._forward(
                actions,
                text_emb,
                frame_starts,
                timestep=per_frame_timestep,
                action_mode=True,
                update_cache=0,
            )
            prediction = prediction.reshape(
                actions.shape[0],
                actions.shape[2],
                self.config.action_per_frame,
                self.config.action_dim,
            )
            prediction = prediction.permute(0, 3, 1, 2).unsqueeze(-1)
            actions = self.action_scheduler.step(
                prediction,
                timestep,
                actions,
            )
            actions.masked_fill_(~action_mask, 0)
            if initial_rows.any():
                actions[initial_rows, :, :1] = 0

        if cache_clean_action:
            self._forward(
                actions,
                text_emb,
                frame_starts,
                timestep=0.0,
                action_mode=True,
                update_cache=1,
            )
        return actions

    def _forward(
        self,
        latents: Tensor,
        text_emb: Tensor,
        frame_starts: Tensor,
        *,
        timestep: float | Tensor,
        action_mode: bool,
        update_cache: int,
    ) -> Tensor:
        frame_count = latents.shape[2]
        if isinstance(timestep, Tensor):
            timestep_tensor = timestep
        else:
            timestep_tensor = torch.full(
                (latents.shape[0], frame_count),
                timestep,
                dtype=torch.float32,
                device=latents.device,
            )

        grid_id = build_grid_ids(
            latents,
            frame_starts,
            action_mode=action_mode,
            patch_size=self.teacher.patch_size,
            get_mesh_id=self._get_mesh_id,
        )
        return self.teacher(
            {
                "noisy_latents": latents,
                "timesteps": timestep_tensor,
                "grid_id": grid_id,
                "text_emb": text_emb,
            },
            update_cache=update_cache,
            cache_name=self.cache_name,
            action_mode=action_mode,
        )
