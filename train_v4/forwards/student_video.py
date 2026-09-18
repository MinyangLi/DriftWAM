"""Differentiable online generation of one-step student videos."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from .action_signature import (
    ROBOTWIN_ACTION_CHANNELS,
    ActionSignatureCondition,
)
from ..config import DistillationConfig
from .inference_utils import build_grid_ids, frame_positions, repeat_candidates
from ..runtime import activate_lingbot_va


class StudentVideoGenerator:
    """Generate causal video candidates with the trainable one-step student."""

    def __init__(
        self,
        student: Any,
        config: DistillationConfig,
        *,
        cache_name: str = "student_video",
    ) -> None:
        activate_lingbot_va(config.lingbot_va_root)
        from wan_va.utils import FlowMatchScheduler, data_seq_to_patch, get_mesh_id

        self.student = student
        self.config = config
        self.cache_name = cache_name
        self._data_seq_to_patch = data_seq_to_patch
        self._get_mesh_id = get_mesh_id
        self.scheduler = FlowMatchScheduler(
            num_train_timesteps=config.video_num_train_timesteps,
            shift=config.video_snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.scheduler.set_timesteps(config.video_student_num_inference_steps)

    def generate(
        self,
        video_noise: Tensor,
        condition: ActionSignatureCondition,
        *,
        initial_frame: Tensor | None = None,
    ) -> Tensor:
        """Return online student videos with shape ``[B,K,C,F,H,W]``.

        ``video_noise`` supplies one independent Gaussian realization for
        every candidate. ``initial_frame`` has shape ``[B,C,1,H,W]`` and is
        required when the current chunk starts at frame zero. The result stays
        connected to the student parameters.
        """

        self._validate_inputs(video_noise, condition, initial_frame)
        parameter = next(self.student.parameters())
        device = parameter.device
        dtype = getattr(torch, self.config.param_dtype)

        batch_size, candidate_count, channels, frame_count = video_noise.shape[:4]
        latent_height, latent_width = video_noise.shape[-2:]
        flat_batch = batch_size * candidate_count
        videos = video_noise.to(device=device, dtype=dtype).reshape(
            flat_batch,
            channels,
            frame_count,
            latent_height,
            latent_width,
        )
        text_emb = repeat_candidates(
            condition.text_emb.to(device=device, dtype=dtype),
            candidate_count,
        )
        frame_starts = frame_positions(
            condition.frame_start,
            batch_size,
            device,
        ).repeat_interleave(candidate_count)
        history_starts = frame_positions(
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

        repeated_initial = None
        if initial_frame is not None:
            repeated_initial = repeat_candidates(
                initial_frame.to(device=device, dtype=dtype),
                candidate_count,
            )

        action_mask = torch.zeros(
            self.config.action_dim,
            dtype=torch.bool,
            device=device,
        )
        action_mask[list(ROBOTWIN_ACTION_CHANNELS)] = True
        action_mask = action_mask.view(1, -1, 1, 1, 1)

        self._create_cache(
            batch_size=flat_batch,
            latent_height=latent_height,
            latent_width=latent_width,
            device=device,
            dtype=dtype,
            history_frames=(
                0 if history_video is None else history_video.shape[2]
            ),
        )
        try:
            if history_video is not None:
                with torch.no_grad():
                    self._prime_history(
                        history_video,
                        history_action,
                        text_emb,
                        history_starts,
                        action_mask,
                    )

            initial_rows = frame_starts.eq(0)
            if initial_rows.any():
                videos = videos.clone()
                videos[initial_rows, :, :1] = repeated_initial[initial_rows]

            timestep = self.scheduler.timesteps[0]
            timesteps = torch.full(
                (flat_batch, frame_count),
                float(timestep),
                dtype=torch.float32,
                device=device,
            )
            if initial_rows.any():
                timesteps[initial_rows, 0] = 0

            velocity_sequence = self.student(
                {
                    "noisy_latents": videos,
                    "timesteps": timesteps,
                    "grid_id": build_grid_ids(
                        videos,
                        frame_starts,
                        action_mode=False,
                        patch_size=self.student.patch_size,
                        get_mesh_id=self._get_mesh_id,
                    ),
                    "text_emb": text_emb,
                },
                update_cache=0,
                cache_name=self.cache_name,
                action_mode=False,
            )
            velocity = self._data_seq_to_patch(
                self.student.patch_size,
                velocity_sequence,
                frame_count,
                latent_height,
                latent_width,
                batch_size=flat_batch,
            )
            generated = self.scheduler.step(
                velocity,
                timestep,
                videos,
                return_dict=False,
            )
            if initial_rows.any():
                generated = generated.clone()
                generated[initial_rows, :, :1] = repeated_initial[initial_rows]

            return generated.reshape(
                batch_size,
                candidate_count,
                channels,
                frame_count,
                latent_height,
                latent_width,
            )
        finally:
            self.student.clear_cache(self.cache_name)

    __call__ = generate

    def _validate_inputs(
        self,
        video_noise: Tensor,
        condition: ActionSignatureCondition,
        initial_frame: Tensor | None,
    ) -> None:
        if video_noise.ndim != 6:
            raise ValueError("video_noise must have shape [B,K,C,F,H,W]")
        candidate_count = video_noise.shape[1]
        if not 1 <= candidate_count <= self.config.student_candidate_count:
            raise ValueError(
                "video_noise candidate count must lie between 1 and the "
                "configured student candidate count"
            )
        if video_noise.shape[3] != self.config.frame_chunk_size:
            raise ValueError(
                f"video_noise must contain {self.config.frame_chunk_size} frames"
            )
        if condition.text_emb.shape[0] != video_noise.shape[0]:
            raise ValueError("condition text batch size does not match video noise")
        if (condition.history_video is None) != (condition.history_action is None):
            raise ValueError("history_video and history_action must be provided together")

        batch_size, _, channels, _, height, width = video_noise.shape
        frame_starts = frame_positions(
            condition.frame_start,
            batch_size,
            video_noise.device,
        )
        if (frame_starts < 0).any():
            raise ValueError("frame_start must be non-negative")
        if frame_starts.eq(0).any() and initial_frame is None:
            raise ValueError("initial_frame is required when frame_start is zero")
        if frame_starts.gt(0).any() and condition.history_video is None:
            raise ValueError("causal history is required after frame zero")
        if initial_frame is not None:
            expected = (batch_size, channels, 1, height, width)
            if tuple(initial_frame.shape) != expected:
                raise ValueError(
                    f"initial_frame must have shape {expected}, "
                    f"got {tuple(initial_frame.shape)}"
                )

        if condition.history_video is None:
            return

        history_video = condition.history_video
        history_action = condition.history_action
        if history_video.ndim != 5 or history_action.ndim != 5:
            raise ValueError("history tensors must have shape [B,C,F,H,W]")
        if tuple(history_video.shape[:2]) != (batch_size, channels):
            raise ValueError("history video batch or channel count differs")
        if history_video.shape[-2:] != (height, width):
            raise ValueError("history video spatial shape differs")
        if history_video.shape[2] != history_action.shape[2]:
            raise ValueError("history video and action frame counts differ")
        if history_video.shape[2] % self.config.frame_chunk_size != 0:
            raise ValueError("history frame count must contain complete chunks")
        expected_action = (
            batch_size,
            self.config.action_dim,
            history_video.shape[2],
            self.config.action_per_frame,
            1,
        )
        if tuple(history_action.shape) != expected_action:
            raise ValueError(
                f"history_action must have shape {expected_action}, "
                f"got {tuple(history_action.shape)}"
            )
        history_starts = frame_positions(
            condition.history_frame_start,
            batch_size,
            video_noise.device,
        )
        if not torch.equal(
            history_starts + history_video.shape[2],
            frame_starts,
        ):
            raise ValueError("history must end at the current frame_start")

    def _create_cache(
        self,
        *,
        batch_size: int,
        latent_height: int,
        latent_width: int,
        device: torch.device,
        dtype: torch.dtype,
        history_frames: int,
    ) -> None:
        history_video_tokens = (
            history_frames
            * latent_height
            * latent_width
            // math.prod(self.student.patch_size)
        )
        history_action_tokens = history_frames * self.config.action_per_frame
        total_tokens = history_video_tokens + history_action_tokens

        self.student.clear_cache(self.cache_name)
        for block in self.student.blocks:
            block.attn1.init_kv_cache(
                self.cache_name,
                total_tokens,
                self.student.num_attention_heads,
                self.student.attention_head_dim,
                device,
                dtype,
                batch_size,
            )

    def _prime_history(
        self,
        history_video: Tensor,
        history_action: Tensor,
        text_emb: Tensor,
        history_starts: Tensor,
        action_mask: Tensor,
    ) -> None:
        chunk_size = self.config.frame_chunk_size
        for offset in range(0, history_video.shape[2], chunk_size):
            frame_slice = slice(offset, offset + chunk_size)
            chunk_starts = history_starts + offset
            self._forward_history(
                history_video[:, :, frame_slice],
                text_emb,
                chunk_starts,
                action_mode=False,
            )
            action_chunk = history_action[:, :, frame_slice]
            self._forward_history(
                action_chunk.masked_fill(~action_mask, 0),
                text_emb,
                chunk_starts,
                action_mode=True,
            )

    def _forward_history(
        self,
        latents: Tensor,
        text_emb: Tensor,
        frame_starts: Tensor,
        *,
        action_mode: bool,
    ) -> None:
        self.student(
            {
                "noisy_latents": latents,
                "timesteps": torch.zeros(
                    (latents.shape[0], latents.shape[2]),
                    dtype=torch.float32,
                    device=latents.device,
                ),
                "grid_id": build_grid_ids(
                    latents,
                    frame_starts,
                    action_mode=action_mode,
                    patch_size=self.student.patch_size,
                    get_mesh_id=self._get_mesh_id,
                ),
                "text_emb": text_emb,
            },
            update_cache=2,
            cache_name=self.cache_name,
            action_mode=action_mode,
        )
