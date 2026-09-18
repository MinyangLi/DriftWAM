"""Offline multi-step video generation with the frozen LingBot-VA teacher."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from ..training_config import TrainingConfig
from .action_signature import (
    ROBOTWIN_ACTION_CHANNELS,
    ActionSignatureCondition,
)
from .inference_utils import build_grid_ids, frame_positions, repeat_candidates
from ..runtime import activate_lingbot_va


_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class TeacherVideoGenerator:
    """Generate one condition-matched teacher video chunk at a time.

    Every current chunk starts from independent Gaussian noise. Complete
    ground-truth video/action chunks before ``frame_start`` are written into
    the causal cache first; the current ground-truth chunk is never exposed to
    the teacher. Classifier-free guidance follows LingBot-VA's native ordering:
    conditioned rows first, empty-prompt rows second.
    """

    def __init__(
        self,
        teacher: Any,
        config: TrainingConfig,
        negative_text_emb: Tensor,
        *,
        cache_name: str = "teacher_video_bank",
    ) -> None:
        activate_lingbot_va(config.lingbot_va_root)
        from wan_va.utils import FlowMatchScheduler, data_seq_to_patch, get_mesh_id

        if not isinstance(negative_text_emb, Tensor):
            raise TypeError("negative_text_emb must be a tensor")
        if negative_text_emb.ndim == 2:
            negative_text_emb = negative_text_emb.unsqueeze(0)
        if negative_text_emb.ndim != 3:
            raise ValueError(
                "negative_text_emb must have shape [L,D] or [B,L,D]"
            )

        self.teacher = teacher
        self.config = config
        self.negative_text_emb = negative_text_emb.detach().cpu().contiguous()
        self.dtype = _DTYPES[config.param_dtype]
        self.cache_name = cache_name
        self._data_seq_to_patch = data_seq_to_patch
        self._get_mesh_id = get_mesh_id
        self.scheduler = FlowMatchScheduler(
            num_train_timesteps=config.video_num_train_timesteps,
            shift=config.video_snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )

    @torch.no_grad()
    def generate(
        self,
        video_noise: Tensor,
        condition: ActionSignatureCondition,
        *,
        initial_frame: Tensor | None = None,
    ) -> Tensor:
        """Return teacher videos with shape ``[B,K,C,2,H,W]``.

        ``K`` may be smaller than the configured bank candidate count so the
        caller can generate the four stored candidates in memory-safe groups.
        """

        self._validate_inputs(video_noise, condition, initial_frame)
        parameter = next(self.teacher.parameters())
        device = parameter.device
        dtype = self.dtype

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
        positive_text = repeat_candidates(
            condition.text_emb.to(device=device, dtype=dtype),
            candidate_count,
        )
        negative_text = repeat_candidates(
            self._negative_batch(batch_size, device=device, dtype=dtype),
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
        use_cfg = self.config.teacher_video_guidance_scale > 1.0

        self._create_cache(
            batch_size=flat_batch * (2 if use_cfg else 1),
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
                    positive_text,
                    negative_text,
                    history_starts,
                    action_mask,
                    use_cfg=use_cfg,
                )

            initial_rows = frame_starts.eq(0)
            if initial_rows.any():
                videos = videos.clone()
                videos[initial_rows, :, :1] = repeated_initial[initial_rows]

            self.scheduler.set_timesteps(
                self.config.teacher_video_num_inference_steps
            )
            timesteps = torch.cat(
                [
                    self.scheduler.timesteps,
                    torch.zeros(1, dtype=self.scheduler.timesteps.dtype),
                ]
            )
            for step_index, timestep in enumerate(timesteps):
                last_step = step_index == len(timesteps) - 1
                per_frame_timestep = torch.full(
                    (flat_batch, frame_count),
                    float(timestep),
                    dtype=torch.float32,
                    device=device,
                )
                if initial_rows.any():
                    per_frame_timestep[initial_rows, 0] = 0

                prediction = self._forward_cfg(
                    videos,
                    positive_text,
                    negative_text,
                    frame_starts,
                    timestep=per_frame_timestep,
                    action_mode=False,
                    update_cache=1 if last_step else 0,
                    use_cfg=use_cfg,
                )
                if not last_step:
                    model_batch = flat_batch * (2 if use_cfg else 1)
                    prediction = self._data_seq_to_patch(
                        self.teacher.patch_size,
                        prediction,
                        frame_count,
                        latent_height,
                        latent_width,
                        batch_size=model_batch,
                    )
                    if use_cfg:
                        conditioned = prediction[:flat_batch]
                        unconditioned = prediction[flat_batch:]
                        prediction = unconditioned + (
                            self.config.teacher_video_guidance_scale
                            * (conditioned - unconditioned)
                        )
                    else:
                        prediction = prediction[:flat_batch]
                    videos = self.scheduler.step(
                        prediction,
                        timestep,
                        videos,
                    )
                    if initial_rows.any():
                        videos[initial_rows, :, :1] = repeated_initial[
                            initial_rows
                        ]

            return videos.reshape(
                batch_size,
                candidate_count,
                channels,
                frame_count,
                latent_height,
                latent_width,
            )
        finally:
            self.teacher.clear_cache(self.cache_name)

    __call__ = generate

    def _negative_batch(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        negative = self.negative_text_emb.to(device=device, dtype=dtype)
        if negative.shape[0] == 1:
            return negative.expand(batch_size, -1, -1)
        if negative.shape[0] != batch_size:
            raise ValueError(
                "negative embedding batch must be one or match the condition"
            )
        return negative

    def _validate_inputs(
        self,
        video_noise: Tensor,
        condition: ActionSignatureCondition,
        initial_frame: Tensor | None,
    ) -> None:
        if video_noise.ndim != 6:
            raise ValueError("video_noise must have shape [B,K,C,F,H,W]")
        if video_noise.shape[1] < 1:
            raise ValueError("video_noise must contain at least one candidate")
        if video_noise.shape[3] != self.config.frame_chunk_size:
            raise ValueError(
                f"video_noise must contain {self.config.frame_chunk_size} frames"
            )
        if condition.text_emb.ndim != 3:
            raise ValueError("condition text_emb must have shape [B,L,D]")
        if condition.text_emb.shape[0] != video_noise.shape[0]:
            raise ValueError("condition text batch size does not match noise")
        if (condition.history_video is None) != (condition.history_action is None):
            raise ValueError(
                "history_video and history_action must be provided together"
            )

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
                    f"initial_frame must have shape {expected}, got "
                    f"{tuple(initial_frame.shape)}"
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
            raise ValueError("history must consist of complete chunks")
        expected_action = (
            batch_size,
            self.config.action_dim,
            history_video.shape[2],
            self.config.action_per_frame,
            1,
        )
        if tuple(history_action.shape) != expected_action:
            raise ValueError(
                f"history_action must have shape {expected_action}, got "
                f"{tuple(history_action.shape)}"
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
    ) -> None:
        latent_tokens = (
            self.config.frame_chunk_size
            * latent_height
            * latent_width
            // math.prod(self.teacher.patch_size)
        )
        action_tokens = (
            self.config.frame_chunk_size * self.config.action_per_frame
        )
        self.teacher.clear_cache(self.cache_name)
        self.teacher.create_empty_cache(
            self.cache_name,
            self.config.attn_window,
            latent_tokens,
            action_tokens,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
        )

    def _prime_history(
        self,
        history_video: Tensor,
        history_action: Tensor,
        positive_text: Tensor,
        negative_text: Tensor,
        history_starts: Tensor,
        action_mask: Tensor,
        *,
        use_cfg: bool,
    ) -> None:
        chunk_size = self.config.frame_chunk_size
        for offset in range(0, history_video.shape[2], chunk_size):
            frame_slice = slice(offset, offset + chunk_size)
            chunk_starts = history_starts + offset
            self._forward_cfg(
                history_video[:, :, frame_slice],
                positive_text,
                negative_text,
                chunk_starts,
                timestep=0.0,
                action_mode=False,
                update_cache=2,
                use_cfg=use_cfg,
            )
            action_chunk = history_action[:, :, frame_slice]
            self._forward_cfg(
                action_chunk.masked_fill(~action_mask, 0),
                positive_text,
                negative_text,
                chunk_starts,
                timestep=0.0,
                action_mode=True,
                update_cache=2,
                use_cfg=use_cfg,
            )

    def _forward_cfg(
        self,
        latents: Tensor,
        positive_text: Tensor,
        negative_text: Tensor,
        frame_starts: Tensor,
        *,
        timestep: float | Tensor,
        action_mode: bool,
        update_cache: int,
        use_cfg: bool,
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

        if use_cfg:
            model_latents = torch.cat([latents, latents], dim=0)
            model_text = torch.cat([positive_text, negative_text], dim=0)
            model_starts = torch.cat([frame_starts, frame_starts], dim=0)
            timestep_tensor = torch.cat(
                [timestep_tensor, timestep_tensor],
                dim=0,
            )
        else:
            model_latents = latents
            model_text = positive_text
            model_starts = frame_starts

        return self.teacher(
            {
                "noisy_latents": model_latents,
                "timesteps": timestep_tensor,
                "grid_id": build_grid_ids(
                    model_latents,
                    model_starts,
                    action_mode=action_mode,
                    patch_size=self.teacher.patch_size,
                    get_mesh_id=self._get_mesh_id,
                ),
                "text_emb": model_text,
            },
            update_cache=update_cache,
            cache_name=self.cache_name,
            action_mode=action_mode,
        )
