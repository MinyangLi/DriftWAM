"""Native LingBot-VA video features for generated candidates."""

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


class _FeatureLayerReached(Exception):
    """Internal signal used to stop the frozen teacher after the target block."""


class VideoFeatureExtractor:
    """Extract conditional post-block video features from the frozen teacher."""

    def __init__(
        self,
        teacher: Any,
        config: DistillationConfig,
        *,
        feature_layer: int = 19,
        noise_coefficient: float = 0.1,
        num_train_timesteps: int = 1000,
        cache_name: str = "video_feature",
    ) -> None:
        activate_lingbot_va(config.lingbot_va_root)
        from wan_va.utils import get_mesh_id

        if not 1 <= feature_layer <= len(teacher.blocks):
            raise ValueError(
                f"feature_layer must be in [1, {len(teacher.blocks)}]"
            )
        if not 0.0 <= noise_coefficient <= 1.0:
            raise ValueError("noise_coefficient must be in [0, 1]")

        self.teacher = teacher
        self.config = config
        self.feature_layer = feature_layer
        self.noise_coefficient = noise_coefficient
        self.feature_timestep = noise_coefficient * num_train_timesteps
        self.cache_name = cache_name
        self._get_mesh_id = get_mesh_id

    def extract(
        self,
        video_candidates: Tensor,
        feature_noise: Tensor,
        condition: ActionSignatureCondition,
    ) -> Tensor:
        """Return layer features with shape ``[B,Q,F,N,D]``.

        ``B`` is the condition count, ``Q`` is the candidate count, ``F`` is
        the latent-frame count, ``N`` is the spatial-token count per frame,
        and ``D`` is the hidden width. ``feature_noise`` has no candidate axis,
        so every candidate under one condition receives the same realization.
        """

        self._validate_inputs(video_candidates, feature_noise, condition)

        model_parameter = next(self.teacher.parameters())
        device = model_parameter.device
        dtype = getattr(torch, self.config.param_dtype)

        batch_size, candidate_count, channels, frame_count = (
            video_candidates.shape[:4]
        )
        latent_height, latent_width = video_candidates.shape[-2:]
        flat_batch = batch_size * candidate_count

        videos = video_candidates.to(device=device, dtype=dtype).reshape(
            flat_batch,
            channels,
            frame_count,
            latent_height,
            latent_width,
        )
        shared_noise = repeat_candidates(
            feature_noise.to(device=device, dtype=dtype),
            candidate_count,
        )
        noisy_videos = (
            (1.0 - self.noise_coefficient) * videos
            + self.noise_coefficient * shared_noise
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
                        history_frame_starts,
                        action_mask,
                    )

            hidden = self._forward(
                noisy_videos,
                text_emb,
                frame_starts,
                timestep=self.feature_timestep,
                action_mode=False,
                update_cache=0,
            )
        finally:
            self.teacher.clear_cache(self.cache_name)

        patch_f, patch_h, patch_w = tuple(self.teacher.patch_size)
        feature_frames = frame_count // patch_f
        tokens_per_frame = (
            (latent_height // patch_h) * (latent_width // patch_w)
        )
        return hidden.reshape(
            batch_size,
            candidate_count,
            feature_frames,
            tokens_per_frame,
            hidden.shape[-1],
        )

    __call__ = extract

    def _validate_inputs(
        self,
        video_candidates: Tensor,
        feature_noise: Tensor,
        condition: ActionSignatureCondition,
    ) -> None:
        if video_candidates.ndim != 6:
            raise ValueError("video_candidates must have shape [B,Q,C,F,H,W]")
        expected_noise_shape = (
            video_candidates.shape[0],
            *video_candidates.shape[2:],
        )
        if tuple(feature_noise.shape) != expected_noise_shape:
            raise ValueError(
                f"feature_noise must have shape {expected_noise_shape}, "
                f"got {tuple(feature_noise.shape)}"
            )
        if video_candidates.shape[1] < 1:
            raise ValueError("video_candidates must contain at least one candidate")
        if video_candidates.shape[3] != self.config.frame_chunk_size:
            raise ValueError(
                f"video feature input must contain "
                f"{self.config.frame_chunk_size} latent frames"
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
        history_frames: int,
    ) -> None:
        patch_f, patch_h, patch_w = tuple(self.teacher.patch_size)
        history_video_tokens = (
            history_frames
            * latent_height
            * latent_width
            // math.prod((patch_f, patch_h, patch_w))
        )
        history_action_tokens = history_frames * self.config.action_per_frame
        total_tokens = history_video_tokens + history_action_tokens
        self.teacher.clear_cache(self.cache_name)
        for block in self.teacher.blocks[: self.feature_layer]:
            block.attn1.init_kv_cache(
                self.cache_name,
                total_tokens,
                self.teacher.num_attention_heads,
                self.teacher.attention_head_dim,
                device,
                dtype,
                batch_size,
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

    def _forward(
        self,
        latents: Tensor,
        text_emb: Tensor,
        frame_starts: Tensor,
        *,
        timestep: float,
        action_mode: bool,
        update_cache: int,
    ) -> Tensor:
        timesteps = torch.full(
            (latents.shape[0], latents.shape[2]),
            timestep,
            dtype=torch.float32,
            device=latents.device,
        )
        captured: list[Tensor] = []

        def stop_after_feature_layer(
            _module: torch.nn.Module,
            _inputs: tuple[Any, ...],
            output: Tensor,
        ) -> None:
            captured.append(output)
            raise _FeatureLayerReached

        block = self.teacher.blocks[self.feature_layer - 1]
        handle = block.register_forward_hook(stop_after_feature_layer)
        try:
            try:
                self.teacher(
                    {
                        "noisy_latents": latents,
                        "timesteps": timesteps,
                        "grid_id": build_grid_ids(
                            latents,
                            frame_starts,
                            action_mode=action_mode,
                            patch_size=self.teacher.patch_size,
                            get_mesh_id=self._get_mesh_id,
                        ),
                        "text_emb": text_emb,
                    },
                    update_cache=update_cache,
                    cache_name=self.cache_name,
                    action_mode=action_mode,
                )
            except _FeatureLayerReached:
                pass
        finally:
            handle.remove()

        if len(captured) != 1:
            raise RuntimeError(
                f"expected one layer-{self.feature_layer} activation, "
                f"captured {len(captured)}"
            )
        return captured[0]
