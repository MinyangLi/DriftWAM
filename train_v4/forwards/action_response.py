"""Single-step frozen action responses to paired teacher-action probes."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from ..config import DistillationConfig
from ..runtime import activate_lingbot_va
from .action_signature import ROBOTWIN_ACTION_CHANNELS, ActionSignatureCondition
from .inference_utils import (
    build_grid_ids,
    frame_positions,
    repeat_candidates,
    select_candidates,
)


# Four BF16 candidates, 16 history + 2 current frames, 120 video + 16 action
# tokens/frame. Longer/larger contexts retain per-candidate recomputation.
_SELECTED_DIRECT_CONTEXT_TOKEN_BUDGET = 4 * (16 + 2) * (120 + 16)


@dataclass(slots=True)
class ActionResponseBatch:
    """Velocity responses produced by all paired teacher-action probes.

    ``teacher_responses`` has shape ``[B,M,C,F,N,1]`` and contains only the
    response of teacher video ``m`` to its own noisy action ``m``.
    ``student_responses`` has shape ``[B,K,M,C,F,N,1]`` because every student
    video is evaluated against every teacher action probe.
    """

    teacher_responses: Tensor
    student_responses: Tensor
    noisy_teacher_actions: Tensor
    action_noise: Tensor
    action_timestep_ids: Tensor


class ActionResponseProbe:
    """Evaluate a frozen action teacher once per teacher-specific action probe."""

    def __init__(
        self,
        teacher: Any,
        config: DistillationConfig,
        *,
        action_channel_ids: Sequence[int] = ROBOTWIN_ACTION_CHANNELS,
        cache_prefix: str = "action_response",
    ) -> None:
        activate_lingbot_va(config.lingbot_va_root)
        from wan_va.utils import FlowMatchScheduler, get_mesh_id

        if any(parameter.requires_grad for parameter in teacher.parameters()):
            raise ValueError("action response teacher parameters must be frozen")
        if config.signature_horizon != 1:
            raise ValueError("action response currently requires signature_horizon=1")

        self.teacher = teacher
        self.config = config
        self.cache_prefix = str(cache_prefix)
        self._get_mesh_id = get_mesh_id
        self._action_channel_ids = tuple(int(index) for index in action_channel_ids)
        self._validate_action_channels()

        self.action_scheduler = FlowMatchScheduler(
            shift=config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.action_scheduler.set_timesteps(
            config.action_num_train_timesteps,
            training=True,
        )

    def compute(
        self,
        teacher_videos: Tensor,
        student_videos: Tensor,
        teacher_actions: Tensor,
        action_noise: Tensor,
        action_timestep_ids: Tensor,
        condition: ActionSignatureCondition,
        *,
        track_student_grad: bool = False,
    ) -> ActionResponseBatch:
        """Return responses using each teacher video's own noisy action.

        ``action_noise`` and ``action_timestep_ids`` are shared across all
        teacher probes and all video candidates within a condition. The clean
        teacher actions differ along ``M`` and are never assigned to students;
        they are only reused as conditional probes.
        """

        self._validate_inputs(
            teacher_videos,
            student_videos,
            teacher_actions,
            action_noise,
            action_timestep_ids,
            condition,
        )
        parameter = next(self.teacher.parameters())
        device = parameter.device
        dtype = getattr(torch, self.config.param_dtype)
        batch_size, teacher_count = teacher_videos.shape[:2]
        student_count = student_videos.shape[1]
        candidate_count = student_count + 1

        teacher_videos = teacher_videos.detach().to(device=device, dtype=dtype)
        if track_student_grad and not torch.is_grad_enabled():
            raise RuntimeError(
                "track_student_grad=True requires autograd to be enabled"
            )
        if not track_student_grad:
            student_videos = student_videos.detach()
        student_videos = student_videos.to(device=device, dtype=dtype)
        teacher_actions = teacher_actions.detach().to(device=device, dtype=dtype)
        action_noise = action_noise.detach().to(device=device, dtype=dtype)

        channel_mask = torch.zeros(
            self.config.action_dim,
            dtype=torch.bool,
            device=device,
        )
        channel_mask[list(self._action_channel_ids)] = True
        channel_mask_5d = channel_mask.view(1, -1, 1, 1, 1)
        teacher_actions = teacher_actions.masked_fill(
            ~channel_mask_5d.unsqueeze(1),
            0,
        )
        action_noise = action_noise.masked_fill(~channel_mask_5d, 0)

        timestep_ids = action_timestep_ids.detach().to(
            device="cpu",
            dtype=torch.long,
        )
        native_timesteps = self.action_scheduler.timesteps.index_select(
            0,
            timestep_ids,
        ).to(device=device, dtype=torch.float32)
        action_timesteps = native_timesteps.unsqueeze(0).expand(
            batch_size,
            -1,
        ).clone()

        frame_starts = frame_positions(
            condition.frame_start,
            batch_size,
            device,
        )
        initial_rows = frame_starts.eq(0)
        if initial_rows.any():
            teacher_actions = teacher_actions.clone()
            teacher_actions[initial_rows, :, :, :1] = 0
            action_timesteps[initial_rows, 0] = 0

        text_emb = repeat_candidates(
            condition.text_emb.to(device=device, dtype=dtype),
            candidate_count,
        )
        repeated_frame_starts = frame_starts.repeat_interleave(candidate_count)
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
            ).masked_fill(~channel_mask_5d, 0)

        repeated_timesteps = repeat_candidates(
            action_timesteps,
            candidate_count,
        )
        shared_cache_name = None
        if history_video is not None and not track_student_grad:
            shared_cache_name = f"{self.cache_prefix}_shared_probes"
            self._create_cache(
                shared_cache_name,
                batch_size=batch_size * candidate_count,
                latent_height=teacher_videos.shape[-2],
                latent_width=teacher_videos.shape[-1],
                device=device,
                dtype=dtype,
                history_frames=history_video.shape[2],
            )
            self._prime_history(
                history_video,
                history_action,
                text_emb,
                history_frame_starts,
                channel_mask_5d,
                shared_cache_name,
            )

        teacher_responses = []
        student_responses = []
        noisy_teacher_actions = []
        for teacher_index in range(teacher_count):
            clean_action = teacher_actions[:, teacher_index]
            noisy_action = self.action_scheduler.add_noise(
                clean_action,
                action_noise,
                native_timesteps,
                t_dim=2,
            ).masked_fill(~channel_mask_5d, 0)
            if initial_rows.any():
                noisy_action = noisy_action.clone()
                noisy_action[initial_rows, :, :1] = 0
            noisy_teacher_actions.append(noisy_action)

            videos = torch.cat(
                [
                    teacher_videos[:, teacher_index : teacher_index + 1],
                    student_videos,
                ],
                dim=1,
            ).reshape(
                batch_size * candidate_count,
                *teacher_videos.shape[2:],
            )
            repeated_noisy_action = repeat_candidates(
                noisy_action,
                candidate_count,
            )
            cache_kwargs = {
                "cache_name": (
                    shared_cache_name
                    or f"{self.cache_prefix}_probe_{teacher_index}"
                )
            }
            if shared_cache_name is not None:
                cache_kwargs["cache_is_primed"] = True
            velocities = self._evaluate_velocity(
                videos,
                repeated_noisy_action,
                repeated_timesteps,
                text_emb,
                repeated_frame_starts,
                history_video,
                history_action,
                history_frame_starts,
                channel_mask_5d,
                **cache_kwargs,
            ).reshape(
                batch_size,
                candidate_count,
                self.config.action_dim,
                teacher_videos.shape[3],
                self.config.action_per_frame,
                1,
            )
            teacher_responses.append(velocities[:, 0].detach())
            student_responses.append(velocities[:, 1:])
            if shared_cache_name is not None:
                self.teacher.clear_pred_cache(shared_cache_name)

        if shared_cache_name is not None:
            self.teacher.clear_cache(shared_cache_name)

        return ActionResponseBatch(
            teacher_responses=torch.stack(teacher_responses, dim=1),
            student_responses=torch.stack(student_responses, dim=2),
            noisy_teacher_actions=torch.stack(noisy_teacher_actions, dim=1),
            action_noise=action_noise,
            action_timestep_ids=timestep_ids,
        )

    def compute_selected_student_responses(
        self,
        student_videos: Tensor,
        noisy_teacher_actions: Tensor,
        action_timestep_ids: Tensor,
        selected_teacher: Tensor,
        condition: ActionSignatureCondition,
    ) -> Tensor:
        """Evaluate only the selected teacher-action probe for each student."""

        if not torch.is_grad_enabled():
            raise RuntimeError(
                "selected student responses require autograd to be enabled"
            )
        if student_videos.ndim != 6:
            raise ValueError("student_videos must have shape [B,K,C,F,H,W]")
        if noisy_teacher_actions.ndim != 6:
            raise ValueError(
                "noisy_teacher_actions must have shape [B,M,C,F,N,1]"
            )
        parameter = next(self.teacher.parameters())
        device = parameter.device
        dtype = getattr(torch, self.config.param_dtype)
        batch_size, student_count = student_videos.shape[:2]
        teacher_batch, teacher_count, channels, frames, positions, trailing = (
            noisy_teacher_actions.shape
        )
        if teacher_batch != batch_size:
            raise ValueError("student and noisy-action batches differ")
        if teacher_count < 1 or student_count < 1:
            raise ValueError(
                "teacher and student candidate axes must be non-empty"
            )
        if tuple(selected_teacher.shape) != (batch_size, student_count):
            raise ValueError("selected_teacher must have shape [B,K]")
        expected_frames = (
            self.config.signature_horizon * self.config.frame_chunk_size
        )
        if (
            channels != self.config.action_dim
            or frames != expected_frames
            or frames != student_videos.shape[3]
            or positions != self.config.action_per_frame
            or trailing != 1
        ):
            raise ValueError("noisy teacher actions have the wrong layout")
        if action_timestep_ids.ndim != 1 or action_timestep_ids.numel() != frames:
            raise ValueError(f"action_timestep_ids must have shape [{frames}]")
        if torch.is_floating_point(action_timestep_ids):
            raise ValueError("action_timestep_ids must use an integer dtype")
        if condition.text_emb.shape[0] != batch_size:
            raise ValueError("condition text batch size does not match videos")
        if (condition.history_video is None) != (
            condition.history_action is None
        ):
            raise ValueError(
                "history_video and history_action must be provided together"
            )
        if condition.history_video is not None:
            if condition.history_video.shape[0] != batch_size:
                raise ValueError("history batch size does not match videos")
            if (
                condition.history_video.shape[2]
                != condition.history_action.shape[2]
            ):
                raise ValueError(
                    "history video and action frame counts differ"
                )

        selected = selected_teacher.detach().to(
            device=device,
            dtype=torch.long,
        )
        if torch.any(selected < 0) or torch.any(selected >= teacher_count):
            raise ValueError("selected_teacher contains an invalid index")
        student_videos = student_videos.to(device=device, dtype=dtype)
        noisy_teacher_actions = noisy_teacher_actions.detach().to(
            device=device,
            dtype=dtype,
        )
        selected_noisy_actions = select_candidates(
            noisy_teacher_actions,
            selected,
        )

        channel_mask = torch.zeros(
            self.config.action_dim,
            dtype=torch.bool,
            device=device,
        )
        channel_mask[list(self._action_channel_ids)] = True
        channel_mask_5d = channel_mask.view(1, -1, 1, 1, 1)
        selected_noisy_actions = selected_noisy_actions.reshape(
            batch_size * student_count,
            channels,
            frames,
            positions,
            trailing,
        ).masked_fill(~channel_mask_5d, 0)

        timestep_ids = action_timestep_ids.detach().to(
            device="cpu",
            dtype=torch.long,
        )
        if torch.any(timestep_ids < 0) or torch.any(
            timestep_ids >= self.config.action_num_train_timesteps
        ):
            raise ValueError("action_timestep_ids contains an invalid index")
        native_timesteps = self.action_scheduler.timesteps.index_select(
            0,
            timestep_ids,
        ).to(device=device, dtype=torch.float32)
        action_timesteps = native_timesteps.unsqueeze(0).expand(
            batch_size * student_count,
            -1,
        ).clone()

        frame_starts = frame_positions(
            condition.frame_start,
            batch_size,
            device,
        ).repeat_interleave(student_count)
        initial_rows = frame_starts.eq(0)
        if initial_rows.any():
            selected_noisy_actions = selected_noisy_actions.clone()
            selected_noisy_actions[initial_rows, :, :1] = 0
            action_timesteps[initial_rows, 0] = 0

        text_emb = repeat_candidates(
            condition.text_emb.to(device=device, dtype=dtype),
            student_count,
        )
        history_frame_starts = frame_positions(
            condition.history_frame_start,
            batch_size,
            device,
        ).repeat_interleave(student_count)
        history_video = None
        history_action = None
        if condition.history_video is not None:
            history_video = repeat_candidates(
                condition.history_video.detach().to(device=device, dtype=dtype),
                student_count,
            )
            history_action = repeat_candidates(
                condition.history_action.detach().to(device=device, dtype=dtype),
                student_count,
            ).masked_fill(~channel_mask_5d, 0)

        flat_videos = student_videos.reshape(
            batch_size * student_count, *student_videos.shape[2:]
        )
        history_frames = 0 if history_video is None else history_video.shape[2]
        patch_volume = math.prod(getattr(self.teacher, "patch_size", (1, 1, 1)))
        context_tokens = flat_videos.shape[0] * (history_frames + frames) * (
            flat_videos.shape[-2] * flat_videos.shape[-1] / patch_volume
            + self.config.action_per_frame
        )
        # Scale the bound for FP32 debug forwards too. This changes only graph
        # storage/recomputation, never candidates, matching, history or losses.
        context_tokens *= flat_videos.element_size() / 2
        if context_tokens <= _SELECTED_DIRECT_CONTEXT_TOKEN_BUDGET:
            velocities = self._evaluate_velocity(
                flat_videos, selected_noisy_actions, action_timesteps,
                text_emb, frame_starts, history_video, history_action,
                history_frame_starts, channel_mask_5d,
                cache_name=f"{self.cache_prefix}_selected_students",
            )
            return velocities.reshape(
                batch_size, student_count, self.config.action_dim,
                frames, self.config.action_per_frame, 1,
            )

        responses = []
        for row in range(flat_videos.shape[0]):
            # Keep only one teacher response graph live during backward.
            # Checkpoint the ENTIRE cache lifecycle: recomputation recreates,
            # primes, consumes and clears its own cache, never a stale one.
            inputs = (
                flat_videos[row:row + 1],
                selected_noisy_actions[row:row + 1],
                action_timesteps[row:row + 1],
                text_emb[row:row + 1],
                frame_starts[row:row + 1],
                None if history_video is None else history_video[row:row + 1],
                None if history_action is None else history_action[row:row + 1],
                history_frame_starts[row:row + 1],
                channel_mask_5d,
            )
            cache_name = f"{self.cache_prefix}_selected_students_{row}"
            if torch.is_grad_enabled() and flat_videos.requires_grad:
                response = checkpoint(
                    self._evaluate_velocity, *inputs, cache_name=cache_name,
                    use_reentrant=False, preserve_rng_state=True,
                )
            else:
                response = self._evaluate_velocity(*inputs, cache_name=cache_name)
            responses.append(response)
        velocities = torch.cat(responses, dim=0)
        return velocities.reshape(
            batch_size,
            student_count,
            self.config.action_dim,
            frames,
            self.config.action_per_frame,
            1,
        )

    __call__ = compute

    def _evaluate_velocity(
        self,
        videos: Tensor,
        noisy_actions: Tensor,
        action_timesteps: Tensor,
        text_emb: Tensor,
        frame_starts: Tensor,
        history_video: Tensor | None,
        history_action: Tensor | None,
        history_frame_starts: Tensor,
        channel_mask: Tensor,
        *,
        cache_name: str,
        cache_is_primed: bool = False,
    ) -> Tensor:
        """Evaluate one current action chunk after priming causal video context."""

        if not cache_is_primed:
            self._create_cache(
                cache_name,
                batch_size=videos.shape[0],
                latent_height=videos.shape[-2],
                latent_width=videos.shape[-1],
                device=videos.device,
                dtype=videos.dtype,
                history_frames=0 if history_video is None else history_video.shape[2],
            )
        try:
            if history_video is not None and not cache_is_primed:
                self._prime_history(
                    history_video,
                    history_action,
                    text_emb,
                    history_frame_starts,
                    channel_mask,
                    cache_name,
                )
            self._forward(
                videos,
                text_emb,
                frame_starts,
                timestep=0.0,
                action_mode=False,
                update_cache=1,
                cache_name=cache_name,
            )
            prediction = self._forward(
                noisy_actions,
                text_emb,
                frame_starts,
                timestep=action_timesteps,
                action_mode=True,
                update_cache=0,
                cache_name=cache_name,
            )
            prediction = prediction.reshape(
                videos.shape[0],
                videos.shape[2],
                self.config.action_per_frame,
                self.config.action_dim,
            )
            return prediction.permute(0, 3, 1, 2).unsqueeze(-1)
        finally:
            if not cache_is_primed:
                self.teacher.clear_cache(cache_name)

    def _create_cache(
        self,
        cache_name: str,
        *,
        batch_size: int,
        latent_height: int,
        latent_width: int,
        device: torch.device,
        dtype: torch.dtype,
        history_frames: int,
    ) -> None:
        patch_f, patch_h, patch_w = tuple(self.teacher.patch_size)
        video_tokens = (
            self.config.frame_chunk_size
            * latent_height
            * latent_width
            // math.prod((patch_f, patch_h, patch_w))
        )
        action_tokens = self.config.frame_chunk_size * self.config.action_per_frame
        history_video_tokens = (
            history_frames
            * latent_height
            * latent_width
            // math.prod((patch_f, patch_h, patch_w))
        )
        history_action_tokens = history_frames * self.config.action_per_frame
        total_tokens = (
            history_video_tokens
            + history_action_tokens
            + video_tokens
            + action_tokens
        )

        self.teacher.clear_cache(cache_name)
        for block in self.teacher.blocks:
            block.attn1.init_kv_cache(
                cache_name,
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
        channel_mask: Tensor,
        cache_name: str,
    ) -> None:
        chunk_size = self.config.frame_chunk_size
        for offset in range(0, history_video.shape[2], chunk_size):
            frame_slice = slice(offset, min(offset + chunk_size, history_video.shape[2]))
            chunk_starts = history_frame_starts + offset
            self._forward(
                history_video[:, :, frame_slice],
                text_emb,
                chunk_starts,
                timestep=0.0,
                action_mode=False,
                update_cache=2,
                cache_name=cache_name,
            )
            action_chunk = history_action[:, :, frame_slice].masked_fill(
                ~channel_mask,
                0,
            )
            self._forward(
                action_chunk,
                text_emb,
                chunk_starts,
                timestep=0.0,
                action_mode=True,
                update_cache=2,
                cache_name=cache_name,
            )

    def _forward(
        self,
        latents: Tensor,
        text_emb: Tensor,
        frame_starts: Tensor,
        *,
        timestep: float | Tensor,
        action_mode: bool,
        update_cache: int,
        cache_name: str,
    ) -> Tensor:
        if isinstance(timestep, Tensor):
            timestep_tensor = timestep
        else:
            timestep_tensor = torch.full(
                (latents.shape[0], latents.shape[2]),
                timestep,
                dtype=torch.float32,
                device=latents.device,
            )
        return self.teacher(
            {
                "noisy_latents": latents,
                "timesteps": timestep_tensor,
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
            cache_name=cache_name,
            action_mode=action_mode,
        )

    def _validate_action_channels(self) -> None:
        if not self._action_channel_ids:
            raise ValueError("action_channel_ids must not be empty")
        if len(set(self._action_channel_ids)) != len(self._action_channel_ids):
            raise ValueError("action_channel_ids must be unique")
        if any(
            index < 0 or index >= self.config.action_dim
            for index in self._action_channel_ids
        ):
            raise ValueError("action_channel_ids contains an invalid channel")

    def _validate_inputs(
        self,
        teacher_videos: Tensor,
        student_videos: Tensor,
        teacher_actions: Tensor,
        action_noise: Tensor,
        action_timestep_ids: Tensor,
        condition: ActionSignatureCondition,
    ) -> None:
        if teacher_videos.ndim != 6 or student_videos.ndim != 6:
            raise ValueError("videos must have shape [B,Q,C,F,H,W]")
        batch_size, teacher_count = teacher_videos.shape[:2]
        if student_videos.shape[0] != batch_size:
            raise ValueError("teacher and student video batch sizes differ")
        if teacher_videos.shape[2:] != student_videos.shape[2:]:
            raise ValueError("teacher and student video layouts differ")
        if teacher_count < 1 or student_videos.shape[1] < 1:
            raise ValueError("teacher and student candidate axes must be non-empty")
        expected_frames = self.config.signature_horizon * self.config.frame_chunk_size
        if teacher_videos.shape[3] != expected_frames:
            raise ValueError(f"videos must contain {expected_frames} frames")
        expected_actions = (
            batch_size,
            teacher_count,
            self.config.action_dim,
            expected_frames,
            self.config.action_per_frame,
            1,
        )
        if tuple(teacher_actions.shape) != expected_actions:
            raise ValueError(f"teacher_actions must have shape {expected_actions}")
        expected_noise = (
            batch_size,
            self.config.action_dim,
            expected_frames,
            self.config.action_per_frame,
            1,
        )
        if tuple(action_noise.shape) != expected_noise:
            raise ValueError(f"action_noise must have shape {expected_noise}")
        if (
            action_timestep_ids.ndim != 1
            or action_timestep_ids.numel() != expected_frames
        ):
            raise ValueError(
                f"action_timestep_ids must have shape [{expected_frames}]"
            )
        if torch.is_floating_point(action_timestep_ids):
            raise ValueError("action_timestep_ids must use an integer dtype")
        if torch.any(action_timestep_ids < 0) or torch.any(
            action_timestep_ids >= self.config.action_num_train_timesteps
        ):
            raise ValueError("action_timestep_ids contains an invalid index")
        if condition.text_emb.shape[0] != batch_size:
            raise ValueError("condition text batch size does not match videos")
        if (condition.history_video is None) != (condition.history_action is None):
            raise ValueError("history_video and history_action must be provided together")
        if condition.history_video is not None:
            if condition.history_video.shape[0] != batch_size:
                raise ValueError("history batch size does not match videos")
            if condition.history_video.shape[2] != condition.history_action.shape[2]:
                raise ValueError("history video and action frame counts differ")
