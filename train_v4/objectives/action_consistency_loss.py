"""Action consistency distillation on aligned ground-truth pairs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ..forwards.action_signature import (
    ROBOTWIN_ACTION_CHANNELS,
    ActionSignatureCondition,
)
from .action_training_pair import ActionTrainingPair
from ..config import DistillationConfig
from ..forwards.inference_utils import build_grid_ids, frame_positions
from ..runtime import activate_lingbot_va


@dataclass(slots=True)
class ActionConsistencyLossResult:
    """Action consistency, flow-matching loss, and sampled noise levels."""

    consistency_loss: Tensor
    flow_matching_loss: Tensor
    timestep_ids: Tensor
    sigma_start: Tensor
    sigma_end: Tensor


@dataclass(slots=True)
class _PackedActionCondition:
    video: Tensor
    clean_action: Tensor
    video_grid_id: Tensor
    action_grid_id: Tensor
    text_emb: Tensor
    history_frames: int


class ActionConsistencyLoss:
    """Distill a two-point action consistency pair from teacher to student."""

    def __init__(
        self,
        teacher: Any,
        online_student: Any,
        target_student: Any,
        config: DistillationConfig,
        *,
        action_channel_ids: Sequence[int] = ROBOTWIN_ACTION_CHANNELS,
    ) -> None:
        activate_lingbot_va(config.lingbot_va_root)
        from wan_va.utils import FlowMatchScheduler, get_mesh_id

        if any(parameter.requires_grad for parameter in teacher.parameters()):
            raise ValueError("action consistency teacher must be frozen")
        if any(parameter.requires_grad for parameter in target_student.parameters()):
            raise ValueError("action consistency target student must be frozen")
        if not any(
            parameter.requires_grad for parameter in online_student.parameters()
        ):
            raise ValueError("online student must contain trainable parameters")

        self.teacher = teacher
        self.online_student = online_student
        self.target_student = target_student
        self.config = config
        self._get_mesh_id = get_mesh_id
        self._action_channel_ids = tuple(int(index) for index in action_channel_ids)
        if len(self._action_channel_ids) != 16:
            raise ValueError("action_channel_ids must contain 16 channels")
        if len(set(self._action_channel_ids)) != 16:
            raise ValueError("action_channel_ids must be unique")
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
        self.action_scheduler.set_timesteps(
            config.action_num_train_timesteps,
            training=True,
        )

    def compute(
        self,
        pair: ActionTrainingPair,
        condition: ActionSignatureCondition,
    ) -> ActionConsistencyLossResult:
        """Compute masked ``x0`` consistency and velocity regularization.

        The GT pair and all causal history are detached inputs. The
        returned losses update online-student parameters but never propagate
        into the video-generation or action-signature graphs.
        """

        self._validate_pair(pair, condition)
        online_parameter = next(self.online_student.parameters())
        device = online_parameter.device
        dtype = getattr(torch, self.config.param_dtype)

        batch_size = pair.video.shape[0]
        current_frames = pair.video.shape[2]
        current_video = pair.video.detach().to(device=device, dtype=dtype)
        # Official DataMixin constructs action noise and FM targets in FP32.
        # Packed model forwards cast their private inputs to the compute dtype.
        clean_action = pair.clean_action.detach().to(
            device=device,
            dtype=torch.float32,
        ).clone()
        action_mask = pair.action_mask.detach().to(
            device=device,
            dtype=torch.bool,
        ).clone()
        clean_action.masked_fill_(~action_mask, 0)

        timestep_ids = torch.randint(
            self.config.action_num_train_timesteps,
            (current_frames,),
            device="cpu",
        )
        end_ids = (
            timestep_ids + self.config.action_consistency_stride
        ).clamp(max=self.config.action_num_train_timesteps - 1)

        base_sigma_start = self.action_scheduler.sigmas.index_select(
            0,
            timestep_ids,
        ).to(device=device, dtype=torch.float32)
        base_sigma_end = self.action_scheduler.sigmas.index_select(
            0,
            end_ids,
        ).to(device=device, dtype=torch.float32)
        sigma_start = base_sigma_start.unsqueeze(0).expand(
            batch_size,
            -1,
        ).clone()
        sigma_end = base_sigma_end.unsqueeze(0).expand(
            batch_size,
            -1,
        ).clone()

        start_timesteps = self.action_scheduler.timesteps.index_select(
            0,
            timestep_ids,
        ).to(device=device, dtype=torch.float32)
        end_timesteps = self.action_scheduler.timesteps.index_select(
            0,
            end_ids,
        ).to(device=device, dtype=torch.float32)
        start_timesteps = start_timesteps.unsqueeze(0).expand(
            batch_size,
            -1,
        ).clone()
        end_timesteps = end_timesteps.unsqueeze(0).expand(
            batch_size,
            -1,
        ).clone()

        action_noise = torch.randn_like(clean_action)
        action_noise.masked_fill_(~action_mask, 0)
        sigma_start_5d = sigma_start[:, None, :, None, None].to(clean_action)
        noisy_action = (
            (1.0 - sigma_start_5d) * clean_action
            + sigma_start_5d * action_noise
        )
        noisy_action.masked_fill_(~action_mask, 0)

        packed = self._pack_condition(
            current_video,
            clean_action,
            condition,
            device=device,
            dtype=dtype,
        )
        start_input = self._model_input(
            packed,
            noisy_action,
            start_timesteps,
        )
        with torch.no_grad():
            teacher_velocity = self._forward_current_action(
                self.teacher,
                start_input,
                current_frames,
            )
        del start_input
        torch.cuda.empty_cache()

        # Match Flash-WAM's velocity-dtype Euler increment. Do not mask the
        # intermediate state: inactive channels can affect EMA's valid outputs.
        teacher_sigma_start = sigma_start[:, None, :, None, None].to(teacher_velocity)
        teacher_sigma_end = sigma_end[:, None, :, None, None].to(teacher_velocity)
        end_action = (
            noisy_action
            + teacher_velocity * (teacher_sigma_end - teacher_sigma_start)
        ).detach()

        with torch.no_grad():
            target_velocity = self._forward_current_action(
                self.target_student,
                self._model_input(packed, end_action, end_timesteps),
                current_frames,
            )
            target_velocity.masked_fill_(~action_mask, 0)
        torch.cuda.empty_cache()

        online_velocity = self._forward_current_action(
            self.online_student,
            self._model_input(packed, noisy_action, start_timesteps),
            current_frames,
        )
        online_velocity = online_velocity.masked_fill(~action_mask, 0)
        online_sigma = sigma_start[:, None, :, None, None].to(online_velocity)
        target_sigma = sigma_end[:, None, :, None, None].to(target_velocity)
        online_x0 = noisy_action - online_sigma * online_velocity
        target_x0 = end_action - target_sigma * target_velocity
        consistency_loss = self._masked_pseudo_huber(
            online_x0,
            target_x0.detach(),
            action_mask,
        )

        flow_matching_target = action_noise - clean_action
        flow_matching_loss = self._masked_mse(
            online_velocity,
            flow_matching_target.detach(),
            action_mask,
        )

        return ActionConsistencyLossResult(
            consistency_loss=consistency_loss,
            flow_matching_loss=flow_matching_loss,
            timestep_ids=timestep_ids.detach(),
            sigma_start=sigma_start.detach(),
            sigma_end=sigma_end.detach(),
        )

    __call__ = compute

    def _pack_condition(
        self,
        current_video: Tensor,
        clean_action: Tensor,
        condition: ActionSignatureCondition,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> _PackedActionCondition:
        batch_size = current_video.shape[0]
        current_starts = frame_positions(
            condition.frame_start,
            batch_size,
            device,
        )
        current_video_grid = build_grid_ids(
            current_video,
            current_starts,
            action_mode=False,
            patch_size=self.online_student.patch_size,
            get_mesh_id=self._get_mesh_id,
        )
        current_action_grid = build_grid_ids(
            clean_action,
            current_starts,
            action_mode=True,
            patch_size=self.online_student.patch_size,
            get_mesh_id=self._get_mesh_id,
        )

        history_frames = 0
        full_video = current_video
        full_clean_action = clean_action
        video_grid_id = current_video_grid
        action_grid_id = current_action_grid
        if condition.history_video is not None:
            history_video = condition.history_video.detach().to(
                device=device,
                dtype=dtype,
            )
            history_action = condition.history_action.detach().to(
                device=device,
                dtype=dtype,
            )
            history_action = self._mask_action_channels(history_action)
            history_frames = history_video.shape[2]
            history_starts = frame_positions(
                condition.history_frame_start,
                batch_size,
                device,
            )
            history_video_grid = build_grid_ids(
                history_video,
                history_starts,
                action_mode=False,
                patch_size=self.online_student.patch_size,
                get_mesh_id=self._get_mesh_id,
            )
            history_action_grid = build_grid_ids(
                history_action,
                history_starts,
                action_mode=True,
                patch_size=self.online_student.patch_size,
                get_mesh_id=self._get_mesh_id,
            )
            full_video = torch.cat([history_video, current_video], dim=2)
            full_clean_action = torch.cat(
                [history_action, clean_action],
                dim=2,
            )
            video_grid_id = torch.cat(
                [history_video_grid, current_video_grid],
                dim=-1,
            )
            action_grid_id = torch.cat(
                [history_action_grid, current_action_grid],
                dim=-1,
            )

        return _PackedActionCondition(
            video=full_video,
            clean_action=full_clean_action,
            video_grid_id=video_grid_id,
            action_grid_id=action_grid_id,
            text_emb=condition.text_emb.detach().to(
                device=device,
                dtype=dtype,
            ),
            history_frames=history_frames,
        )

    def _model_input(
        self,
        packed: _PackedActionCondition,
        current_query_action: Tensor,
        current_timesteps: Tensor,
    ) -> dict[str, Any]:
        batch_size = packed.video.shape[0]
        history_frames = packed.history_frames
        if history_frames:
            history_action = packed.clean_action[:, :, :history_frames]
            query_action = torch.cat(
                [history_action, current_query_action],
                dim=2,
            )
            history_timesteps = torch.zeros(
                batch_size,
                history_frames,
                dtype=torch.float32,
                device=packed.video.device,
            )
            action_timesteps = torch.cat(
                [history_timesteps, current_timesteps],
                dim=1,
            )
        else:
            query_action = current_query_action
            action_timesteps = current_timesteps

        video_timesteps = torch.zeros(
            batch_size,
            packed.video.shape[2],
            dtype=torch.float32,
            device=packed.video.device,
        )
        action_condition_timesteps = torch.zeros_like(action_timesteps)
        return {
            "latent_dict": {
                "noisy_latents": packed.video,
                "latent": packed.video,
                "timesteps": video_timesteps,
                "cond_timesteps": video_timesteps,
                "grid_id": packed.video_grid_id,
                "text_emb": packed.text_emb,
            },
            "action_dict": {
                "noisy_latents": query_action,
                "latent": packed.clean_action,
                "timesteps": action_timesteps,
                "cond_timesteps": action_condition_timesteps,
                "grid_id": packed.action_grid_id,
                "text_emb": packed.text_emb,
            },
            "chunk_size": self.config.frame_chunk_size,
            "window_size": self.config.attn_window,
        }

    def _forward_current_action(
        self,
        model: Any,
        input_dict: dict[str, Any],
        current_frames: int,
    ) -> Tensor:
        _, action_prediction = model(input_dict, train_mode=True)
        batch_size = input_dict["action_dict"]["noisy_latents"].shape[0]
        total_frames = input_dict["action_dict"]["noisy_latents"].shape[2]
        action_velocity = action_prediction.reshape(
            batch_size,
            total_frames,
            self.config.action_per_frame,
            self.config.action_dim,
        )
        action_velocity = action_velocity.permute(0, 3, 1, 2).unsqueeze(-1)
        return action_velocity[:, :, -current_frames:]

    def _mask_action_channels(self, action: Tensor) -> Tensor:
        channel_mask = torch.zeros(
            self.config.action_dim,
            dtype=torch.bool,
            device=action.device,
        )
        channel_mask[list(self._action_channel_ids)] = True
        return action.masked_fill(~channel_mask.view(1, -1, 1, 1, 1), 0)

    def _masked_pseudo_huber(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor,
    ) -> Tensor:
        difference = prediction.float() - target.float()
        c = self.config.action_huber_c
        values = torch.sqrt(difference.square() + c * c) - c
        return self._masked_mean(values, mask)

    @classmethod
    def _masked_mse(
        cls,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor,
    ) -> Tensor:
        return cls._masked_mean(
            (prediction.float() - target.float()).square(),
            mask,
        )

    @staticmethod
    def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
        weights = mask.float()
        denominator = weights.sum()
        if denominator.item() == 0:
            raise ValueError("action consistency requires a valid action element")
        return (values * weights).sum() / denominator

    def _validate_pair(
        self,
        pair: ActionTrainingPair,
        condition: ActionSignatureCondition,
    ) -> None:
        if pair.video.ndim != 5:
            raise ValueError("pair.video must have shape [B,C,F,H,W]")
        if pair.clean_action.ndim != 5:
            raise ValueError(
                "pair.clean_action must have shape [B,C,F,N,1]"
            )
        if tuple(pair.action_mask.shape) != tuple(pair.clean_action.shape):
            raise ValueError("pair.action_mask must match pair.clean_action")
        if pair.action_mask.dtype != torch.bool:
            raise ValueError("pair.action_mask must use a boolean dtype")
        if pair.video.shape[0] != pair.clean_action.shape[0]:
            raise ValueError("pair video/action batch sizes differ")
        if pair.video.shape[2] != pair.clean_action.shape[2]:
            raise ValueError("pair video/action frame counts differ")
        expected_frames = pair.video.shape[2]
        if expected_frames < 1:
            raise ValueError("action training requires a nonempty full sequence")
        expected_action_tail = (
            self.config.action_dim,
            expected_frames,
            self.config.action_per_frame,
            1,
        )
        if tuple(pair.clean_action.shape[1:]) != expected_action_tail:
            raise ValueError(
                f"pair.clean_action must have trailing shape {expected_action_tail}"
            )
        if condition.text_emb.shape[0] != pair.video.shape[0]:
            raise ValueError("condition text batch size does not match pair")
        if (condition.history_video is None) != (condition.history_action is None):
            raise ValueError(
                "history_video and history_action must be provided together"
            )
        if condition.history_video is not None:
            if condition.history_video.ndim != 5:
                raise ValueError("history_video must have shape [B,C,F,H,W]")
            if condition.history_action.ndim != 5:
                raise ValueError("history_action must have shape [B,C,F,N,1]")
            if condition.history_video.shape[0] != pair.video.shape[0]:
                raise ValueError("history batch size does not match pair")
            if condition.history_video.shape[1] != pair.video.shape[1]:
                raise ValueError("history/current video channels differ")
            if condition.history_video.shape[3:] != pair.video.shape[3:]:
                raise ValueError("history/current video spatial layouts differ")
            if condition.history_action.shape[1:] != (
                self.config.action_dim,
                condition.history_video.shape[2],
                self.config.action_per_frame,
                1,
            ):
                raise ValueError("history action layout does not match history video")
            if condition.history_video.shape[2] % self.config.frame_chunk_size:
                raise ValueError("history frames must contain complete chunks")

        model_parameters = (
            next(self.teacher.parameters()),
            next(self.online_student.parameters()),
            next(self.target_student.parameters()),
        )
        model_device = model_parameters[1].device
        if any(parameter.device != model_device for parameter in model_parameters):
            raise ValueError("teacher, online student, and target must share a device")
