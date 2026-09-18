"""Hard-matched action-velocity preservation for student videos."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from ..forwards.action_signature import ROBOTWIN_ACTION_CHANNELS


@dataclass(slots=True)
class ActionExecutionLossResult:
    """Execution loss, per-student errors, and selected teachers."""

    loss: Tensor
    per_candidate_loss: Tensor
    selected_teacher: Tensor


class ActionExecutionLoss:
    """Match selected student responses to paired teacher responses."""

    def __init__(
        self,
        action_dim: int = 30,
        *,
        action_channel_ids: Sequence[int] = ROBOTWIN_ACTION_CHANNELS,
    ) -> None:
        self.action_dim = int(action_dim)
        self.action_channel_ids = tuple(
            int(index) for index in action_channel_ids
        )
        if self.action_dim < 1:
            raise ValueError("action_dim must be positive")
        if not self.action_channel_ids:
            raise ValueError("action_channel_ids must not be empty")
        if len(set(self.action_channel_ids)) != len(
            self.action_channel_ids
        ):
            raise ValueError("action_channel_ids must be unique")
        if any(
            index < 0 or index >= self.action_dim
            for index in self.action_channel_ids
        ):
            raise ValueError("action_channel_ids contains an invalid channel")

    @staticmethod
    def select_teachers(positive_weights: Tensor) -> Tensor:
        """Return the detached maximum-weight teacher index for each student."""

        if positive_weights.ndim != 3:
            raise ValueError("positive_weights must have shape [B,K,M]")
        if positive_weights.shape[2] < 1:
            raise ValueError("positive_weights must contain a teacher")
        if not torch.isfinite(positive_weights).all():
            raise ValueError("positive_weights must be finite")
        return positive_weights.detach().argmax(dim=-1)

    @staticmethod
    def select_student_responses(
        student_responses: Tensor,
        selected_teacher: Tensor,
    ) -> Tensor:
        """Gather one teacher-probe response for every student candidate."""

        if student_responses.ndim != 7:
            raise ValueError(
                "student_responses must have shape [B,K,M,C,F,N,1]"
            )
        batch_size, student_count, teacher_count = student_responses.shape[:3]
        if tuple(selected_teacher.shape) != (batch_size, student_count):
            raise ValueError("selected_teacher must have shape [B,K]")
        selected = selected_teacher.to(
            device=student_responses.device,
            dtype=torch.long,
        )
        if torch.any(selected < 0) or torch.any(selected >= teacher_count):
            raise ValueError("selected_teacher contains an invalid index")
        batch_indices = torch.arange(
            batch_size,
            device=student_responses.device,
        )[:, None]
        student_indices = torch.arange(
            student_count,
            device=student_responses.device,
        )[None, :]
        return student_responses[batch_indices, student_indices, selected]

    def compute(
        self,
        teacher_responses: Tensor,
        selected_student_responses: Tensor,
        selected_teacher: Tensor,
        valid_mask: Tensor | None = None,
    ) -> ActionExecutionLossResult:
        """Compute valid-channel velocity MSE for selected teacher pairs."""

        self._validate_inputs(
            teacher_responses,
            selected_student_responses,
            selected_teacher,
        )
        batch_size, teacher_count = teacher_responses.shape[:2]
        selected = selected_teacher.to(
            device=teacher_responses.device,
            dtype=torch.long,
        )
        if torch.any(selected < 0) or torch.any(selected >= teacher_count):
            raise ValueError("selected_teacher contains an invalid index")
        batch_indices = torch.arange(
            batch_size,
            device=teacher_responses.device,
        )[:, None]
        teacher_targets = teacher_responses[batch_indices, selected].detach()

        mask = self._prepare_mask(teacher_responses, valid_mask)
        valid_counts = mask.sum(dim=(1, 2, 3, 4)).float()
        if torch.any(valid_counts == 0):
            raise ValueError("each condition needs a valid action element")
        squared_error = (
            selected_student_responses.float() - teacher_targets.float()
        ).square()
        squared_error = squared_error * mask[:, None].float()
        per_candidate_loss = squared_error.sum(dim=(2, 3, 4, 5)) / (
            valid_counts[:, None]
        )
        if not torch.isfinite(per_candidate_loss).all():
            raise ValueError("execution losses must be finite")
        return ActionExecutionLossResult(
            loss=per_candidate_loss.mean(),
            per_candidate_loss=per_candidate_loss.detach(),
            selected_teacher=selected.detach(),
        )

    __call__ = compute

    def _prepare_mask(
        self,
        teacher_responses: Tensor,
        valid_mask: Tensor | None,
    ) -> Tensor:
        batch_size, _, channels, frames, positions, _ = (
            teacher_responses.shape
        )
        channel_mask = torch.zeros(
            channels,
            dtype=torch.bool,
            device=teacher_responses.device,
        )
        channel_mask[list(self.action_channel_ids)] = True
        channel_mask = channel_mask.view(1, channels, 1, 1, 1)
        if valid_mask is None:
            valid = torch.ones(
                batch_size,
                channels,
                frames,
                positions,
                1,
                dtype=torch.bool,
                device=teacher_responses.device,
            )
        else:
            valid = valid_mask.to(
                device=teacher_responses.device,
                dtype=torch.bool,
            )
            expected = (batch_size, channels, frames, positions, 1)
            if tuple(valid.shape) != expected:
                raise ValueError(f"valid_mask must have shape {expected}")
        return valid & channel_mask

    def _validate_inputs(
        self,
        teacher_responses: Tensor,
        selected_student_responses: Tensor,
        selected_teacher: Tensor,
    ) -> None:
        if teacher_responses.ndim != 6:
            raise ValueError(
                "teacher_responses must have shape [B,M,C,F,N,1]"
            )
        if selected_student_responses.ndim != 6:
            raise ValueError(
                "selected_student_responses must have shape [B,K,C,F,N,1]"
            )
        batch_size, teacher_count, channels = teacher_responses.shape[:3]
        student_count = selected_student_responses.shape[1]
        if selected_student_responses.shape[0] != batch_size:
            raise ValueError("teacher and student response batches differ")
        if teacher_count < 1 or student_count < 1:
            raise ValueError(
                "responses must contain teacher and student candidates"
            )
        if tuple(selected_student_responses.shape[2:]) != tuple(
            teacher_responses.shape[2:]
        ):
            raise ValueError("teacher and student response layouts differ")
        if tuple(selected_teacher.shape) != (batch_size, student_count):
            raise ValueError("selected_teacher must have shape [B,K]")
        if channels != self.action_dim:
            raise ValueError("response channel count differs from action_dim")
        if teacher_responses.shape[-1] != 1:
            raise ValueError("response trailing dimension must be one")
        if teacher_responses.device != selected_student_responses.device:
            raise ValueError("teacher and student responses must share one device")
        if not torch.isfinite(teacher_responses).all() or not torch.isfinite(
            selected_student_responses
        ).all():
            raise ValueError("responses must be finite")
