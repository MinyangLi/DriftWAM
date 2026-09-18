"""Masked velocity-response distances for the action-aware kernel."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from ..forwards.action_signature import ROBOTWIN_ACTION_CHANNELS


@dataclass(slots=True)
class ActionResponseMetricResult:
    """Student-teacher and student-student scalar response distances."""

    student_teacher: Tensor
    student_student: Tensor


class ActionResponseMetric:
    """Compute pairwise MSE without decoding velocities into physical actions."""

    def __init__(
        self,
        action_dim: int = 30,
        *,
        action_channel_ids: Sequence[int] = ROBOTWIN_ACTION_CHANNELS,
    ) -> None:
        self.action_dim = int(action_dim)
        self.action_channel_ids = tuple(int(index) for index in action_channel_ids)
        if self.action_dim < 1:
            raise ValueError("action_dim must be positive")
        if not self.action_channel_ids:
            raise ValueError("action_channel_ids must not be empty")
        if len(set(self.action_channel_ids)) != len(self.action_channel_ids):
            raise ValueError("action_channel_ids must be unique")
        if any(
            index < 0 or index >= self.action_dim
            for index in self.action_channel_ids
        ):
            raise ValueError("action_channel_ids contains an invalid channel")

    @torch.no_grad()
    def compute(
        self,
        teacher_responses: Tensor,
        student_responses: Tensor,
        valid_mask: Tensor | None = None,
        *,
        selected_teacher: Tensor,
    ) -> ActionResponseMetricResult:
        """Compare both endpoints under the compared candidate's probe.

        ``selected_teacher[B,j]`` is student j's detached positive-kernel
        argmax. The directed negative entry (k,j) uses this same probe for
        both student k and student j; it need not equal entry (j,k).
        """

        return ActionResponseMetricResult(
            student_teacher=self.compute_student_teacher(
                teacher_responses, student_responses, valid_mask
            ),
            student_student=self.compute_student_student(
                student_responses, selected_teacher, valid_mask
            ),
        )

    @torch.no_grad()
    def compute_student_teacher(
        self,
        teacher_responses: Tensor,
        student_responses: Tensor,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        """Return [B,K,M] distances using each teacher's own action probe.

        This first stage is independent of repulsion and supplies the
        action distances needed to select the positive-kernel matches.
        """

        self._validate_responses(teacher_responses, student_responses)
        mask = self._prepare_mask(teacher_responses, valid_mask)
        valid_counts = mask.sum(dim=(1, 2, 3, 4)).float()
        if torch.any(valid_counts == 0):
            raise ValueError("each condition needs a valid action element")
        teacher = teacher_responses.detach().float()
        student = student_responses.detach().float()
        error = (student - teacher[:, None]).square()
        error = error * mask[:, None, None].float()
        distances = error.sum(dim=(3, 4, 5, 6)) / valid_counts[:, None, None]
        if not torch.isfinite(distances).all():
            raise ValueError("action responses must produce finite distances")
        return distances

    @torch.no_grad()
    def compute_student_student(
        self,
        student_responses: Tensor,
        selected_teacher: Tensor,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        """Return D[k,j] = MSE(r_mj(S_k), r_mj(S_j)), with zero diagonal."""

        if student_responses.ndim != 7:
            raise ValueError("student_responses must have shape [B,K,M,C,F,N,1]")
        batch_size, student_count, teacher_count = student_responses.shape[:3]
        if student_count < 2 or teacher_count < 1:
            raise ValueError("responses require one teacher and two students")
        if student_responses.shape[3] != self.action_dim:
            raise ValueError("action responses have the wrong channel count")
        if student_responses.shape[-1] != 1:
            raise ValueError("response trailing dimension must be one")
        if tuple(selected_teacher.shape) != (batch_size, student_count):
            raise ValueError("selected_teacher must have shape [B,K]")
        if selected_teacher.dtype not in (
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
        ):
            raise ValueError("selected_teacher must contain integer indices")
        selected = selected_teacher.detach().to(
            device=student_responses.device, dtype=torch.long
        )
        if torch.any(selected < 0) or torch.any(selected >= teacher_count):
            raise ValueError("selected_teacher contains an invalid index")
        if not torch.isfinite(student_responses).all():
            raise ValueError("action responses must be finite")

        mask = self._prepare_mask(student_responses[:, 0], valid_mask)
        valid_counts = mask.sum(dim=(1, 2, 3, 4)).float()
        if torch.any(valid_counts == 0):
            raise ValueError("each condition needs a valid action element")
        student = student_responses.detach().float()
        batches = torch.arange(batch_size, device=student.device)
        candidates = torch.arange(student_count, device=student.device)
        # Gather the column candidate j's probe on BOTH endpoints. This
        # avoids materializing the former [B,K,K,M,C,F,N,1] differences.
        query_responses = student[
            batches[:, None, None], candidates[None, :, None], selected[:, None, :]
        ]
        neighbor_responses = student[
            batches[:, None], candidates[None, :], selected
        ]
        error = (query_responses - neighbor_responses[:, None]).square()
        error = error * mask[:, None, None].float()
        distances = error.sum(dim=(3, 4, 5, 6)) / valid_counts[:, None, None]
        diagonal = torch.eye(student_count, dtype=torch.bool, device=student.device)
        distances = distances.masked_fill(diagonal[None], 0)
        if not torch.isfinite(distances).all():
            raise ValueError("action responses must produce finite distances")
        return distances

    __call__ = compute

    def _prepare_mask(
        self,
        responses: Tensor,
        valid_mask: Tensor | None,
    ) -> Tensor:
        batch_size, _, channels, frames, positions, _ = responses.shape
        channel_mask = torch.zeros(
            channels,
            dtype=torch.bool,
            device=responses.device,
        )
        channel_mask[list(self.action_channel_ids)] = True
        channel_mask = channel_mask.view(1, channels, 1, 1, 1)

        if valid_mask is None:
            valid = torch.ones(
                batch_size,
                1,
                frames,
                positions,
                1,
                dtype=torch.bool,
                device=responses.device,
            )
        else:
            valid = valid_mask.to(device=responses.device, dtype=torch.bool)
            if valid.ndim == 2:
                if tuple(valid.shape) != (frames, positions):
                    raise ValueError("2D valid_mask must have shape [F,N]")
                valid = valid.view(1, 1, frames, positions, 1).expand(
                    batch_size,
                    -1,
                    -1,
                    -1,
                    -1,
                )
            elif valid.ndim == 3:
                if tuple(valid.shape) != (batch_size, frames, positions):
                    raise ValueError("3D valid_mask must have shape [B,F,N]")
                valid = valid[:, None, :, :, None]
            elif valid.ndim == 5:
                expected = (batch_size, channels, frames, positions, 1)
                if tuple(valid.shape) != expected:
                    raise ValueError(f"native valid_mask must have shape {expected}")
            else:
                raise ValueError(
                    "valid_mask must have shape [F,N], [B,F,N], or [B,C,F,N,1]"
                )
        return valid & channel_mask

    def _validate_responses(
        self,
        teacher_responses: Tensor,
        student_responses: Tensor,
    ) -> None:
        if teacher_responses.ndim != 6:
            raise ValueError("teacher_responses must have shape [B,M,C,F,N,1]")
        if student_responses.ndim != 7:
            raise ValueError(
                "student_responses must have shape [B,K,M,C,F,N,1]"
            )
        if teacher_responses.shape[0] != student_responses.shape[0]:
            raise ValueError("teacher and student response batches differ")
        if teacher_responses.shape[1] != student_responses.shape[2]:
            raise ValueError("teacher response and probe axes differ")
        if teacher_responses.shape[2:] != student_responses.shape[3:]:
            raise ValueError("teacher and student response layouts differ")
        if teacher_responses.shape[1] < 1 or student_responses.shape[1] < 2:
            raise ValueError("responses require one teacher and two students")
        if teacher_responses.shape[2] != self.action_dim:
            raise ValueError("response channel count differs from action_dim")
        if teacher_responses.shape[-1] != 1:
            raise ValueError("response trailing dimension must be one")
        if not torch.isfinite(teacher_responses).all() or not torch.isfinite(
            student_responses
        ).all():
            raise ValueError("action responses must be finite")

