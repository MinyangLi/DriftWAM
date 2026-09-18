"""Batch-shared video-feature normalization and scalar video distances."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor


def distinct_pair_values(distances: Tensor) -> Tensor:
    """Flatten the strict upper triangle of [B,Q,Q] distances."""

    if distances.ndim != 3 or distances.shape[1] != distances.shape[2]:
        raise ValueError("distances must have shape [B,Q,Q]")
    candidate_count = distances.shape[1]
    indices = torch.triu_indices(
        candidate_count,
        candidate_count,
        offset=1,
        device=distances.device,
    )
    return distances[:, indices[0], indices[1]].reshape(-1)


def directed_pair_values(distances: Tensor) -> Tensor:
    """Flatten all non-self entries, preserving both directions of a pair."""

    if distances.ndim != 3 or distances.shape[1] != distances.shape[2]:
        raise ValueError("distances must have shape [B,Q,Q]")
    diagonal = torch.eye(
        distances.shape[1], dtype=torch.bool, device=distances.device
    )
    return distances[:, ~diagonal].reshape(-1)


@dataclass(slots=True)
class VideoDistances:
    """Within-condition scalar distances between candidate sets."""

    teacher_teacher: Tensor
    student_teacher: Tensor | None
    student_student: Tensor | None


@dataclass(slots=True)
class VideoMetricResult:
    """Normalized token features, their shared scale, and pair distances."""

    teacher_features: Tensor
    student_features: Tensor | None
    feature_scale: Tensor
    distances: VideoDistances


class VideoMetric:
    """Apply the Section 3.2 video geometry without token pooling."""

    def __init__(self, *, epsilon: float = 1e-8) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = epsilon

    def compute(
        self,
        teacher_features: Tensor,
        student_features: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> VideoMetricResult:
        """Normalize features and compute ``D_V`` for all required pairs.

        Teacher features have shape ``[B,M,F,N,D]`` and student features have
        shape ``[B,K,F,N,D]``. The returned distance matrices have shapes
        ``[B,M,M]``, ``[B,K,M]``, and ``[B,K,K]`` respectively.
        """

        self._validate_features(teacher_features, student_features)
        valid = self._prepare_valid_mask(teacher_features, valid_mask)

        raw_teacher_teacher = self._pairwise_rms(
            teacher_features.detach(),
            teacher_features.detach(),
            valid,
        )
        raw_teacher_teacher = self._zero_diagonal(raw_teacher_teacher)

        raw_student_teacher = None
        raw_student_student = None
        if student_features is not None:
            raw_student_teacher = self._pairwise_rms(
                student_features.detach(),
                teacher_features.detach(),
                valid,
            )
            raw_student_student = self._pairwise_rms(
                student_features.detach(),
                student_features.detach(),
                valid,
            )
            raw_student_student = self._zero_diagonal(raw_student_student)

        feature_scale = self._batch_feature_scale(
            raw_teacher_teacher,
            raw_student_teacher,
            raw_student_student,
        )
        normalization_scale = feature_scale.clamp_min(self.epsilon).detach()

        normalized_teacher = teacher_features / normalization_scale.to(
            dtype=teacher_features.dtype
        )
        normalized_student = None
        if student_features is not None:
            normalized_student = student_features / normalization_scale.to(
                dtype=student_features.dtype
            )

        distances = VideoDistances(
            teacher_teacher=raw_teacher_teacher / normalization_scale,
            student_teacher=(
                None
                if raw_student_teacher is None
                else raw_student_teacher / normalization_scale
            ),
            student_student=(
                None
                if raw_student_student is None
                else raw_student_student / normalization_scale
            ),
        )
        return VideoMetricResult(
            teacher_features=normalized_teacher,
            student_features=normalized_student,
            feature_scale=feature_scale,
            distances=distances,
        )

    __call__ = compute

    @staticmethod
    def _validate_features(
        teacher_features: Tensor,
        student_features: Tensor | None,
    ) -> None:
        if teacher_features.ndim != 5:
            raise ValueError("teacher_features must have shape [B,M,F,N,D]")
        if teacher_features.shape[1] < 1:
            raise ValueError("teacher_features must contain at least one candidate")
        if student_features is None:
            return
        if student_features.ndim != 5:
            raise ValueError("student_features must have shape [B,K,F,N,D]")
        if student_features.shape[1] < 1:
            raise ValueError("student_features must contain at least one candidate")
        if teacher_features.device != student_features.device:
            raise ValueError("teacher and student features must be on one device")
        if teacher_features.shape[0] != student_features.shape[0]:
            raise ValueError("teacher and student batch sizes differ")
        if teacher_features.shape[2:] != student_features.shape[2:]:
            raise ValueError("teacher and student feature layouts differ")

    @staticmethod
    def _prepare_valid_mask(
        features: Tensor,
        valid_mask: Tensor | None,
    ) -> Tensor:
        batch_size, _, frames, tokens, _ = features.shape
        if valid_mask is None:
            valid = torch.ones(
                batch_size,
                frames,
                tokens,
                dtype=torch.bool,
                device=features.device,
            )
        else:
            valid = valid_mask.to(device=features.device, dtype=torch.bool)
            if valid.ndim == 2:
                valid = valid.unsqueeze(0).expand(batch_size, -1, -1)
            if tuple(valid.shape) != (batch_size, frames, tokens):
                raise ValueError(
                    f"valid_mask must have shape [{batch_size},{frames},{tokens}] "
                    f"or [{frames},{tokens}]"
                )
        if torch.any(valid.sum(dim=(1, 2)) == 0):
            raise ValueError("each condition must contain at least one valid token")
        return valid

    @staticmethod
    def _pairwise_rms(
        left: Tensor,
        right: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        valid = valid_mask.float().unsqueeze(1).unsqueeze(-1)
        left_weighted = (left.float() * valid).flatten(start_dim=2)
        right_weighted = (right.float() * valid).flatten(start_dim=2)

        left_squared = left_weighted.square().sum(dim=-1, keepdim=True)
        right_squared = right_weighted.square().sum(dim=-1).unsqueeze(1)
        cross = torch.bmm(left_weighted, right_weighted.transpose(1, 2))
        squared_distance = (left_squared + right_squared - 2.0 * cross).clamp_min(0)

        hidden_width = left.shape[-1]
        denominator = (
            valid_mask.sum(dim=(1, 2)).float() * hidden_width
        ).view(-1, 1, 1)
        return torch.sqrt(squared_distance / denominator)

    @staticmethod
    def _zero_diagonal(distances: Tensor) -> Tensor:
        candidate_count = distances.shape[1]
        diagonal = torch.eye(
            candidate_count,
            dtype=torch.bool,
            device=distances.device,
        ).unsqueeze(0)
        return distances.masked_fill(diagonal, 0)

    def _batch_feature_scale(
        self,
        teacher_teacher: Tensor,
        student_teacher: Tensor | None,
        student_student: Tensor | None,
    ) -> Tensor:
        pair_groups = [distinct_pair_values(teacher_teacher)]
        if student_teacher is not None:
            pair_groups.append(student_teacher.reshape(-1))
        if student_student is not None:
            pair_groups.append(distinct_pair_values(student_student))

        local_sum = torch.zeros(
            (),
            dtype=torch.float32,
            device=teacher_teacher.device,
        )
        local_count = 0
        for values in pair_groups:
            local_sum = local_sum + values.float().sum()
            local_count += values.numel()

        stats = torch.stack(
            [
                local_sum,
                torch.tensor(
                    float(local_count),
                    dtype=torch.float32,
                    device=local_sum.device,
                ),
            ]
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        if stats[1].item() == 0:
            raise ValueError("feature scale requires at least one non-self pair")
        return (stats[0] / stats[1]).detach()
