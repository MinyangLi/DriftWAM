"""Action-aware candidate weights and token-resolved drifting field."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .action_response_metric import ActionResponseMetricResult
from .bandwidth import KernelBandwidths
from .video_metric import VideoMetricResult


@dataclass(slots=True)
class DriftingKernelResult:
    """Joint distances, candidate weights, and feature-space drift."""

    positive_video_term: Tensor
    positive_action_term: Tensor
    negative_video_term: Tensor
    negative_action_term: Tensor
    positive_joint_distance: Tensor
    negative_joint_distance: Tensor
    positive_weights: Tensor
    negative_weights: Tensor
    positive_drift: Tensor
    near_student_drift: Tensor
    drift: Tensor


class DriftingKernel:
    """Apply the Section 3.3 scalar Laplacian candidate kernel."""

    def __init__(self, *, beta_rep: float = 1.0) -> None:
        if beta_rep < 0:
            raise ValueError("beta_rep must be non-negative")
        self.beta_rep = float(beta_rep)

    @staticmethod
    @torch.no_grad()
    def compute_positive_weights(
        video_distances: Tensor,
        action_distances: Tensor,
        bandwidths: KernelBandwidths,
    ) -> Tensor:
        """Shared attraction rule for both probe matching and drifting."""

        joint_distance = (
            video_distances.detach().float() / bandwidths.video
            + action_distances.detach().float() / bandwidths.action
        )
        return torch.softmax(-joint_distance, dim=-1)

    @torch.no_grad()
    def compute(
        self,
        video: VideoMetricResult,
        action: ActionResponseMetricResult,
        bandwidths: KernelBandwidths,
    ) -> DriftingKernelResult:
        """Compute candidate weights and a drift with shape ``[B,K,F,N,D]``.

        ``B`` is the condition count, ``K`` the student-candidate count,
        ``F`` the latent-frame count, ``N`` the spatial-token count, and
        ``D`` the hidden width. Teacher candidates use the separate ``M``
        axis. Only ``M`` or the neighboring-student axis is reduced.
        """

        self._validate_inputs(video, action)
        assert video.student_features is not None
        assert video.distances.student_teacher is not None
        assert video.distances.student_student is not None

        positive_video_term = (
            video.distances.student_teacher.detach().float()
            / bandwidths.video
        )
        positive_action_term = (
            action.student_teacher.detach().float()
            / bandwidths.action
        )
        negative_video_term = (
            video.distances.student_student.detach().float()
            / bandwidths.video
        )
        negative_action_term = (
            action.student_student.detach().float()
            / bandwidths.action
        )

        positive_joint_distance = positive_video_term + positive_action_term
        negative_joint_distance = negative_video_term + negative_action_term

        positive_weights = self.compute_positive_weights(
            video.distances.student_teacher, action.student_teacher, bandwidths
        )
        negative_logits = -negative_joint_distance
        student_count = negative_logits.shape[1]
        diagonal = torch.eye(
            student_count,
            dtype=torch.bool,
            device=negative_logits.device,
        ).unsqueeze(0)
        negative_weights = torch.softmax(
            negative_logits.masked_fill(diagonal, -torch.inf),
            dim=-1,
        )

        teacher_features = video.teacher_features.detach()
        student_features = video.student_features.detach()
        feature_dtype = student_features.dtype
        weighted_teacher = torch.einsum(
            "bkm,bmfnd->bkfnd",
            positive_weights.to(dtype=feature_dtype),
            teacher_features.to(dtype=feature_dtype),
        )
        weighted_students = torch.einsum(
            "bkj,bjfnd->bkfnd",
            negative_weights.to(dtype=feature_dtype),
            student_features,
        )

        positive_drift = weighted_teacher - student_features
        near_student_drift = weighted_students - student_features
        drift = positive_drift - self.beta_rep * near_student_drift

        return DriftingKernelResult(
            positive_video_term=positive_video_term,
            positive_action_term=positive_action_term,
            negative_video_term=negative_video_term,
            negative_action_term=negative_action_term,
            positive_joint_distance=positive_joint_distance,
            negative_joint_distance=negative_joint_distance,
            positive_weights=positive_weights,
            negative_weights=negative_weights,
            positive_drift=positive_drift,
            near_student_drift=near_student_drift,
            drift=drift,
        )

    __call__ = compute

    @staticmethod
    def _validate_inputs(
        video: VideoMetricResult,
        action: ActionResponseMetricResult,
    ) -> None:
        if video.student_features is None:
            raise ValueError("video metric result has no student features")
        if video.distances.student_teacher is None:
            raise ValueError("video metric result has no student-teacher distances")
        if video.distances.student_student is None:
            raise ValueError("video metric result has no student-student distances")

        teacher_features = video.teacher_features
        student_features = video.student_features
        batch_size, teacher_count = teacher_features.shape[:2]
        student_batch, student_count = student_features.shape[:2]
        if batch_size != student_batch:
            raise ValueError("teacher and student feature batch sizes differ")
        if teacher_features.shape[2:] != student_features.shape[2:]:
            raise ValueError("teacher and student feature layouts differ")
        if teacher_count < 2:
            raise ValueError("the kernel requires at least two teacher candidates")
        if student_count < 2:
            raise ValueError("the kernel requires at least two student candidates")

        expected_positive = (batch_size, student_count, teacher_count)
        expected_negative = (batch_size, student_count, student_count)
        positive_distances = (
            video.distances.student_teacher,
            action.student_teacher,
        )
        negative_distances = (
            video.distances.student_student,
            action.student_student,
        )
        if any(
            tuple(distances.shape) != expected_positive
            for distances in positive_distances
        ):
            raise ValueError(
                "student-teacher distances must have shape [B,K,M]"
            )
        if any(
            tuple(distances.shape) != expected_negative
            for distances in negative_distances
        ):
            raise ValueError(
                "student-student distances must have shape [B,K,K]"
            )

        device = teacher_features.device
        tensors = (
            student_features,
            *positive_distances,
            *negative_distances,
        )
        if any(tensor.device != device for tensor in tensors):
            raise ValueError("kernel inputs must share one device")
