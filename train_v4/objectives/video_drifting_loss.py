"""Batch-global normalized video drifting loss from Section 3.4."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor


@dataclass(slots=True)
class VideoDriftingLossResult:
    """Scalar training loss and its detached global drift scale."""

    loss: Tensor
    drift_rms: Tensor


class VideoDriftingLoss:
    """Regress student features toward one stopped unit drift step."""

    def __init__(self, *, epsilon: float = 1e-8) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = float(epsilon)

    def compute(
        self,
        student_features: Tensor,
        drift: Tensor,
        valid_mask: Tensor | None = None,
    ) -> VideoDriftingLossResult:
        """Compute valid-element MSE for [B,K,F,N,D] features.

        B is the condition count, K the student-candidate count, F the
        latent-frame count, N the spatial-token count, and D the hidden width.
        The drift RMS is shared across every valid element and every
        initialized data-parallel rank.
        """

        self._validate_features(student_features, drift)
        valid = self._prepare_valid_mask(student_features, valid_mask)
        detached_drift = drift.detach().float()
        valid_weights = valid[:, None, :, :, None].float()

        drift_rms = self._global_drift_rms(
            detached_drift,
            valid,
            valid_weights,
        )
        normalized_drift = detached_drift / drift_rms
        prediction = student_features.float()
        target = (student_features.detach().float() + normalized_drift).detach()

        squared_error = (prediction - target).square() * valid_weights
        candidate_count = student_features.shape[1]
        hidden_width = student_features.shape[-1]
        denominator = valid.sum().float() * candidate_count * hidden_width
        loss = squared_error.sum() / denominator
        return VideoDriftingLossResult(
            loss=loss,
            drift_rms=drift_rms,
        )

    __call__ = compute

    def _global_drift_rms(
        self,
        drift: Tensor,
        valid: Tensor,
        valid_weights: Tensor,
    ) -> Tensor:
        candidate_count = drift.shape[1]
        hidden_width = drift.shape[-1]
        numerator = (drift.square() * valid_weights).sum()
        denominator = valid.sum().float() * candidate_count * hidden_width
        stats = torch.stack([numerator, denominator])
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        if stats[1].item() == 0:
            raise ValueError("drift normalization requires a valid feature")
        return torch.sqrt(stats[0] / stats[1] + self.epsilon).detach()

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
            raise ValueError("each condition must contain a valid feature token")
        return valid

    @staticmethod
    def _validate_features(student_features: Tensor, drift: Tensor) -> None:
        if student_features.ndim != 5:
            raise ValueError("student_features must have shape [B,K,F,N,D]")
        if student_features.shape[0] < 1 or student_features.shape[1] < 1:
            raise ValueError("student_features must contain a candidate")
        if tuple(drift.shape) != tuple(student_features.shape):
            raise ValueError("drift and student_features must have one shape")
        if drift.device != student_features.device:
            raise ValueError("drift and student_features must share a device")
