"""One-time teacher-teacher bandwidth calibration for Section 3.3."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.distributed as dist
from torch import Tensor

from .video_metric import VideoMetric, directed_pair_values, distinct_pair_values


def exact_median(values: Tensor) -> Tensor:
    """Return the conventional median, averaging two middle values."""

    if values.numel() == 0:
        raise ValueError("median requires at least one value")
    ordered = values.reshape(-1).sort().values
    middle = ordered.numel() // 2
    if ordered.numel() % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) * 0.5


@dataclass(frozen=True, slots=True)
class KernelBandwidths:
    """Fixed video and action bandwidths used by the drifting kernel."""

    video: float
    action: float

    def __post_init__(self) -> None:
        video = float(self.video)
        action = float(self.action)
        if not math.isfinite(video) or video <= 0:
            raise ValueError("video bandwidth must be finite and positive")
        if not math.isfinite(action) or action <= 0:
            raise ValueError("action bandwidth must be finite and positive")
        object.__setattr__(self, "video", video)
        object.__setattr__(self, "action", action)

    def state_dict(self) -> dict[str, float]:
        """Return checkpoint-friendly scalar state."""

        return {
            "video": self.video,
            "action": self.action,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "KernelBandwidths":
        return cls(
            video=float(state["video"]),
            action=float(state["action"]),
        )


@dataclass(slots=True)
class BandwidthCalibrationResult:
    """Bandwidths and the globally pooled distances that produced them."""

    bandwidths: KernelBandwidths
    feature_scale: Tensor
    video_distances: Tensor
    action_distances: Tensor


class BandwidthCalibrator:
    """Calibrate fixed kernel bandwidths from one teacher-only batch."""

    def __init__(self, video_metric: VideoMetric) -> None:
        self.video_metric = video_metric

    @torch.no_grad()
    def calibrate(
        self,
        teacher_features: Tensor,
        teacher_action_distances: Tensor,
        *,
        video_valid_mask: Tensor | None = None,
    ) -> BandwidthCalibrationResult:
        """Return medians of pooled distinct teacher-teacher distances.

        ``teacher_features`` has shape ``[B,M,F,N,D]`` and
        ``teacher_action_distances`` has shape ``[B,M,M]``. The latter is
        measured by applying every teacher-action probe to every teacher
        video, comparing (i,j) under j's own probe. Video pairs contribute
        once; directed action pairs contribute in both non-self directions.
        """

        self._validate_candidate_axes(
            teacher_features,
            teacher_action_distances,
        )

        video_result = self.video_metric.compute(
            teacher_features,
            valid_mask=video_valid_mask,
        )
        local_video = distinct_pair_values(
            video_result.distances.teacher_teacher
        ).float().detach()
        local_action = directed_pair_values(
            teacher_action_distances
        ).float().detach()
        video_distances = self._gather_across_data_parallel(local_video)
        action_distances = self._gather_across_data_parallel(local_action)

        self._validate_distances(video_distances, name="video")
        self._validate_distances(action_distances, name="action")

        video_bandwidth = exact_median(video_distances)
        action_bandwidth = exact_median(action_distances)
        bandwidths = KernelBandwidths(
            video=float(video_bandwidth.item()),
            action=float(action_bandwidth.item()),
        )
        return BandwidthCalibrationResult(
            bandwidths=bandwidths,
            feature_scale=video_result.feature_scale.detach(),
            video_distances=video_distances,
            action_distances=action_distances,
        )

    __call__ = calibrate

    @staticmethod
    def _validate_candidate_axes(
        teacher_features: Tensor,
        teacher_action_distances: Tensor,
    ) -> None:
        if teacher_features.ndim != 5:
            raise ValueError("teacher_features must have shape [B,M,F,N,D]")
        if teacher_action_distances.ndim != 3:
            raise ValueError(
                "teacher_action_distances must have shape [B,M,M]"
            )
        if teacher_features.shape[:2] != teacher_action_distances.shape[:2]:
            raise ValueError(
                "teacher feature and action-distance candidate axes differ"
            )
        if teacher_action_distances.shape[1] != teacher_action_distances.shape[2]:
            raise ValueError("teacher action-distance matrix must be square")
        if teacher_features.shape[0] < 1:
            raise ValueError("calibration requires at least one condition")
        if teacher_features.shape[1] < 2:
            raise ValueError(
                "calibration requires at least two teacher candidates"
            )
        if teacher_features.device != teacher_action_distances.device:
            raise ValueError(
                "teacher features and action distances must share a device"
            )

    @staticmethod
    def _gather_across_data_parallel(values: Tensor) -> Tensor:
        if not (dist.is_available() and dist.is_initialized()):
            return values

        world_size = dist.get_world_size()
        if world_size == 1:
            return values

        local_count = torch.tensor(
            [values.numel()],
            dtype=torch.long,
            device=values.device,
        )
        gathered_counts = [torch.zeros_like(local_count) for _ in range(world_size)]
        dist.all_gather(gathered_counts, local_count)
        counts = [int(count.item()) for count in gathered_counts]
        maximum_count = max(counts)

        padded = values.new_zeros(maximum_count)
        padded[: values.numel()] = values
        gathered = [torch.empty_like(padded) for _ in range(world_size)]
        dist.all_gather(gathered, padded)
        return torch.cat(
            [rank_values[:count] for rank_values, count in zip(gathered, counts)]
        )

    @staticmethod
    def _validate_distances(values: Tensor, *, name: str) -> None:
        if values.numel() == 0:
            raise ValueError(f"{name} bandwidth requires a distinct pair")
        if not torch.isfinite(values).all():
            raise ValueError(f"{name} calibration distances must be finite")
        if torch.any(values < 0):
            raise ValueError(f"{name} calibration distances must be non-negative")
