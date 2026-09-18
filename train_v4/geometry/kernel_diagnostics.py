"""Detached diagnostics for the Section 3.3 drifting kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.distributed as dist
from torch import Tensor

from .action_response_metric import ActionResponseMetricResult
from .bandwidth import KernelBandwidths, exact_median
from .drifting_kernel import DriftingKernelResult
from .video_metric import (
    VideoMetricResult,
    directed_pair_values,
    distinct_pair_values,
)


@dataclass(slots=True)
class KernelDiagnosticsResult:
    """Flat scalar values ready for the training logger."""

    scalars: dict[str, Tensor]


class KernelDiagnostics:
    """Measure kernel sharpness and teacher coverage."""

    def __init__(
        self,
        *,
        quantiles: Sequence[float] = (0.1, 0.25, 0.75, 0.9),
        epsilon: float = 1e-8,
    ) -> None:
        parsed_quantiles = tuple(float(value) for value in quantiles)
        if any(value <= 0 or value >= 1 for value in parsed_quantiles):
            raise ValueError("quantiles must lie strictly between zero and one")
        if len(set(parsed_quantiles)) != len(parsed_quantiles):
            raise ValueError("quantiles must be unique")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.quantiles = tuple(sorted(parsed_quantiles))
        self.epsilon = float(epsilon)

    @torch.no_grad()
    def compute(
        self,
        video: VideoMetricResult,
        action: ActionResponseMetricResult,
        kernel: DriftingKernelResult,
        bandwidths: KernelBandwidths,
    ) -> KernelDiagnosticsResult:
        """Return detached statistics pooled over the data-parallel batch."""

        self._validate_inputs(video, action, kernel)
        assert video.distances.student_teacher is not None
        assert video.distances.student_student is not None
        positive_weights = kernel.positive_weights.detach().float()
        negative_weights = kernel.negative_weights.detach().float()
        positive_entropy = self._entropy(positive_weights)
        negative_entropy = self._entropy(negative_weights)

        student_count = positive_weights.shape[1]
        coverage_mass = positive_weights.sum(dim=1)
        coverage_distribution = coverage_mass / student_count
        coverage_entropy = self._entropy(coverage_distribution)

        nearest_joint, nearest_student = (
            kernel.positive_joint_distance.detach().float().min(dim=1)
        )
        nearest_index = nearest_student.unsqueeze(1)
        nearest_video = torch.gather(
            kernel.positive_video_term.detach().float(),
            dim=1,
            index=nearest_index,
        ).squeeze(1)
        nearest_action = torch.gather(
            kernel.positive_action_term.detach().float(),
            dim=1,
            index=nearest_index,
        ).squeeze(1)

        student_teacher_video = (
            video.distances.student_teacher.detach().float()
        )
        student_teacher_action = action.student_teacher.detach().float()
        student_student_video = distinct_pair_values(
            video.distances.student_student
        ).float().detach()
        student_student_action = directed_pair_values(
            action.student_student
        ).float().detach()

        groups = {
            "st_video_raw": student_teacher_video.reshape(-1),
            "st_action_raw": student_teacher_action.reshape(-1),
            "st_video_scaled": kernel.positive_video_term.reshape(-1),
            "st_action_scaled": kernel.positive_action_term.reshape(-1),
            "st_joint": kernel.positive_joint_distance.reshape(-1),
            "ss_video_raw": student_student_video,
            "ss_action_raw": student_student_action,
            "ss_video_scaled": distinct_pair_values(
                kernel.negative_video_term
            ).float().detach(),
            "ss_action_scaled": directed_pair_values(
                kernel.negative_action_term
            ).float().detach(),
            "ss_joint": directed_pair_values(
                kernel.negative_joint_distance
            ).float().detach(),
            "positive_entropy": positive_entropy.reshape(-1),
            "positive_max_weight": positive_weights.max(dim=-1).values.reshape(-1),
            "positive_effective_neighbors": positive_entropy.exp().reshape(-1),
            "negative_entropy": negative_entropy.reshape(-1),
            "negative_max_weight": negative_weights.max(dim=-1).values.reshape(-1),
            "negative_effective_neighbors": negative_entropy.exp().reshape(-1),
            "coverage_mass_min": coverage_mass.min(dim=-1).values,
            "coverage_mass_median": self._median_last_axis(coverage_mass),
            "coverage_mass_max": coverage_mass.max(dim=-1).values,
            "coverage_entropy": coverage_entropy,
            "coverage_effective_teachers": coverage_entropy.exp(),
            "nearest_joint": nearest_joint.reshape(-1),
            "nearest_video": nearest_video.reshape(-1),
            "nearest_action": nearest_action.reshape(-1),
        }
        pooled = self._gather_groups(groups)

        scalars: dict[str, Tensor] = {
            "kernel/video_bandwidth": torch.tensor(
                bandwidths.video,
                dtype=torch.float32,
                device=positive_weights.device,
            ),
            "kernel/action_bandwidth": torch.tensor(
                bandwidths.action,
                dtype=torch.float32,
                device=positive_weights.device,
            ),
            "video/feature_scale": video.feature_scale.detach().float(),
        }
        distribution_groups = {
            "distance/student_teacher/video_raw": "st_video_raw",
            "distance/student_teacher/action_raw": "st_action_raw",
            "distance/student_teacher/video_over_bandwidth": "st_video_scaled",
            "distance/student_teacher/action_over_bandwidth": "st_action_scaled",
            "distance/student_teacher/joint": "st_joint",
            "distance/student_student/video_raw": "ss_video_raw",
            "distance/student_student/action_raw": "ss_action_raw",
            "distance/student_student/video_over_bandwidth": "ss_video_scaled",
            "distance/student_student/action_over_bandwidth": "ss_action_scaled",
            "distance/student_student/joint": "ss_joint",
            "weights/positive/entropy": "positive_entropy",
            "weights/positive/max": "positive_max_weight",
            "weights/positive/effective_neighbors": (
                "positive_effective_neighbors"
            ),
            "weights/negative/entropy": "negative_entropy",
            "weights/negative/max": "negative_max_weight",
            "weights/negative/effective_neighbors": (
                "negative_effective_neighbors"
            ),
            "coverage/mass_min": "coverage_mass_min",
            "coverage/mass_median": "coverage_mass_median",
            "coverage/mass_max": "coverage_mass_max",
            "coverage/entropy": "coverage_entropy",
            "coverage/effective_teachers": "coverage_effective_teachers",
            "nearest_student/joint": "nearest_joint",
            "nearest_student/video_term": "nearest_video",
            "nearest_student/action_term": "nearest_action",
        }
        for prefix, group_name in distribution_groups.items():
            self._add_summary(scalars, prefix, pooled[group_name])

        return KernelDiagnosticsResult(scalars=scalars)

    __call__ = compute

    def _add_summary(
        self,
        destination: dict[str, Tensor],
        prefix: str,
        values: Tensor,
    ) -> None:
        if values.numel() == 0:
            raise ValueError(f"cannot summarize empty diagnostic: {prefix}")
        values = values.float()
        destination[f"{prefix}/mean"] = values.mean()
        destination[f"{prefix}/median"] = exact_median(values)
        destination[f"{prefix}/min"] = values.min()
        destination[f"{prefix}/max"] = values.max()
        for quantile in self.quantiles:
            label = f"q{round(100 * quantile):02d}"
            destination[f"{prefix}/{label}"] = torch.quantile(values, quantile)

    @staticmethod
    def _entropy(weights: Tensor) -> Tensor:
        terms = torch.where(
            weights > 0,
            weights * weights.clamp_min(torch.finfo(weights.dtype).tiny).log(),
            torch.zeros_like(weights),
        )
        return -terms.sum(dim=-1)

    @classmethod
    def _median_last_axis(cls, values: Tensor) -> Tensor:
        ordered = values.sort(dim=-1).values
        middle = ordered.shape[-1] // 2
        if ordered.shape[-1] % 2:
            return ordered[..., middle]
        return (ordered[..., middle - 1] + ordered[..., middle]) * 0.5

    @staticmethod
    def _gather_groups(groups: dict[str, Tensor]) -> dict[str, Tensor]:
        names = tuple(groups)
        first = groups[names[0]]
        flattened = {
            name: value.detach().float().reshape(-1)
            for name, value in groups.items()
        }
        if any(value.device != first.device for value in flattened.values()):
            raise ValueError("diagnostic tensors must share one device")
        if not (dist.is_available() and dist.is_initialized()):
            return flattened

        world_size = dist.get_world_size()
        if world_size == 1:
            return flattened

        local_counts = torch.tensor(
            [flattened[name].numel() for name in names],
            dtype=torch.long,
            device=first.device,
        )
        rank_counts = [torch.empty_like(local_counts) for _ in range(world_size)]
        dist.all_gather(rank_counts, local_counts)
        count_matrix = torch.stack(rank_counts)

        local_payload = torch.cat([flattened[name] for name in names])
        payload_sizes = count_matrix.sum(dim=1)
        maximum_size = int(payload_sizes.max().item())
        padded = local_payload.new_zeros(maximum_size)
        padded[: local_payload.numel()] = local_payload
        rank_payloads = [torch.empty_like(padded) for _ in range(world_size)]
        dist.all_gather(rank_payloads, padded)

        gathered: dict[str, list[Tensor]] = {name: [] for name in names}
        for rank, payload in enumerate(rank_payloads):
            offset = 0
            for group_index, name in enumerate(names):
                count = int(count_matrix[rank, group_index].item())
                gathered[name].append(payload[offset : offset + count])
                offset += count
        return {
            name: torch.cat(rank_values)
            for name, rank_values in gathered.items()
        }

    @staticmethod
    def _validate_inputs(
        video: VideoMetricResult,
        action: ActionResponseMetricResult,
        kernel: DriftingKernelResult,
    ) -> None:
        if video.student_features is None:
            raise ValueError("video metric result has no student features")
        if video.distances.student_teacher is None:
            raise ValueError("video metric result has no student-teacher distances")
        if video.distances.student_student is None:
            raise ValueError("video metric result has no student-student distances")
        batch_size, teacher_count = video.teacher_features.shape[:2]
        student_count = video.student_features.shape[1]
        expected_positive = (batch_size, student_count, teacher_count)
        expected_negative = (batch_size, student_count, student_count)
        positive_tensors = (
            kernel.positive_video_term,
            kernel.positive_action_term,
            kernel.positive_joint_distance,
            kernel.positive_weights,
        )
        negative_tensors = (
            kernel.negative_video_term,
            kernel.negative_action_term,
            kernel.negative_joint_distance,
            kernel.negative_weights,
        )
        if any(tuple(tensor.shape) != expected_positive for tensor in positive_tensors):
            raise ValueError("positive kernel tensors must have shape [B,K,M]")
        if any(tuple(tensor.shape) != expected_negative for tensor in negative_tensors):
            raise ValueError("negative kernel tensors must have shape [B,K,K]")
