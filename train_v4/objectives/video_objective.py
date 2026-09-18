"""Video drifting objective weighted by action-response geometry."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from ..config import DistillationConfig
from ..geometry.action_response_metric import ActionResponseMetricResult
from ..geometry.bandwidth import KernelBandwidths
from ..geometry.drifting_kernel import DriftingKernel, DriftingKernelResult
from ..geometry.kernel_diagnostics import KernelDiagnostics, KernelDiagnosticsResult
from ..geometry.video_metric import VideoMetricResult
from .video_drifting_loss import VideoDriftingLoss, VideoDriftingLossResult


@dataclass(slots=True)
class VideoObjectiveResult:
    """Drifting loss, its kernel state, and optional detached logs."""

    loss: Tensor
    drifting: VideoDriftingLossResult
    kernel: DriftingKernelResult
    diagnostics: KernelDiagnosticsResult | None
    scalars: dict[str, Tensor]


class VideoObjective:
    """Train the video student with an action-response-weighted drift field."""

    def __init__(
        self,
        config: DistillationConfig,
        *,
        kernel: DriftingKernel | None = None,
        drifting_loss: VideoDriftingLoss | None = None,
        diagnostics: KernelDiagnostics | None = None,
    ) -> None:
        self.config = config
        self.kernel = kernel or DriftingKernel(beta_rep=config.beta_rep)
        self.drifting_loss = drifting_loss or VideoDriftingLoss()
        self.diagnostics = diagnostics or KernelDiagnostics()

    def compute(
        self,
        video: VideoMetricResult,
        action: ActionResponseMetricResult,
        bandwidths: KernelBandwidths,
        *,
        video_valid_mask: Tensor | None = None,
        collect_diagnostics: bool = False,
        release_drift_fields: bool = False,
    ) -> VideoObjectiveResult:
        """Compute the action-response-guided video drifting loss.

        ``action`` contains detached one-step velocity-response distances.
        It changes only the kernel weights. The optional execution loss is
        combined by ``TrainingStep``; this objective never optimizes the
        frozen action model.
        """

        if video.student_features is None:
            raise ValueError("video metric result has no student features")

        kernel_result = self.kernel(video, action, bandwidths)
        drifting_result = self.drifting_loss(
            video.student_features,
            kernel_result.drift,
            video_valid_mask,
        )
        loss = self.config.video_drifting_loss_weight * drifting_result.loss

        diagnostics_result = None
        if collect_diagnostics:
            diagnostics_result = self.diagnostics(
                video,
                action,
                kernel_result,
                bandwidths,
            )

        if release_drift_fields:
            # The drifting target has already copied and detached these
            # token-sized tensors; its backward graph no longer reads them.
            kernel_result.positive_drift = kernel_result.positive_drift.new_empty(0)
            kernel_result.near_student_drift = (
                kernel_result.near_student_drift.new_empty(0)
            )
            kernel_result.drift = kernel_result.drift.new_empty(0)

        scalars = {
            "loss/video_total": loss.detach(),
            "loss/video_drifting": drifting_result.loss.detach(),
            "loss/video_drifting_contribution": loss.detach(),
            "video/drift_rms": drifting_result.drift_rms.detach(),
        }
        if diagnostics_result is not None:
            scalars.update(diagnostics_result.scalars)

        return VideoObjectiveResult(
            loss=loss,
            drifting=drifting_result,
            kernel=kernel_result,
            diagnostics=diagnostics_result,
            scalars=scalars,
        )

    __call__ = compute
