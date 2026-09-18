"""One differentiable micro-batch for action-response-guided video drifting."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor

from ..forwards.action_response import ActionResponseProbe
from ..forwards.action_signature import ActionSignatureCondition
from ..forwards.student_video import StudentVideoGenerator
from ..forwards.video_feature import VideoFeatureExtractor
from ..geometry.action_response_metric import (
    ActionResponseMetric,
    ActionResponseMetricResult,
)
from ..geometry.bandwidth import KernelBandwidths
from ..geometry.video_metric import VideoMetric
from ..objectives.action_execution_loss import (
    ActionExecutionLoss,
    ActionExecutionLossResult,
)
from ..objectives.action_consistency_loss import (
    ActionConsistencyLoss,
    ActionConsistencyLossResult,
)
from ..objectives.action_training_pair import ActionTrainingPair
from ..objectives.video_objective import VideoObjective, VideoObjectiveResult
from ..training_config import TrainingConfig
from .teacher_signature_cache import TeacherSignatureCache
from .training_data import TrainingBatch


@dataclass(slots=True)
class DeferredExecutionPlan:
    """Detached inputs required to rebuild students after video backward."""

    video_noise: Tensor
    condition: ActionSignatureCondition
    initial_frame: Tensor | None
    teacher_responses: Tensor
    noisy_teacher_actions: Tensor
    action_timestep_ids: Tensor
    selected_teacher: Tensor
    action_valid_mask: Tensor


@dataclass(slots=True)
class ActionConsistencyPlan:
    """Detached GT pair for the deferred action consistency pass."""

    pair: ActionTrainingPair
    condition: ActionSignatureCondition


@dataclass(slots=True)
class TrainingStepResult:
    """Loss graphs and detached state produced by one primary forward."""

    video_loss: Tensor
    combined_loss: Tensor | None
    video_objective: VideoObjectiveResult
    execution: ActionExecutionLossResult | None
    deferred_execution: DeferredExecutionPlan | None
    action_consistency_plan: ActionConsistencyPlan
    scalars: dict[str, Tensor]


class TrainingStep:
    """Build video drifting and optional low-memory action execution."""

    def __init__(
        self,
        config: TrainingConfig,
        bandwidths: KernelBandwidths,
        student_video_generator: StudentVideoGenerator,
        action_response_probe: ActionResponseProbe,
        video_feature_extractor: VideoFeatureExtractor,
        video_metric: VideoMetric,
        action_response_metric: ActionResponseMetric,
        video_objective: VideoObjective,
        teacher_signature_cache: TeacherSignatureCache,
        action_execution_loss: ActionExecutionLoss | None = None,
        action_consistency_loss: ActionConsistencyLoss | None = None,
    ) -> None:
        if config.signature_horizon != 1:
            raise ValueError(
                "the online one-step video generator currently requires "
                "signature_horizon=1"
            )
        self.config = config
        self.bandwidths = bandwidths
        self.student_video_generator = student_video_generator
        self.action_response_probe = action_response_probe
        self.video_feature_extractor = video_feature_extractor
        self.video_metric = video_metric
        self.action_response_metric = action_response_metric
        self.video_objective = video_objective
        self.teacher_signature_cache = teacher_signature_cache
        self.action_execution_loss = action_execution_loss
        if action_consistency_loss is None:
            raise ValueError("v4 requires an action consistency loss")
        self.action_consistency_loss = action_consistency_loss
        if (
            config.execution_response_mode != "off"
            and action_execution_loss is None
        ):
            raise ValueError(
                "an action execution loss is required when execution is enabled"
            )

    def compute(
        self,
        batch: TrainingBatch,
        *,
        collect_diagnostics: bool = False,
    ) -> TrainingStepResult:
        """Build video drifting with an optional action execution loss.

        Teacher action signatures are clean paired probes. A fresh action
        noise realization and fresh timestep vector are sampled for every
        micro-batch, then reused across all teacher probes and candidates.
        """

        student = self.student_video_generator.student
        parameter = next(student.parameters())
        device = parameter.device
        dtype = getattr(torch, self.config.param_dtype)
        batch = batch.to(device, dtype=dtype)
        self._validate_batch(batch)
        condition = batch.condition()

        batch_size = batch.batch_size
        teacher_videos = batch.teacher_videos.detach()
        video_shape = tuple(teacher_videos.shape[2:])
        video_noise = torch.randn(
            batch_size,
            self.config.student_candidate_count,
            *video_shape,
            device=device,
            dtype=dtype,
        )
        # Deferred execution reuses this exact noise after the video graph
        # has been backpropagated and released.
        student_videos = self.student_video_generator(
            video_noise,
            condition,
            initial_frame=batch.initial_frame,
        )

        frame_count = teacher_videos.shape[3]
        with torch.no_grad():
            signature_cache_result = self.teacher_signature_cache.get_or_compute(
                batch.sample_ids,
                teacher_videos,
                condition,
            )
            teacher_actions = signature_cache_result.signatures

            action_channel_mask = torch.zeros(
                self.config.action_dim,
                dtype=torch.bool,
                device=device,
            )
            action_channel_mask[list(self.config.used_action_channel_ids)] = True
            action_consistency_plan = ActionConsistencyPlan(
                pair=ActionTrainingPair(
                    video=batch.gt_video.detach(),
                    clean_action=batch.gt_action.detach(),
                    action_mask=(
                        batch.gt_action_mask
                        & action_channel_mask.view(1, -1, 1, 1, 1)
                    ).detach(),
                ),
                # Full-sequence action supervision starts at source frame 0.
                # Its causal history is provided by clean tokens within the
                # packed sequence, independently of the sampled video chunk.
                condition=ActionSignatureCondition(
                    text_emb=batch.text_emb.detach(),
                    frame_start=torch.zeros_like(batch.frame_start),
                    history_frame_start=torch.zeros_like(batch.frame_start),
                ),
            )

            response_noise = torch.randn(
                batch_size,
                self.config.action_dim,
                frame_count,
                self.config.action_per_frame,
                1,
                device=device,
                dtype=dtype,
            )
            action_timestep_ids = torch.randint(
                self.config.action_num_train_timesteps,
                (frame_count,),
                device="cpu",
            )

        reuse_all = self.config.execution_response_mode == "reuse_all"
        if reuse_all:
            response_batch = self.action_response_probe(
                teacher_videos,
                student_videos,
                teacher_actions,
                response_noise,
                action_timestep_ids,
                condition,
                track_student_grad=True,
            )
        else:
            with torch.no_grad():
                response_batch = self.action_response_probe(
                    teacher_videos,
                    student_videos,
                    teacher_actions,
                    response_noise,
                    action_timestep_ids,
                    condition,
                    track_student_grad=False,
                )

        with torch.no_grad():
            student_teacher_action = (
                self.action_response_metric.compute_student_teacher(
                    response_batch.teacher_responses,
                    response_batch.student_responses,
                    batch.action_valid_mask,
                )
            )

        feature_noise = torch.randn(
            batch_size,
            *video_shape,
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            teacher_features = self.video_feature_extractor(
                teacher_videos,
                feature_noise,
                condition,
            )
        student_features = self.video_feature_extractor(
            student_videos,
            feature_noise,
            condition,
        )

        feature_frames = teacher_features.shape[2]
        tokens_per_frame = teacher_features.shape[3]
        if feature_frames != batch.video_valid_frames.shape[1]:
            raise ValueError(
                "video feature frame count differs from the data valid mask"
            )
        video_valid_mask = batch.video_valid_frames[:, :, None].expand(
            batch_size,
            feature_frames,
            tokens_per_frame,
        )
        video_metric = self.video_metric(
            teacher_features,
            student_features,
            video_valid_mask,
        )
        # Matching depends only on attraction. Repulsion then compares
        # each pair using the negative candidate's matched action probe.
        positive_weights = self.video_objective.kernel.compute_positive_weights(
            video_metric.distances.student_teacher,
            student_teacher_action,
            self.bandwidths,
        )
        selected_teacher = ActionExecutionLoss.select_teachers(positive_weights)
        action_metric = ActionResponseMetricResult(
            student_teacher=student_teacher_action,
            student_student=self.action_response_metric.compute_student_student(
                response_batch.student_responses,
                selected_teacher,
                batch.action_valid_mask,
            ),
        )
        video_objective = self.video_objective(
            video_metric,
            action_metric,
            self.bandwidths,
            video_valid_mask=video_valid_mask,
            collect_diagnostics=collect_diagnostics,
            release_drift_fields=True,
        )
        video_loss = video_objective.loss
        combined_loss: Tensor | None = video_loss
        scalars = dict(video_objective.scalars)
        execution_result = None
        deferred_execution = None
        if self.config.execution_response_mode != "off":
            assert self.action_execution_loss is not None
            if reuse_all:
                selected_student_responses = (
                    self.action_execution_loss.select_student_responses(
                        response_batch.student_responses,
                        selected_teacher,
                    )
                )
                execution_result = self.action_execution_loss(
                    response_batch.teacher_responses,
                    selected_student_responses,
                    selected_teacher,
                    batch.action_valid_mask,
                )
                execution_contribution = (
                    self.config.execution_loss_weight * execution_result.loss
                )
                combined_loss = video_loss + execution_contribution
                scalars.update(
                    {
                        "loss/video_execution": execution_result.loss.detach(),
                        "loss/video_execution_contribution": (
                            execution_contribution.detach()
                        ),
                        "loss/video_total": combined_loss.detach(),
                    }
                )
            else:
                # Do not build the selected-response graph while the much
                # larger video-feature graph is alive. The trainer first
                # backpropagates video_loss, releases that graph, and then
                # regenerates the exact same students from video_noise.
                combined_loss = None
                deferred_execution = DeferredExecutionPlan(
                    video_noise=video_noise.detach(),
                    condition=self._detach_condition(condition),
                    initial_frame=(
                        None
                        if batch.initial_frame is None
                        else batch.initial_frame.detach()
                    ),
                    teacher_responses=(
                        response_batch.teacher_responses.detach()
                    ),
                    noisy_teacher_actions=(
                        response_batch.noisy_teacher_actions.detach()
                    ),
                    action_timestep_ids=(
                        response_batch.action_timestep_ids.detach()
                    ),
                    selected_teacher=selected_teacher.detach(),
                    action_valid_mask=batch.action_valid_mask.detach(),
                )

            scalars["execution/reuse_all"] = torch.tensor(
                float(reuse_all),
                device=device,
            )
            for teacher_index in range(self.config.teacher_candidate_count):
                scalars[
                    f"execution/selected_teacher_{teacher_index}_fraction"
                ] = (
                    selected_teacher.eq(teacher_index)
                    .float()
                    .mean()
                    .to(device)
                )

        cache_total = signature_cache_result.hits + signature_cache_result.misses
        scalars.update(
            {
                "cache/teacher_signature_hits": torch.tensor(
                    float(signature_cache_result.hits),
                    device=device,
                ),
                "cache/teacher_signature_misses": torch.tensor(
                    float(signature_cache_result.misses),
                    device=device,
                ),
                "cache/teacher_signature_hit_rate": torch.tensor(
                    signature_cache_result.hits / max(cache_total, 1),
                    device=device,
                ),
            }
        )

        if combined_loss is not None:
            scalars["loss/total"] = combined_loss.detach()
        scalars.update(
            {
                "response/timestep_id_mean": (
                    response_batch.action_timestep_ids.float().mean().to(device)
                ),
                "data/frame_start_mean": batch.frame_start.float().mean().detach(),
                "data/history_frames": torch.tensor(
                    0
                    if batch.history_video is None
                    else batch.history_video.shape[2],
                    dtype=torch.float32,
                    device=device,
                ),
            }
        )
        return TrainingStepResult(
            video_loss=video_loss,
            combined_loss=combined_loss,
            video_objective=video_objective,
            execution=execution_result,
            deferred_execution=deferred_execution,
            action_consistency_plan=action_consistency_plan,
            scalars=scalars,
        )

    __call__ = compute

    def compute_deferred_execution(
        self,
        plan: DeferredExecutionPlan,
    ) -> ActionExecutionLossResult:
        """Rebuild identical students and compute the selected response loss."""

        if self.config.execution_response_mode != "recompute_selected":
            raise RuntimeError(
                "deferred execution requires recompute_selected mode"
            )
        if self.action_execution_loss is None:
            raise RuntimeError("action execution loss is unavailable")

        student_videos = self.student_video_generator(
            plan.video_noise,
            plan.condition,
            initial_frame=plan.initial_frame,
        )
        selected_student_responses = (
            self.action_response_probe.compute_selected_student_responses(
                student_videos,
                plan.noisy_teacher_actions,
                plan.action_timestep_ids,
                plan.selected_teacher,
                plan.condition,
            )
        )
        return self.action_execution_loss(
            plan.teacher_responses,
            selected_student_responses,
            plan.selected_teacher,
            plan.action_valid_mask,
        )

    def compute_action_consistency(
        self,
        plan: ActionConsistencyPlan,
    ) -> ActionConsistencyLossResult:
        """Compute deferred action consistency/flow matching on the GT pair."""

        return self.action_consistency_loss(plan.pair, plan.condition)

    @staticmethod
    def _detach_condition(
        condition: ActionSignatureCondition,
    ) -> ActionSignatureCondition:
        """Keep only graph-free conditioning tensors for the second forward."""

        return ActionSignatureCondition(
            text_emb=condition.text_emb.detach(),
            frame_start=(
                condition.frame_start.detach()
                if isinstance(condition.frame_start, Tensor)
                else condition.frame_start
            ),
            history_video=(
                None
                if condition.history_video is None
                else condition.history_video.detach()
            ),
            history_action=(
                None
                if condition.history_action is None
                else condition.history_action.detach()
            ),
            history_frame_start=(
                condition.history_frame_start.detach()
                if isinstance(condition.history_frame_start, Tensor)
                else condition.history_frame_start
            ),
        )

    def _validate_batch(self, batch: TrainingBatch) -> None:
        expected_frame_start_shape = (batch.batch_size,)
        if tuple(batch.frame_start.shape) != expected_frame_start_shape:
            raise ValueError(
                f"frame_start must have shape {expected_frame_start_shape}"
            )
        local_min = batch.frame_start.min()
        local_max = batch.frame_start.max()
        if not torch.equal(local_min, local_max):
            raise ValueError("one local batch cannot mix frame_start values")
        if dist.is_available() and dist.is_initialized():
            global_min = local_min.clone()
            global_max = local_max.clone()
            dist.all_reduce(global_min, op=dist.ReduceOp.MIN)
            dist.all_reduce(global_max, op=dist.ReduceOp.MAX)
            if not torch.equal(global_min, global_max):
                raise RuntimeError(
                    "all ranks must use the same frame_start before FSDP forward"
                )

        expected_teacher_shape = (
            batch.batch_size,
            self.config.teacher_candidate_count,
        )
        if tuple(batch.teacher_videos.shape[:2]) != expected_teacher_shape:
            raise ValueError(
                f"teacher videos must begin with {expected_teacher_shape}"
            )
        if batch.teacher_videos.ndim != 6:
            raise ValueError("teacher videos must have shape [B,M,C,F,H,W]")
        if batch.teacher_videos.shape[3] != self.config.frame_chunk_size:
            raise ValueError("teacher videos contain the wrong frame count")
        if batch.gt_video.ndim != 5 or batch.gt_video.shape[2] < self.config.frame_chunk_size:
            raise ValueError("gt_video must contain a full GT sequence [B,C,F,H,W]")
        expected_gt_video = (
            batch.batch_size, batch.teacher_videos.shape[2],
            batch.gt_video.shape[2], *batch.teacher_videos.shape[-2:],
        )
        if tuple(batch.gt_video.shape) != expected_gt_video:
            raise ValueError(f"gt_video must have shape {expected_gt_video}")
        if tuple(batch.video_valid_frames.shape) != (
            batch.batch_size,
            self.config.frame_chunk_size,
        ):
            raise ValueError("video_valid_frames must have shape [B,F]")
        expected_action_mask = (
            batch.batch_size,
            self.config.action_dim,
            self.config.frame_chunk_size,
            self.config.action_per_frame,
            1,
        )
        expected_gt_action = (
            batch.batch_size, self.config.action_dim, batch.gt_video.shape[2],
            self.config.action_per_frame, 1,
        )
        if tuple(batch.gt_action.shape) != expected_gt_action:
            raise ValueError(f"gt_action must have shape {expected_gt_action}")
        if batch.gt_action_mask.shape != batch.gt_action.shape or batch.gt_action_mask.dtype != torch.bool:
            raise ValueError("gt_action_mask must be a Boolean mask matching the full GT action")
        if tuple(batch.action_valid_mask.shape) != expected_action_mask:
            raise ValueError(
                f"action_valid_mask must have shape {expected_action_mask}"
            )
