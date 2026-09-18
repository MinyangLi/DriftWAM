"""Distributed optimization loop for action-response-guided video drifting."""

from __future__ import annotations

import gc
import json
import logging
import math
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping
from uuid import uuid4

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..forwards.action_response import ActionResponseProbe
from ..geometry.action_response_metric import ActionResponseMetric
from ..geometry.bandwidth import (
    BandwidthCalibrationResult,
    BandwidthCalibrator,
    KernelBandwidths,
)
from .checkpoint_retention import prune_old_checkpoints
from .ema import update_ema
from .precision import TRAINING_NUMERICS, assert_fp32_model, assert_fp32_optimizer
from .resume_state import (
    RESUME_STATE_DIRNAME,
    capture_rng_state,
    load_resume_state,
    restore_rng_state,
    save_resume_state,
)
from ..training_config import TrainingConfig
from .teacher_signature_cache import TeacherSignatureCache
from .training_data import TrainingBatch
from .training_step import ActionConsistencyPlan, TrainingStep
from ..forwards.video_feature import VideoFeatureExtractor


logger = logging.getLogger(__name__)

def resolve_resume_checkpoint(config: TrainingConfig) -> Path | None:
    """Resolve the checkpoint root containing both student directories."""

    if config.resume_from_path is not None:
        return Path(config.resume_from_path).expanduser().resolve()
    if config.resume_from_step is not None:
        return (
            config.output_dir
            / "checkpoints"
            / f"step_{config.resume_from_step}"
        ).resolve()
    return None


def load_checkpoint_step(
    config: TrainingConfig,
    checkpoint_root: Path | None,
) -> int:
    """Read the completed optimizer-step count for resume."""

    if checkpoint_root is None:
        return 0
    state = _read_trainer_state(checkpoint_root)
    stored_step = state.get("step") if state is not None else None
    if stored_step is not None:
        stored_step = int(stored_step)
        if (
            config.resume_from_step is not None
            and stored_step != config.resume_from_step
        ):
            raise ValueError(
                f"checkpoint stores step {stored_step}, but "
                f"resume_from_step={config.resume_from_step}"
            )
        return stored_step
    if config.resume_from_step is not None:
        return config.resume_from_step
    if checkpoint_root.name.startswith("step_"):
        suffix = checkpoint_root.name.removeprefix("step_")
        if suffix.isdigit():
            return int(suffix)
    raise FileNotFoundError(
        f"cannot determine resume step from {checkpoint_root}"
    )


def load_checkpoint_bandwidths(
    checkpoint_root: Path | None,
) -> KernelBandwidths | None:
    """Load the fixed kernel bandwidth pair stored in a checkpoint."""

    if checkpoint_root is None:
        return None
    state = _read_trainer_state(checkpoint_root)
    if state is None or "bandwidths" not in state:
        return None
    return KernelBandwidths.from_state_dict(state["bandwidths"])


@torch.no_grad()
def calibrate_bandwidths(
    config: TrainingConfig,
    batch: TrainingBatch,
    action_response_probe: ActionResponseProbe,
    action_response_metric: ActionResponseMetric,
    video_feature_extractor: VideoFeatureExtractor,
    calibrator: BandwidthCalibrator,
    teacher_signature_cache: TeacherSignatureCache,
) -> BandwidthCalibrationResult:
    """Run the one-time teacher-only calibration batch from Section 3.3."""

    parameter = next(action_response_probe.teacher.parameters())
    device = parameter.device
    dtype = getattr(torch, config.param_dtype)
    batch = batch.to(device, dtype=dtype)
    condition = batch.condition()
    teacher_videos = batch.teacher_videos.detach()
    batch_size = teacher_videos.shape[0]
    frame_count = teacher_videos.shape[3]

    cached = teacher_signature_cache.get_or_compute(
        batch.sample_ids,
        teacher_videos,
        condition,
    )
    teacher_actions = cached.signatures

    response_noise = torch.randn(
        batch_size,
        config.action_dim,
        frame_count,
        config.action_per_frame,
        1,
        device=device,
        dtype=dtype,
    )
    action_timestep_ids = torch.randint(
        config.action_num_train_timesteps,
        (frame_count,),
        device="cpu",
    )
    # Probe every teacher video with every teacher action. The positive
    # matrix directly measures D(i,j) under candidate j's own probe, so
    # calibration follows the same directed rule without needing bandwidths.
    response_batch = action_response_probe(
        teacher_videos,
        teacher_videos,
        teacher_actions,
        response_noise,
        action_timestep_ids,
        condition,
    )
    teacher_action_distances = action_response_metric.compute_student_teacher(
        response_batch.teacher_responses,
        response_batch.student_responses,
        batch.action_valid_mask,
    )
    feature_noise = torch.randn(
        batch_size,
        *teacher_videos.shape[2:],
        device=device,
        dtype=dtype,
    )
    teacher_features = video_feature_extractor(
        teacher_videos,
        feature_noise,
        condition,
    )
    video_valid_mask = batch.video_valid_frames[:, :, None].expand(
        batch_size,
        teacher_features.shape[2],
        teacher_features.shape[3],
    )
    return calibrator(
        teacher_features,
        teacher_action_distances,
        video_valid_mask=video_valid_mask,
    )


class JointDistillationTrainer:
    """Own optimizer state and execute video-student/EMA updates."""

    def __init__(
        self,
        config: TrainingConfig,
        student: nn.Module,
        target_student: nn.Module,
        training_step: TrainingStep,
        train_loader: DataLoader[TrainingBatch],
        *,
        initial_step: int = 0,
        checkpoint_root: Path | None = None,
    ) -> None:
        self.config = config
        self.student = student
        self.target_student = target_student
        self.training_step = training_step
        self.train_loader = train_loader
        self.step = int(initial_step)
        self.epoch = 0
        self.batches_consumed_in_epoch = 0
        self._loader_iterator = None
        self._pending_rng_state: dict[str, Any] | None = None
        self._last_saved_step = -1
        self._memory_profile_logged = False
        self.device = next(student.parameters()).device

        parameters = [
            parameter
            for parameter in student.parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            raise ValueError("online student has no trainable parameters")
        assert_fp32_model(student, "online_student")
        assert_fp32_model(target_student, "ema_target")
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.optimizer_epsilon,
            weight_decay=config.weight_decay,
            fused=True,
            foreach=False,
        )
        self.optimizer.zero_grad(set_to_none=True)
        if checkpoint_root is not None:
            if config.save_optimizer_state:
                self._restore_training_state(checkpoint_root)
            elif config.rank == 0:
                logger.warning(
                    "Resuming model weights without optimizer/RNG state because "
                    "optimizer checkpointing is disabled"
                )
        if config.rank == 0:
            logger.info("Action consistency/FM supervision: %s", config.action_supervision_source)
        if config.offload_optimizer_state:
            self._move_optimizer_state(torch.device("cpu"))
            if config.rank == 0:
                logger.info("Optimizer state CPU offload is enabled")

        self.checkpoint_dir = config.output_dir / "checkpoints"
        if config.rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        self._wandb_run = None
        if config.enable_wandb and config.rank == 0:
            try:
                import wandb
            except ImportError:
                logger.warning("wandb is unavailable; continuing without it")
            else:
                self._wandb_run = wandb.init(
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    config=_jsonable(asdict(config)),
                    dir=str(config.output_dir),
                )

    def train(self) -> None:
        """Run until ``max_train_steps`` successful optimizer updates."""

        config = self.config
        progress = tqdm(
            total=config.max_train_steps,
            initial=self.step,
            desc="DriftWAM",
            disable=config.rank != 0,
            dynamic_ncols=True,
        )
        micro_index = 0
        invalid_accumulation = False
        scalar_sums: dict[str, float] = {}
        scalar_counts: dict[str, int] = {}

        if config.rank == 0:
            logger.info(
                "Starting action-response-guided distillation at step %d/%d; "
                "microbatch=%d, accumulation=%d, world_size=%d",
                self.step,
                config.max_train_steps,
                config.batch_size,
                config.gradient_accumulation_steps,
                config.world_size,
            )

        try:
            while self.step < config.max_train_steps:
                if micro_index == 0:
                    update_started_at = perf_counter()
                batch = self._next_batch()
                should_update = (
                    micro_index + 1
                    == config.gradient_accumulation_steps
                )
                deferred_mode = (
                    config.execution_response_mode == "recompute_selected"
                )
                profile_memory = (
                    not self._memory_profile_logged
                    and config.rank == 0
                    and self.device.type == "cuda"
                )
                if config.history_stress_test:
                    if micro_index == 0 and self.device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(self.device)
                elif profile_memory:
                    torch.cuda.reset_peak_memory_stats(self.device)
                # Defer FSDP gradient synchronization until the optimizer
                # microbatch. Exact-size causal caches leave enough headroom
                # for the full accumulated gradient on earlier microbatches.
                history_frames = (
                    0
                    if batch.history_video is None
                    else int(batch.history_video.shape[2])
                )
                action_frames = int(batch.gt_video.shape[2])
                largest_history = max(
                    history_frames, action_frames - config.frame_chunk_size,
                )
                sync_gradients = (
                    should_update
                    or (
                        deferred_mode
                        # Full GT lengths differ across ranks. Every rank must
                        # make the same FSDP synchronization decision.
                        and not self._all_ranks_true(torch.tensor(
                            largest_history <= config.max_unsynced_history_frames,
                            device=self.device,
                        ))
                    )
                )
                self._set_gradient_sync(sync_gradients)
                next_step = self.step + 1
                collect_kernel_diagnostics = (
                    should_update
                    and next_step % config.diagnostics_interval == 0
                )
                result = self.training_step(
                    batch,
                    collect_diagnostics=collect_kernel_diagnostics,
                )
                if profile_memory:
                    self._log_cuda_memory(
                        f"micro {micro_index + 1} primary forward "
                        f"(history={history_frames}, action_frames={action_frames}, sync={sync_gradients})"
                    )

                if result.deferred_execution is None:
                    if deferred_mode:
                        raise RuntimeError(
                            "recompute_selected did not return a deferred plan"
                        )
                    if result.combined_loss is None:
                        raise RuntimeError(
                            "non-deferred training result has no combined loss"
                        )
                    action_plan = result.action_consistency_plan
                    scalars = result.scalars
                    primary_loss = result.combined_loss
                    finite_loss = self._all_ranks_true(
                        torch.isfinite(primary_loss.detach()).all()
                    )
                    if not finite_loss:
                        invalid_accumulation = True
                    else:
                        (
                            primary_loss / config.gradient_accumulation_steps
                        ).backward()
                    detached_primary_loss = primary_loss.detach()
                    del result, batch, primary_loss
                    torch.cuda.empty_cache()

                    if finite_loss:
                        (
                            action_loss,
                            action_flow_loss,
                            action_contribution,
                            finite_action,
                        ) = self._backward_action_consistency(
                            action_plan,
                            sync_gradients=sync_gradients,
                            profile_memory=profile_memory,
                        )
                        if not finite_action:
                            invalid_accumulation = True
                        scalars.update(
                            {
                                "loss/action_consistency": action_loss,
                                "loss/action_consistency_contribution": (
                                    config.action_consistency_loss_weight
                                    * action_loss
                                ),
                                "loss/action_flow_matching": (
                                    action_flow_loss
                                ),
                                "loss/action_flow_matching_contribution": (
                                    config.action_flow_matching_loss_weight
                                    * action_flow_loss
                                ),
                                "loss/action_total": action_contribution,
                                "loss/total": (
                                    detached_primary_loss + action_contribution
                                ),
                            }
                        )
                    self._accumulate_scalars(
                        scalars, scalar_sums, scalar_counts
                    )
                    del action_plan, scalars, detached_primary_loss
                else:
                    if not deferred_mode or result.combined_loss is not None:
                        raise RuntimeError(
                            "invalid deferred execution result contract"
                        )
                    plan = result.deferred_execution
                    action_plan = result.action_consistency_plan
                    scalars = result.scalars
                    video_loss = result.video_loss
                    finite_video_loss = self._all_ranks_true(
                        torch.isfinite(video_loss.detach()).all()
                    )
                    if not finite_video_loss:
                        invalid_accumulation = True
                    else:
                        (
                            video_loss / config.gradient_accumulation_steps
                        ).backward()

                    detached_video_loss = video_loss.detach()
                    # Drop every reference to the video-feature graph before
                    # constructing the selected action-response graph.
                    del result, video_loss, batch
                    torch.cuda.empty_cache()
                    if profile_memory:
                        self._log_cuda_memory(
                            f"micro {micro_index + 1} video backward released"
                        )

                    finite_execution_loss = False
                    if finite_video_loss:
                        # Match the primary backward's accumulation policy.
                        self._set_gradient_sync(sync_gradients)
                        execution = (
                            self.training_step.compute_deferred_execution(plan)
                        )
                        if profile_memory:
                            self._log_cuda_memory(
                                f"micro {micro_index + 1} execution forward"
                            )
                        execution_contribution = (
                            config.execution_loss_weight * execution.loss
                        )
                        finite_execution_loss = self._all_ranks_true(
                            torch.isfinite(
                                execution_contribution.detach()
                            ).all()
                        )
                        if not finite_execution_loss:
                            invalid_accumulation = True
                        else:
                            (
                                execution_contribution
                                / config.gradient_accumulation_steps
                            ).backward()

                        detached_execution_loss = execution.loss.detach()
                        detached_execution_contribution = (
                            execution_contribution.detach()
                        )
                        scalars.update(
                            {
                                "loss/video_execution": (
                                    detached_execution_loss
                                ),
                                "loss/video_execution_contribution": (
                                    detached_execution_contribution
                                ),
                                "loss/video_total": (
                                    detached_video_loss
                                    + detached_execution_contribution
                                ),
                                "loss/total": (
                                    detached_video_loss
                                    + detached_execution_contribution
                                ),
                            }
                        )
                        del execution, execution_contribution
                        if profile_memory:
                            self._log_cuda_memory(
                                f"micro {micro_index + 1} execution backward"
                            )
                    torch.cuda.empty_cache()
                    if finite_video_loss and finite_execution_loss:
                        (
                            action_loss,
                            action_flow_loss,
                            action_contribution,
                            finite_action,
                        ) = self._backward_action_consistency(
                            action_plan,
                            sync_gradients=sync_gradients,
                            profile_memory=profile_memory,
                        )
                        if not finite_action:
                            invalid_accumulation = True
                        scalars.update(
                            {
                                "loss/action_consistency": action_loss,
                                "loss/action_consistency_contribution": (
                                    config.action_consistency_loss_weight
                                    * action_loss
                                ),
                                "loss/action_flow_matching": (
                                    action_flow_loss
                                ),
                                "loss/action_flow_matching_contribution": (
                                    config.action_flow_matching_loss_weight
                                    * action_flow_loss
                                ),
                                "loss/action_total": action_contribution,
                                "loss/total": (
                                    scalars["loss/total"] + action_contribution
                                ),
                            }
                        )
                    self._accumulate_scalars(
                        scalars,
                        scalar_sums,
                        scalar_counts,
                    )
                    del plan, action_plan, scalars, detached_video_loss

                if profile_memory and should_update:
                    self._memory_profile_logged = True
                micro_index += 1
                if not should_update:
                    continue

                update_succeeded = False
                total_grad_norm = torch.tensor(
                    float("nan"),
                    device=self.device,
                )
                clipping_factor = torch.tensor(
                    0.0,
                    device=self.device,
                )
                if not invalid_accumulation:
                    self._set_learning_rate(next_step)
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.student.parameters(),
                        config.max_grad_norm,
                    )
                    finite_grad = self._all_ranks_true(
                        torch.isfinite(total_grad_norm.detach()).all()
                    )
                    if finite_grad:
                        clipping_factor = torch.clamp(
                            config.max_grad_norm
                            / (total_grad_norm.detach().float() + 1e-12),
                            max=1.0,
                        )
                        if config.offload_optimizer_state:
                            torch.cuda.empty_cache()
                            self._move_optimizer_state(self.device)
                        try:
                            self.optimizer.step()
                            if not getattr(self, "_optimizer_precision_checked", False):
                                assert_fp32_optimizer(self.optimizer)
                                self._optimizer_precision_checked = True
                        finally:
                            if config.offload_optimizer_state:
                                self._move_optimizer_state(
                                    torch.device("cpu")
                                )
                                torch.cuda.empty_cache()
                        update_ema(
                            self.target_student,
                            self.student,
                            decay=config.ema_decay,
                        )
                        update_succeeded = True

                self.optimizer.zero_grad(set_to_none=True)
                micro_index = 0
                invalid_accumulation = False

                if not update_succeeded:
                    if config.history_stress_test:
                        raise RuntimeError(
                            "History stress test failed: non-finite loss or gradients"
                        )
                    if config.rank == 0:
                        logger.warning(
                            "Skipped an optimizer update because loss or "
                            "gradients were non-finite"
                        )
                    scalar_sums.clear()
                    scalar_counts.clear()
                    continue

                self.step = next_step
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                update_seconds = perf_counter() - update_started_at
                if config.rank == 0:
                    logger.info("Optimizer step %d duration_seconds=%.3f", self.step, update_seconds)
                progress.update(1)
                local_total_loss = (
                    scalar_sums.get("loss/total", float("nan"))
                    / max(scalar_counts.get("loss/total", 1), 1)
                )
                progress.set_postfix(
                    loss=f"{local_total_loss:.4f}",
                    grad=f"{_scalar_value(total_grad_norm):.3f}",
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                )

                should_log = (
                    config.history_stress_test
                    or self.step % config.log_interval == 0
                    or collect_kernel_diagnostics
                )
                if should_log:
                    logs = self._reduce_scalars(
                        scalar_sums,
                        scalar_counts,
                    )
                    logs.update(
                        {
                            "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                            "timing/optimizer_step_seconds": update_seconds,
                            "gradient/total_norm_before_clip": _scalar_value(
                                total_grad_norm
                            ),
                            "gradient/clipping_factor": _scalar_value(
                                clipping_factor
                            ),
                        }
                    )
                    if config.history_stress_test and self.device.type == "cuda":
                        peaks = torch.tensor(
                            [torch.cuda.max_memory_allocated(self.device),
                             torch.cuda.max_memory_reserved(self.device)],
                            device=self.device, dtype=torch.float64,
                        )
                        if dist.is_available() and dist.is_initialized():
                            dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
                        allocated, reserved = (peaks / 1024**3).tolist()
                        logs["memory/peak_allocated_gib"] = allocated
                        logs["memory/peak_reserved_gib"] = reserved
                        if config.rank == 0:
                            logger.info(
                                "Stress update %d peak across all ranks, including "
                                "optimizer/EMA: allocated=%.2f GiB reserved=%.2f GiB",
                                self.step, allocated, reserved,
                            )
                    self._log(logs)
                scalar_sums.clear()
                scalar_counts.clear()

                if self.step % config.gc_interval == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                if self.step % config.save_interval == 0:
                    self.save_checkpoint()
            if config.save_final_checkpoint and self._last_saved_step != self.step:
                self.save_checkpoint()
            if config.rank == 0:
                logger.info("Training completed at optimizer step %d", self.step)
        finally:
            progress.close()
            if self._wandb_run is not None:
                self._wandb_run.finish()

    def _backward_action_consistency(
        self,
        plan: ActionConsistencyPlan,
        *,
        sync_gradients: bool,
        profile_memory: bool,
    ) -> tuple[Tensor, Tensor, Tensor, bool]:
        """Backpropagate consistency plus native action flow matching."""

        self._set_gradient_sync(sync_gradients)
        result = self.training_step.compute_action_consistency(plan)
        consistency_contribution = (
            self.config.action_consistency_loss_weight
            * result.consistency_loss
        )
        flow_matching_contribution = (
            self.config.action_flow_matching_loss_weight
            * result.flow_matching_loss
        )
        contribution = consistency_contribution + flow_matching_contribution
        finite = self._all_ranks_true(
            torch.isfinite(contribution.detach()).all()
        )
        if finite:
            (
                contribution / self.config.gradient_accumulation_steps
            ).backward()
        detached_loss = result.consistency_loss.detach()
        detached_flow_loss = result.flow_matching_loss.detach()
        detached_contribution = contribution.detach()
        del (
            result,
            consistency_contribution,
            flow_matching_contribution,
            contribution,
        )
        if profile_memory:
            self._log_cuda_memory("action objective backward")
        return (
            detached_loss,
            detached_flow_loss,
            detached_contribution,
            finite,
        )

    def _log_cuda_memory(self, stage: str) -> None:
        """Log rank-zero allocator state for the first optimizer step."""

        gibibyte = float(1024**3)
        logger.info(
            "CUDA memory after %s: allocated=%.2f GiB reserved=%.2f GiB "
            "peak_allocated=%.2f GiB peak_reserved=%.2f GiB",
            stage,
            torch.cuda.memory_allocated(self.device) / gibibyte,
            torch.cuda.memory_reserved(self.device) / gibibyte,
            torch.cuda.max_memory_allocated(self.device) / gibibyte,
            torch.cuda.max_memory_reserved(self.device) / gibibyte,
        )

    def _move_optimizer_state(self, device: torch.device) -> None:
        """Move AdamW state without changing parameters or accumulated grads."""

        for state in self.optimizer.state.values():
            for name, value in tuple(state.items()):
                if isinstance(value, Tensor):
                    state[name] = value.to(
                        device=device,
                        non_blocking=False,
                    )

    def save_checkpoint(self) -> Path:
        """Atomically publish model, optimizer, RNG, and checkpoint markers."""

        save_started = perf_counter()
        checkpoint_root = self.checkpoint_dir / f"step_{self.step}"
        temporary_value: list[str | None] = [None]
        error = None
        if self.config.rank == 0:
            if checkpoint_root.exists():
                error = (
                    f"checkpoint already exists: {checkpoint_root}; use a new "
                    "output directory or resume beyond that step"
                )
            else:
                temporary_value[0] = str(
                    self.checkpoint_dir
                    / f".step_{self.step}.{uuid4().hex}.tmp"
                )

        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(temporary_value, src=0)
        self._raise_distributed_error(error)
        if temporary_value[0] is None:
            raise RuntimeError("rank 0 did not provide a temporary checkpoint path")
        temporary_root = Path(temporary_value[0])
        rng_state = None
        if self.config.save_optimizer_state:
            rng_state = capture_rng_state(self.device, self.train_loader)

        # User-selected disk policy: release old checkpoint space first. Model,
        # EMA, optimizer and RNG remain in memory while the new save is written.
        # A failure after deletion can leave no recoverable checkpoint on disk.
        self._delete_checkpoints_before_save(checkpoint_root)
        try:
            for name, model in (
                ("online_student", self.student),
                ("target_student", self.target_student),
            ):
                self._save_model(
                    model,
                    temporary_root / name / "transformer",
                )
        except BaseException:
            self._cleanup_temporary_checkpoint(temporary_root)
            raise

        if self.config.save_optimizer_state:
            error = None
            try:
                save_resume_state(
                    self.student,
                    self.optimizer,
                    temporary_root,
                    rng_state,
                )
            except Exception as exc:
                error = f"failed to save optimizer/RNG state: {exc}"
            self._raise_checkpoint_error(error, temporary_root)

        error = None
        if self.config.rank == 0:
            try:
                _atomic_json_dump(
                    {
                        "step": self.step,
                        "epoch": self.epoch,
                        "batches_consumed_in_epoch": (
                            self.batches_consumed_in_epoch
                        ),
                        "world_size": self.config.world_size,
                        "batch_size": self.config.batch_size,
                        "gradient_accumulation_steps": (
                            self.config.gradient_accumulation_steps
                        ),
                        "seed": self.config.seed,
                        "execution_response_mode": (
                            self.config.execution_response_mode
                        ),
                        "execution_loss_weight": (
                            self.config.execution_loss_weight
                        ),
                        "action_consistency_loss_weight": (
                            self.config.action_consistency_loss_weight
                        ),
                        "action_flow_matching_loss_weight": (
                            self.config.action_flow_matching_loss_weight
                        ),
                        "video_drifting_loss_weight": (
                            self.config.video_drifting_loss_weight
                        ),
                        "beta_rep": self.config.beta_rep,
                        "ema_decay": self.config.ema_decay,
                        "learning_rate": self.config.learning_rate,
                        "warmup_steps": self.config.warmup_steps,
                        "beta1": self.config.beta1,
                        "beta2": self.config.beta2,
                        "weight_decay": self.config.weight_decay,
                        "optimizer_epsilon": self.config.optimizer_epsilon,
                        "max_grad_norm": self.config.max_grad_norm,
                        "teacher_signature_noise_seed": (
                            self.config.teacher_signature_noise_seed
                        ),
                        "bandwidths": self.training_step.bandwidths.state_dict(),
                        "training_numerics": TRAINING_NUMERICS,
                        "param_dtype": self.config.param_dtype,
                        "action_supervision_source": self.config.action_supervision_source,
                    },
                    temporary_root / "trainer_state.json",
                )
                _atomic_json_dump(
                    _jsonable(asdict(self.config)),
                    temporary_root / "training_config.json",
                )
                temporary_root.replace(checkpoint_root)
            except Exception as exc:
                error = f"failed to publish checkpoint directory: {exc}"
        self._raise_checkpoint_error(error, temporary_root)

        marker_path = checkpoint_root / "CHECKPOINT_COMPLETE"
        error = None
        if self.config.rank == 0:
            try:
                initialization = (
                    "flash-wam-500"
                    if self.config.student_init_source == "flash_wam"
                    else "teacher-one-step"
                )
                _atomic_json_dump(
                    {
                        "checkpoint_marker_version": 2,
                        "checkpoint_step": self.step,
                        "default_eval_weight_kind": "ema_target",
                        "online_student_transformer_relative_path": (
                            "online_student/transformer"
                        ),
                        "ema_target_transformer_relative_path": (
                            "target_student/transformer"
                        ),
                        "available_weight_kinds": {
                            "online_student": "online_student/transformer",
                            "ema_target": "target_student/transformer",
                        },
                        "training_method": (
                            "action_response_guided_drifting_with_"
                            "action_consistency_and_flow_matching"
                        ),
                        "initialization": initialization,
                        "initialization_mode": (
                            self.config.student_init_source
                        ),
                        "initialization_source": str(
                            self.config.selected_student_init_path
                        ),
                        "exact_resume_available": (
                            self.config.save_optimizer_state
                        ),
                    },
                    marker_path,
                )
                if self.config.save_optimizer_state:
                    _atomic_json_dump(
                        {
                            "checkpoint_step": self.step,
                            "resume_state_relative_path": RESUME_STATE_DIRNAME,
                            "world_size": self.config.world_size,
                            "rng_rank_count": self.config.world_size,
                        },
                        checkpoint_root / "RESUME_STATE_COMPLETE",
                    )
            except Exception as exc:
                error = f"failed to write checkpoint completion marker: {exc}"
        self._raise_distributed_error(error)

        self._last_saved_step = self.step
        if self.config.rank == 0:
            logger.info(
                "Saved complete checkpoint to %s; delete_and_save_seconds=%.3f",
                checkpoint_root, perf_counter() - save_started,
            )
        return checkpoint_root

    def _cleanup_temporary_checkpoint(self, temporary_root: Path) -> None:
        """Best-effort cleanup of an unpublished checkpoint directory."""

        if self.config.rank != 0 or not temporary_root.exists():
            return
        try:
            shutil.rmtree(temporary_root)
            logger.warning(
                "Removed incomplete checkpoint directory %s",
                temporary_root,
            )
        except OSError:
            logger.exception(
                "Failed to remove incomplete checkpoint directory %s",
                temporary_root,
            )

    def _raise_checkpoint_error(
        self,
        error: str | None,
        temporary_root: Path,
    ) -> None:
        try:
            self._raise_distributed_error(error)
        except BaseException:
            self._cleanup_temporary_checkpoint(temporary_root)
            raise

    def _restore_training_state(self, checkpoint_root: Path) -> None:
        state = _read_trainer_state(checkpoint_root)
        if state is None:
            raise FileNotFoundError(
                f"trainer_state.json is missing from {checkpoint_root}"
            )
        expected = {
            "training_numerics": TRAINING_NUMERICS,
            "param_dtype": self.config.param_dtype,
            "action_supervision_source": self.config.action_supervision_source,
            "step": self.step,
            "world_size": self.config.world_size,
            "batch_size": self.config.batch_size,
            "gradient_accumulation_steps": (
                self.config.gradient_accumulation_steps
            ),
            "seed": self.config.seed,
            "execution_response_mode": self.config.execution_response_mode,
            "execution_loss_weight": self.config.execution_loss_weight,
            "action_consistency_loss_weight": (
                self.config.action_consistency_loss_weight
            ),
            "action_flow_matching_loss_weight": (
                self.config.action_flow_matching_loss_weight
            ),
            "video_drifting_loss_weight": (
                self.config.video_drifting_loss_weight
            ),
            "beta_rep": self.config.beta_rep,
            "ema_decay": self.config.ema_decay,
            "learning_rate": self.config.learning_rate,
            "warmup_steps": self.config.warmup_steps,
            "beta1": self.config.beta1,
            "beta2": self.config.beta2,
            "weight_decay": self.config.weight_decay,
            "optimizer_epsilon": self.config.optimizer_epsilon,
            "max_grad_norm": self.config.max_grad_norm,
            "teacher_signature_noise_seed": (
                self.config.teacher_signature_noise_seed
            ),
        }
        mismatches = {
            name: {"checkpoint": state.get(name), "configured": value}
            for name, value in expected.items()
            if state.get(name) != value
        }
        if mismatches:
            raise ValueError(
                f"resume-state training contract differs: {mismatches}"
            )
        self.epoch = int(state.get("epoch", 0))
        self.batches_consumed_in_epoch = int(
            state.get("batches_consumed_in_epoch", 0)
        )
        if not 0 <= self.batches_consumed_in_epoch <= len(self.train_loader):
            raise ValueError(
                "saved batches_consumed_in_epoch is outside the loader"
            )
        self._pending_rng_state = load_resume_state(
            self.student,
            self.optimizer,
            checkpoint_root,
        )
        assert_fp32_optimizer(self.optimizer)
        if self.config.rank == 0:
            logger.info(
                "Restored optimizer/RNG state at step=%d epoch=%d batch=%d",
                self.step,
                self.epoch,
                self.batches_consumed_in_epoch,
            )

    def _delete_checkpoints_before_save(self, checkpoint_root: Path) -> None:
        """Delete on rank 0, then synchronize success before any rank writes."""

        error = None
        if self.config.rank == 0:
            try:
                prune_old_checkpoints(
                    checkpoint_root, self.config.retain_checkpoint_count,
                )
            except Exception as exc:
                error = f"failed to delete old checkpoints before saving: {exc}"
        self._raise_distributed_error(error)

    def _save_model(self, model: nn.Module, destination: Path) -> None:
        error = None
        try:
            assert_fp32_model(model)
            try:
                from safetensors.torch import save_file
            except ImportError:
                save_file = None
            state_dict = get_model_state_dict(
                model,
                options=StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=True,
                ),
            )
            if self.config.rank == 0:
                destination.mkdir(parents=True, exist_ok=True)
                serialized = {
                    name: value.contiguous()
                    for name, value in state_dict.items()
                }
                if save_file is None:
                    torch.save(
                        serialized,
                        destination / "diffusion_pytorch_model.bin",
                    )
                    logger.warning(
                        "safetensors is unavailable; saved PyTorch .bin checkpoint"
                    )
                else:
                    save_file(
                        serialized,
                        destination / "diffusion_pytorch_model.safetensors",
                    )
                model_config = dict(model.config)
                model_config.pop("_name_or_path", None)
                model_config["torch_dtype"] = "float32"
                _atomic_json_dump(
                    _jsonable(model_config),
                    destination / "config.json",
                )
        except Exception as exc:  # synchronize failure across ranks
            error = f"failed to save model at {destination}: {exc}"
        finally:
            if "state_dict" in locals():
                del state_dict
            if "serialized" in locals():
                del serialized
        self._raise_distributed_error(error)

    def _next_batch(self) -> TrainingBatch:
        if self._loader_iterator is None:
            self._set_sampler_epoch()
            self._loader_iterator = iter(self.train_loader)
            if self._pending_rng_state is not None:
                for _ in range(self.batches_consumed_in_epoch):
                    try:
                        next(self._loader_iterator)
                    except StopIteration as exc:
                        raise RuntimeError(
                            "saved batch position exceeds the resumed epoch"
                        ) from exc
                restore_rng_state(
                    self._pending_rng_state,
                    self.device,
                    self.train_loader,
                )
                self._pending_rng_state = None
        try:
            batch = next(self._loader_iterator)
            self.batches_consumed_in_epoch += 1
            return batch
        except StopIteration:
            self.epoch += 1
            self.batches_consumed_in_epoch = 0
            self._set_sampler_epoch()
            self._loader_iterator = iter(self.train_loader)
            batch = next(self._loader_iterator)
            self.batches_consumed_in_epoch = 1
            return batch

    def _set_sampler_epoch(self) -> None:
        sampler = self.train_loader.sampler
        set_epoch = getattr(sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(self.epoch)

    def _set_gradient_sync(self, enabled: bool) -> None:
        setter = getattr(self.student, "set_requires_gradient_sync", None)
        if setter is not None:
            setter(enabled)

    def _set_learning_rate(self, optimizer_step: int) -> None:
        if self.config.warmup_steps == 0:
            scale = 1.0
        else:
            scale = min(
                float(optimizer_step) / self.config.warmup_steps,
                1.0,
            )
        learning_rate = self.config.learning_rate * scale
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    @staticmethod
    def _accumulate_scalars(
        values: Mapping[str, Tensor],
        sums: dict[str, float],
        counts: dict[str, int],
    ) -> None:
        for name, value in values.items():
            if value.numel() != 1:
                raise ValueError(f"training scalar {name!r} is not scalar")
            sums[name] = sums.get(name, 0.0) + float(value.detach().item())
            counts[name] = counts.get(name, 0) + 1

    def _reduce_scalars(
        self,
        sums: Mapping[str, float],
        counts: Mapping[str, int],
    ) -> dict[str, float]:
        names = tuple(sorted(sums))
        payload = torch.tensor(
            [[sums[name], float(counts[name])] for name in names],
            dtype=torch.float64,
            device=self.device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(payload, op=dist.ReduceOp.SUM)
        return {
            name: float(payload[index, 0].item() / payload[index, 1].item())
            for index, name in enumerate(names)
        }

    def _log(self, values: Mapping[str, float]) -> None:
        error = None
        if self.config.rank == 0:
            try:
                logger.info(
                    "step=%d total=%.6f drifting=%.6f execution=%.6f "
                    "action_consistency=%.6f action_flow=%.6f "
                    "grad_preclip=%.4f clip_factor=%.4f lr=%.3e "
                    "teacher_signature_cache_hit=%.3f",
                    self.step,
                    values.get("loss/total", float("nan")),
                    values.get(
                        "loss/video_drifting_contribution",
                        float("nan"),
                    ),
                    values.get(
                        "loss/video_execution_contribution",
                        float("nan"),
                    ),
                    values.get("loss/action_consistency", float("nan")),
                    values.get("loss/action_flow_matching", float("nan")),
                    values.get(
                        "gradient/total_norm_before_clip",
                        float("nan"),
                    ),
                    values.get("gradient/clipping_factor", float("nan")),
                    values["train/learning_rate"],
                    values.get(
                        "cache/teacher_signature_hit_rate",
                        float("nan"),
                    ),
                )
                if "coverage/effective_teachers/mean" in values:
                    logger.info(
                        "diagnostics step=%d drift_rms=%.6g "
                        "positive_effective_neighbors=%.4f "
                        "negative_effective_neighbors=%.4f "
                        "effective_teachers=%.4f st_joint_median=%.4f",
                        self.step,
                        values.get("video/drift_rms", float("nan")),
                        values.get(
                            "weights/positive/effective_neighbors/mean",
                            float("nan"),
                        ),
                        values.get(
                            "weights/negative/effective_neighbors/mean",
                            float("nan"),
                        ),
                        values.get(
                            "coverage/effective_teachers/mean",
                            float("nan"),
                        ),
                        values.get(
                            "distance/student_teacher/joint/median",
                            float("nan"),
                        ),
                    )
                self._append_metrics_jsonl(values)
                if self._wandb_run is not None:
                    self._wandb_run.log(dict(values), step=self.step)
            except Exception as exc:
                error = f"failed to persist training diagnostics: {exc}"
        self._raise_distributed_error(error)

    def _append_metrics_jsonl(self, values: Mapping[str, float]) -> None:
        record: dict[str, Any] = {
            "step": self.step,
            "wall_time_utc": datetime.now(timezone.utc).isoformat(),
        }
        for name, value in values.items():
            scalar = float(value)
            record[name] = scalar if math.isfinite(scalar) else None
        with (self.config.output_dir / "metrics.jsonl").open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                json.dumps(record, allow_nan=False, sort_keys=True) + "\n"
            )

    def _all_ranks_true(self, value: Tensor) -> bool:
        flag = _local_tensor(value.detach()).to(
            device=self.device,
            dtype=torch.int32,
        ).reshape(())
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    @staticmethod
    def _raise_distributed_error(error: str | None) -> None:
        if dist.is_available() and dist.is_initialized():
            errors: list[str | None] = [None] * dist.get_world_size()
            dist.all_gather_object(errors, error)
            failures = [message for message in errors if message is not None]
            if failures:
                raise RuntimeError("; ".join(failures))
        elif error is not None:
            raise RuntimeError(error)


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _read_trainer_state(checkpoint_root: Path) -> Mapping[str, Any] | None:
    path = checkpoint_root / "trainer_state.json"
    if not path.is_file():
        return None
    return _read_json_object(path)


def _local_tensor(tensor: Tensor) -> Tensor:
    return getattr(tensor, "_local_tensor", tensor)


def _scalar_value(value: Tensor) -> float:
    # FSDP2 norm scalars can carry a pending cross-rank reduction. Every
    # caller runs on all ranks; materialize the global value before logging.
    value = value.detach()
    full_tensor = getattr(value, "full_tensor", None)
    if callable(full_tensor):
        value = full_tensor()
    return float(value.item())


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _atomic_json_dump(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)
