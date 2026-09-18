"""Assemble and launch action-response-guided video drifting."""

from __future__ import annotations

import argparse
import logging
import os
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .forwards.action_response import ActionResponseProbe
from .config import EXECUTION_RESPONSE_MODES
from .geometry.action_response_metric import ActionResponseMetric
from .geometry.bandwidth import BandwidthCalibrator, KernelBandwidths
from .model_setup import (
    configure_frozen_teacher,
    load_online_student,
    load_target_student,
)
from .runtime import activate_lingbot_va
from .forwards.student_video import StudentVideoGenerator
from .engine.trainer import (
    JointDistillationTrainer,
    calibrate_bandwidths,
    load_checkpoint_bandwidths,
    load_checkpoint_step,
    resolve_resume_checkpoint,
)
from .engine.teacher_signature_cache import TeacherSignatureCache
from .run_manifest import distributed_preflight, write_resolved_config
from .training_config import TrainingConfig
from .engine.training_data import build_training_dataloader
from .engine.training_step import TrainingStep
from .forwards.video_feature import VideoFeatureExtractor
from .geometry.video_metric import VideoMetric
from .objectives.video_objective import VideoObjective
from .objectives.action_execution_loss import ActionExecutionLoss
from .objectives.action_consistency_loss import ActionConsistencyLoss


logger = logging.getLogger(__name__)

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Action-response-guided video drifting distillation"
    )
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--empty-emb-path", type=Path)
    parser.add_argument("--teacher-video-bank-path", type=Path)
    parser.add_argument("--teacher-signature-cache-path", type=Path)
    parser.add_argument("--teacher-model-path", type=Path)
    parser.add_argument("--student-init-path", type=Path)
    parser.add_argument(
        "--student-init-source",
        choices=("flash_wam", "teacher"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from-path", type=Path)
    parser.add_argument("--resume-from-step", type=int)
    parser.add_argument("--experiment-name")
    parser.add_argument("--expected-dataset-count", type=int)
    parser.add_argument("--expected-source-count", type=int)
    parser.add_argument("--expected-world-size", type=int)
    parser.add_argument("--expected-global-batch-size", type=int)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument(
        "--history-stress-test", action="store_true",
        help="Exercise longest and unsynced-boundary histories in 1-3 updates",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--teacher-signature-noise-seed", type=int)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--diagnostics-interval", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--video-bandwidth", type=float)
    parser.add_argument("--action-bandwidth", type=float)
    parser.add_argument(
        "--execution-response-mode",
        choices=tuple(sorted(EXECUTION_RESPONSE_MODES)),
    )
    parser.add_argument("--execution-loss-weight", type=float)
    parser.add_argument("--action-consistency-loss-weight", type=float)
    parser.add_argument("--action-flow-matching-loss-weight", type=float)
    parser.add_argument("--disable-optimizer-checkpoint", action="store_true")
    parser.add_argument(
        "--disable-optimizer-state-offload",
        action="store_true",
    )
    parser.add_argument("--skip-final-checkpoint", action="store_true")
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainingConfig:
    base = TrainingConfig()
    overrides = {
        "dataset_path": args.dataset_path,
        "empty_emb_path": args.empty_emb_path,
        "teacher_video_bank_path": args.teacher_video_bank_path,
        "teacher_signature_cache_path": args.teacher_signature_cache_path,
        "teacher_model_path": args.teacher_model_path,
        "student_init_path": args.student_init_path,
        "student_init_source": args.student_init_source,
        "output_dir": args.output_dir,
        "resume_from_path": args.resume_from_path,
        "resume_from_step": args.resume_from_step,
        "experiment_name": args.experiment_name,
        "expected_dataset_count": args.expected_dataset_count,
        "expected_source_count": args.expected_source_count,
        "expected_world_size": args.expected_world_size,
        "expected_global_batch_size": args.expected_global_batch_size,
        "max_train_steps": args.max_train_steps,
        "history_stress_test": args.history_stress_test,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_steps": args.warmup_steps,
        "seed": args.seed,
        "teacher_signature_noise_seed": args.teacher_signature_noise_seed,
        "save_interval": args.save_interval,
        "diagnostics_interval": args.diagnostics_interval,
        "num_workers": args.num_workers,
        "execution_response_mode": args.execution_response_mode,
        "execution_loss_weight": args.execution_loss_weight,
        "action_consistency_loss_weight": (
            args.action_consistency_loss_weight
        ),
        "action_flow_matching_loss_weight": (
            args.action_flow_matching_loss_weight
        ),
    }
    if args.disable_optimizer_checkpoint:
        overrides["save_optimizer_state"] = False
    if args.disable_optimizer_state_offload:
        overrides["offload_optimizer_state"] = False
    if args.skip_final_checkpoint:
        overrides["save_final_checkpoint"] = False
    if args.disable_wandb:
        overrides["enable_wandb"] = False
    config = replace(
        base,
        **{name: value for name, value in overrides.items() if value is not None},
    )

    if (args.video_bandwidth is None) != (args.action_bandwidth is None):
        raise ValueError(
            "--video-bandwidth and --action-bandwidth must be supplied together"
        )
    if args.video_bandwidth is not None:
        config.set_kernel_bandwidths(
            video=args.video_bandwidth,
            action=args.action_bandwidth,
        )
    config.validate_runtime_contract()
    return config


@torch.no_grad()
def assert_fresh_students_identical(student, target_student) -> None:
    """Verify exact local shards before the first optimizer update."""

    online_parameters = list(student.parameters())
    target_parameters = list(target_student.parameters())
    mismatch = None
    if len(online_parameters) != len(target_parameters):
        mismatch = (
            f"parameter counts differ: online={len(online_parameters)} "
            f"target={len(target_parameters)}"
        )
    else:
        for index, (online, target) in enumerate(
            zip(online_parameters, target_parameters, strict=True)
        ):
            online_local = getattr(online, "_local_tensor", online)
            target_local = getattr(target, "_local_tensor", target)
            if (
                online_local.shape != target_local.shape
                or online_local.dtype != target_local.dtype
                or not torch.equal(online_local, target_local)
            ):
                mismatch = f"parameter shard {index} differs"
                break

    if dist.is_available() and dist.is_initialized():
        mismatches: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(mismatches, mismatch)
        failures = [
            f"rank {rank}: {message}"
            for rank, message in enumerate(mismatches)
            if message is not None
        ]
    else:
        failures = [] if mismatch is None else [mismatch]
    if failures:
        raise RuntimeError(
            "online and EMA students are not identical at initialization: "
            + "; ".join(failures)
        )
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info("Verified identical online and EMA initialization")


def initialize_distributed(config: TrainingConfig) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("training requires CUDA")
    torch.cuda.set_device(config.local_rank)
    launched_by_torchrun = "RANK" in os.environ
    if launched_by_torchrun and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=config.rank,
            world_size=config.world_size,
        )
    return torch.device("cuda", config.local_rank)


def configure_models(
    config: TrainingConfig,
    device: torch.device,
    checkpoint_root: Path | None,
):
    activate_lingbot_va(config.lingbot_va_root)
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from wan_va.distributed.util import _configure_model
    from .forwards.activation_checkpointing import apply_packed_activation_checkpointing

    dtype = _DTYPES[config.param_dtype]

    fp32_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        cast_forward_inputs=False,
    )
    student_policy = MixedPrecisionPolicy(
        param_dtype=dtype,
        reduce_dtype=torch.float32,
        cast_forward_inputs=False,
    )

    def shard_student(model, *, reshard_after_forward: bool):
        # All persistent parameters are FP32. Timestep embedders also compute
        # in FP32; the remaining groups use the configured forward dtype.
        for condition_embedder in (
            model.condition_embedder,
            model.condition_embedder_action,
        ):
            fully_shard(
                condition_embedder.time_embedder,
                mp_policy=fp32_policy,
                reshard_after_forward=reshard_after_forward,
            )
        fsdp_config = {
            "mp_policy": student_policy,
            "reshard_after_forward": reshard_after_forward,
        }
        for block in model.blocks:
            fully_shard(block.attn1, **fsdp_config)
            fully_shard(block.attn2, **fsdp_config)
            fully_shard(block.ffn, **fsdp_config)
            fully_shard(block, **fsdp_config)
        fully_shard(model, **fsdp_config)
        return model

    logger.info("Loading frozen teacher")
    teacher = configure_frozen_teacher(config, device=device)

    online_config = config
    target_config = config
    if checkpoint_root is not None:
        online_path = checkpoint_root / "online_student"
        target_path = checkpoint_root / "target_student"
        if not online_path.is_dir() or not target_path.is_dir():
            raise FileNotFoundError(
                "resume checkpoint must contain online_student and "
                f"target_student: {checkpoint_root}"
            )
        online_config = replace(
            config,
            student_init_source="flash_wam",
            student_init_path=online_path,
        )
        target_config = replace(
            config,
            student_init_source="flash_wam",
            student_init_path=target_path,
        )

    logger.info("Loading online video student")
    student = load_online_student(online_config, device="cpu")
    apply_packed_activation_checkpointing(student)
    student = _configure_model(
        student,
        shard_fn=lambda model: shard_student(model, reshard_after_forward=True),
        param_dtype=torch.float32,
        device=device,
        eval_mode=False,
    )
    student.train().requires_grad_(True)

    logger.info("Loading frozen EMA target student")
    target_student = load_target_student(target_config, device="cpu")
    apply_packed_activation_checkpointing(target_student)
    target_student = _configure_model(
        target_student,
        shard_fn=lambda model: shard_student(
            model,
            reshard_after_forward=True,
        ),
        param_dtype=torch.float32,
        device=device,
        eval_mode=True,
    )
    target_student.eval().requires_grad_(False)
    from .engine.precision import assert_fp32_model
    assert_fp32_model(student, "online_student")
    assert_fp32_model(target_student, "ema_target")
    logger.info("Activation checkpointing enabled for packed action training; cached forwards bypass recomputation")
    logger.info("Training state: FP32 masters/Adam/EMA; forward dtype=%s; explicit causal masks", dtype)
    if checkpoint_root is None:
        assert_fresh_students_identical(student, target_student)
    return teacher, student, target_student


def resolve_bandwidths(
    config: TrainingConfig,
    checkpoint_root: Path | None,
    train_loader,
    action_response_probe: ActionResponseProbe,
    action_response_metric: ActionResponseMetric,
    video_feature_extractor: VideoFeatureExtractor,
    video_metric: VideoMetric,
    teacher_signature_cache: TeacherSignatureCache,
) -> KernelBandwidths:
    configured = None
    if config.video_bandwidth is not None:
        configured = KernelBandwidths(
            video=config.video_bandwidth,
            action=config.action_bandwidth,
        )

    checkpoint_bandwidths = load_checkpoint_bandwidths(checkpoint_root)
    if checkpoint_bandwidths is not None:
        if configured is not None and configured != checkpoint_bandwidths:
            raise ValueError(
                "configured kernel bandwidths differ from the resume checkpoint"
            )
        bandwidths = checkpoint_bandwidths
        config.set_kernel_bandwidths(
            video=bandwidths.video,
            action=bandwidths.action,
        )
        logger.info(
            "Loaded fixed bandwidths from checkpoint: video=%.6g action=%.6g",
            bandwidths.video,
            bandwidths.action,
        )
        return bandwidths
    if configured is not None:
        logger.info(
            "Using configured fixed bandwidths: video=%.6g action=%.6g",
            configured.video,
            configured.action,
        )
        return configured

    logger.info("Calibrating fixed bandwidths from one teacher-only batch")
    calibration_batch = next(iter(train_loader))
    calibration = calibrate_bandwidths(
        config,
        calibration_batch,
        action_response_probe,
        action_response_metric,
        video_feature_extractor,
        BandwidthCalibrator(video_metric),
        teacher_signature_cache,
    )
    bandwidths = calibration.bandwidths
    config.set_kernel_bandwidths(
        video=bandwidths.video,
        action=bandwidths.action,
    )
    logger.info(
        "Calibrated fixed bandwidths: video=%.6g action=%.6g",
        bandwidths.video,
        bandwidths.action,
    )
    return bandwidths


def run(config: TrainingConfig) -> None:
    device = initialize_distributed(config)
    rank_seed = config.seed + config.rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed_all(rank_seed)
    torch.set_float32_matmul_precision("high")
    if os.environ.get("DRIFTWAM_DETECT_ANOMALY") == "1":
        torch.autograd.set_detect_anomaly(True)
        logger.warning("Autograd anomaly detection is enabled")

    checkpoint_root = resolve_resume_checkpoint(config)
    if checkpoint_root is not None and not checkpoint_root.is_dir():
        raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_root}")
    initial_step = load_checkpoint_step(config, checkpoint_root)
    distributed_preflight(config, checkpoint_root, initial_step)

    if config.rank == 0:
        logger.info("Dataset: %s", config.dataset_path)
        logger.info("Teacher video bank: %s", config.teacher_video_bank_path)
        logger.info("Teacher: %s", config.teacher_model_path)
        logger.info(
            "Student initialization: %s (%s)",
            config.selected_student_init_path,
            config.student_init_source,
        )
        logger.info("Output: %s", config.output_dir)

    teacher, student, target_student = configure_models(
        config,
        device,
        checkpoint_root,
    )
    train_loader = build_training_dataloader(config)

    student_video_generator = StudentVideoGenerator(student, config)
    teacher_signature_cache = TeacherSignatureCache(
        config,
        allow_misses=False,
    )
    if config.rank == 0:
        logger.info(
            "Completed teacher-signature cache: %s",
            config.teacher_signature_cache_path,
        )
    video_feature_extractor = VideoFeatureExtractor(teacher, config)
    video_metric = VideoMetric()
    action_response_probe = ActionResponseProbe(
        teacher,
        config,
        action_channel_ids=config.used_action_channel_ids,
    )
    action_response_metric = ActionResponseMetric(
        config.action_dim,
        action_channel_ids=config.used_action_channel_ids,
    )
    action_execution_loss = ActionExecutionLoss(
        config.action_dim,
        action_channel_ids=config.used_action_channel_ids,
    )
    action_consistency_loss = ActionConsistencyLoss(
        teacher,
        student,
        target_student,
        config,
        action_channel_ids=config.used_action_channel_ids,
    )
    bandwidths = resolve_bandwidths(
        config,
        checkpoint_root,
        train_loader,
        action_response_probe,
        action_response_metric,
        video_feature_extractor,
        video_metric,
        teacher_signature_cache,
    )
    write_resolved_config(config)

    video_objective = VideoObjective(config)
    training_step = TrainingStep(
        config,
        bandwidths,
        student_video_generator,
        action_response_probe,
        video_feature_extractor,
        video_metric,
        action_response_metric,
        video_objective,
        teacher_signature_cache,
        action_execution_loss,
        action_consistency_loss,
    )
    trainer = JointDistillationTrainer(
        config,
        student,
        target_student,
        training_step,
        train_loader,
        initial_step=initial_step,
        checkpoint_root=checkpoint_root,
    )
    trainer.train()


def main() -> None:
    args = parse_args()
    config = build_config(args)
    logging.basicConfig(
        level=logging.INFO,
        format=(
            f"%(asctime)s | rank={config.rank} | "
            "%(levelname)s | %(name)s | %(message)s"
        ),
    )
    try:
        run(config)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
