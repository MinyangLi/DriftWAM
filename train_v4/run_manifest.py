"""Preflight validation and auditable run manifests."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch.distributed as dist

from .engine.bank_validation import verify_teacher_bank_marker
from .engine.checkpoint_retention import reclaimable_checkpoint_bytes
from .engine.precision import TRAINING_NUMERICS
from .engine.teacher_signature_cache import (
    verify_teacher_signature_cache_complete,
)
from .training_config import TrainingConfig


RUN_MANIFEST_VERSION = 7
CHECKPOINT_COMPLETE_FILENAME = "CHECKPOINT_COMPLETE"
RESUME_COMPLETE_FILENAME = "RESUME_STATE_COMPLETE"
CHECKPOINT_DISK_HEADROOM_BYTES = 8 * 1024**3
OPTIMIZER_STATE_BYTES_PER_MODEL_BYTE = 2


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _atomic_json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(_jsonable(value), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _transformer_dir(model_path: str | Path) -> Path:
    root = Path(model_path).expanduser().resolve()
    for candidate in (root, root / "transformer"):
        if (candidate / "config.json").is_file() and any(
            candidate.glob("*.safetensors")
        ):
            return candidate
    raise FileNotFoundError(
        f"cannot find a Transformer checkpoint below {root}"
    )


def _planned_checkpoint_steps(
    config: TrainingConfig,
    initial_step: int,
) -> tuple[int, ...]:
    """Return every future step that will publish a checkpoint."""

    if initial_step >= config.max_train_steps:
        return ()
    first_interval = (
        initial_step // config.save_interval + 1
    ) * config.save_interval
    steps = list(
        range(
            first_interval,
            config.max_train_steps + 1,
            config.save_interval,
        )
    )
    if (
        config.save_final_checkpoint
        and config.max_train_steps not in steps
    ):
        steps.append(config.max_train_steps)
    return tuple(steps)


def estimate_checkpoint_storage(
    config: TrainingConfig,
    student_transformer: Path,
    initial_step: int,
) -> dict[str, Any]:
    """Conservatively estimate peak additional bytes for future saves.

    A model checkpoint stores both the online student and EMA target. When
    exact resume is enabled, AdamW adds two moment tensors whose observed size
    is twice the FP32 model tensor payload. Obsolete checkpoints are deleted
    BEFORE each new save, reserving one retention slot for the new payload.
    Credit reclaimable existing files on resume; existing retained files already
    count against reported free space. This policy has no old-checkpoint fallback
    if the replacement save fails after deletion.
    """

    checkpoint_steps = _planned_checkpoint_steps(config, initial_step)
    model_bytes = sum(
        path.stat().st_size
        for path in student_transformer.glob("*.safetensors")
    )
    if model_bytes <= 0:
        raise FileNotFoundError(
            f"student Transformer has no safetensors weights: "
            f"{student_transformer}"
        )
    # Inspect shapes in safetensors headers; never materialize 5B weights.
    from safetensors import safe_open
    fp32_model_bytes = 0
    for path in student_transformer.glob("*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as tensors:
            for key in tensors.keys():
                view = tensors.get_slice(key)
                item_bytes = {
                    "F64": 8, "I64": 8, "U64": 8, "C64": 8, "C128": 16,
                }.get(view.get_dtype(), 4)
                fp32_model_bytes += math.prod(view.get_shape()) * item_bytes
    model_pair_bytes = 2 * fp32_model_bytes
    resume_state_bytes = (
        OPTIMIZER_STATE_BYTES_PER_MODEL_BYTE * fp32_model_bytes
        if config.save_optimizer_state
        else 0
    )
    checkpoint_count = len(checkpoint_steps)
    peak_checkpoint_count = min(
        checkpoint_count, config.retain_checkpoint_count,
    )
    full_checkpoint_bytes = model_pair_bytes + resume_state_bytes
    reclaimable_bytes = (
        reclaimable_checkpoint_bytes(
            config.output_dir / "checkpoints" / f"step_{checkpoint_steps[0]}",
            config.retain_checkpoint_count,
        )
        if checkpoint_steps else 0
    )
    peak_additional_bytes = max(
        0, peak_checkpoint_count * full_checkpoint_bytes - reclaimable_bytes,
    )
    headroom_bytes = (
        CHECKPOINT_DISK_HEADROOM_BYTES if checkpoint_count else 0
    )
    return {
        "checkpoint_steps": checkpoint_steps,
        "model_bytes": model_bytes,
        "fp32_model_bytes": fp32_model_bytes,
        "model_pair_bytes": model_pair_bytes,
        "resume_state_estimate_bytes": resume_state_bytes,
        "full_checkpoint_estimate_bytes": full_checkpoint_bytes,
        "retained_checkpoint_count": config.retain_checkpoint_count,
        "checkpoint_replacement_policy": "delete_before_save",
        "reclaimable_existing_checkpoint_bytes": reclaimable_bytes,
        "peak_future_checkpoint_count": peak_checkpoint_count,
        "peak_additional_bytes": peak_additional_bytes,
        "headroom_bytes": headroom_bytes,
        "required_free_bytes": peak_additional_bytes + headroom_bytes,
    }


def _existing_ancestor(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise FileNotFoundError(
                f"cannot find an existing parent for output path: {path}"
            )
        candidate = parent
    return candidate


def validate_checkpoint_capacity(
    config: TrainingConfig,
    student_transformer: Path,
    initial_step: int,
) -> dict[str, Any]:
    """Fail before model loading when planned atomic saves cannot fit."""

    estimate = estimate_checkpoint_storage(
        config,
        student_transformer,
        initial_step,
    )
    storage_root = _existing_ancestor(config.output_dir)
    available_bytes = shutil.disk_usage(storage_root).free
    estimate["storage_root"] = str(storage_root)
    estimate["available_free_bytes"] = available_bytes
    required_bytes = estimate["required_free_bytes"]
    if available_bytes < required_bytes:
        gib = float(1024**3)
        raise OSError(
            "insufficient free space for planned atomic checkpoints: "
            f"available={available_bytes / gib:.1f} GiB, "
            f"required={required_bytes / gib:.1f} GiB, "
            f"steps={list(estimate['checkpoint_steps'])}, "
            f"output={config.output_dir}"
        )
    return estimate


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _architecture_config(transformer_dir: Path) -> dict[str, Any]:
    config = _read_json(transformer_dir / "config.json")
    for key in ("_name_or_path", "_diffusers_version", "torch_dtype"):
        config.pop(key, None)
    return config


def _code_sha256(project_root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in (project_root / "train_v4").rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".sh"}
        and "__pycache__" not in path.parts
    )
    for path in files:
        digest.update(path.relative_to(project_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_output_location(
    config: TrainingConfig,
    checkpoint_root: Path | None,
) -> None:
    if checkpoint_root is None:
        if config.output_dir.exists() and any(config.output_dir.iterdir()):
            raise FileExistsError(
                f"fresh-run output directory is not empty: {config.output_dir}"
            )
        return

    checkpoint_root = checkpoint_root.resolve()
    if not (checkpoint_root / CHECKPOINT_COMPLETE_FILENAME).is_file():
        raise FileNotFoundError(
            f"resume checkpoint is incomplete: {checkpoint_root}"
        )
    if config.save_optimizer_state and not (
        checkpoint_root / RESUME_COMPLETE_FILENAME
    ).is_file():
        raise FileNotFoundError(
            f"checkpoint has no exact-resume state: {checkpoint_root}"
        )
    state = _read_json(checkpoint_root / "trainer_state.json")
    if state.get("action_supervision_source") != config.action_supervision_source:
        raise ValueError(
            "resume action_supervision_source differs: "
            f"checkpoint={state.get('action_supervision_source')!r}, "
            f"configured={config.action_supervision_source!r}"
        )
    expected_output = checkpoint_root.parent.parent.resolve()
    if config.output_dir != expected_output:
        raise ValueError(
            f"resume checkpoint belongs to {expected_output}, but OUTPUT_DIR is "
            f"{config.output_dir}"
        )


def validate_and_write_run_manifest(
    config: TrainingConfig,
    checkpoint_root: Path | None,
    initial_step: int,
) -> dict[str, Any]:
    """Validate every immutable run input before loading large models."""

    config.validate_runtime_contract()
    if config.save_optimizer_state and config.num_workers != 0:
        raise ValueError(
            "exact RNG/data-order resume currently requires num_workers=0"
        )
    if not config.lingbot_va_root.is_dir():
        raise FileNotFoundError(
            f"LingBot-VA runtime not found: {config.lingbot_va_root}"
        )
    if not config.dataset_path.is_dir():
        raise FileNotFoundError(f"dataset not found: {config.dataset_path}")
    if not config.empty_emb_path.is_file():
        raise FileNotFoundError(
            f"empty text embedding not found: {config.empty_emb_path}"
        )
    if not config.teacher_model_path.is_dir():
        raise FileNotFoundError(
            f"teacher checkpoint not found: {config.teacher_model_path}"
        )
    if not config.selected_student_init_path.is_dir():
        raise FileNotFoundError(
            f"student initialization not found: "
            f"{config.selected_student_init_path}"
        )

    _validate_output_location(config, checkpoint_root)
    teacher_transformer = _transformer_dir(config.teacher_model_path)
    student_transformer = _transformer_dir(config.selected_student_init_path)
    teacher_architecture = _architecture_config(teacher_transformer)
    student_architecture = _architecture_config(student_transformer)
    if teacher_architecture != student_architecture:
        raise ValueError(
            "teacher and student Transformer architecture configs differ"
        )
    checkpoint_storage = validate_checkpoint_capacity(
        config,
        student_transformer,
        initial_step,
    )

    bank_manifest = verify_teacher_bank_marker(config)
    signature_cache_manifest = verify_teacher_signature_cache_complete(config)
    project_root = Path(__file__).resolve().parents[1]
    now = datetime.now(timezone.utc)
    manifest = {
        "manifest_version": RUN_MANIFEST_VERSION,
        "training_numerics": TRAINING_NUMERICS,
        "created_at_utc": now.isoformat(),
        "experiment_name": config.experiment_name,
        "training_method": (
            "action_response_guided_drifting_with_action_consistency_"
            "and_flow_matching"
        ),
        "initialization_mode": config.student_init_source,
        "initialization_source": str(config.selected_student_init_path),
        "teacher_model_path": str(config.teacher_model_path),
        "dataset_path": str(config.dataset_path),
        "teacher_video_bank_path": str(config.teacher_video_bank_path),
        "teacher_signature_cache_path": str(config.teacher_signature_cache_path),
        "teacher_signature_cache_contract": signature_cache_manifest,
        "teacher_signature_noise_seed": config.teacher_signature_noise_seed,
        "teacher_bank_contract": bank_manifest,
        "step_zero_transformer": str(student_transformer),
        "seed": config.seed,
        "world_size": config.world_size,
        "per_rank_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "global_batch_size": config.effective_global_batch_size,
        "save_interval": config.save_interval,
        "max_train_steps": config.max_train_steps,
        "optimizer_warmup_steps": config.warmup_steps,
        "loss_warmup_steps": 0,
        "execution_response_mode": config.execution_response_mode,
        "execution_loss_weight": config.execution_loss_weight,
        "action_supervision_source": config.action_supervision_source,
        "action_consistency_loss_weight": (
            config.action_consistency_loss_weight
        ),
        "action_flow_matching_loss_weight": (
            config.action_flow_matching_loss_weight
        ),
        "checkpoint_storage": checkpoint_storage,
        "initial_step": initial_step,
        "resume_from": None if checkpoint_root is None else str(checkpoint_root),
        "train_v4_code_sha256": _code_sha256(project_root),
        "lingbot_model_actual_source_sha256": hashlib.sha256(
            (config.lingbot_va_root / "wan_va/modules/model.py").read_bytes()
        ).hexdigest(),
    }

    config.output_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_root is None:
        _atomic_json_dump(asdict(config), config.output_dir / "run_config.json")
        _atomic_json_dump(manifest, config.output_dir / "run_manifest.json")
    else:
        event_name = (
            f"resume_step_{initial_step}_{now.strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        _atomic_json_dump(
            manifest,
            config.output_dir / "resume_events" / event_name,
        )
    return manifest


def distributed_preflight(
    config: TrainingConfig,
    checkpoint_root: Path | None,
    initial_step: int,
) -> None:
    """Run filesystem-heavy validation once and propagate any failure."""

    payload: list[str | None] = [None]
    if config.rank == 0:
        try:
            validate_and_write_run_manifest(
                config,
                checkpoint_root,
                initial_step,
            )
        except Exception as exc:
            payload[0] = f"{type(exc).__name__}: {exc}"
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)
    if payload[0] is not None:
        raise RuntimeError(f"training preflight failed: {payload[0]}")


def write_resolved_config(config: TrainingConfig) -> None:
    """Persist the final config again after bandwidth calibration."""

    if config.rank == 0:
        _atomic_json_dump(
            asdict(config),
            config.output_dir / "resolved_config.json",
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
