"""Validation and completion manifests for offline teacher-video banks."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from ..training_config import TrainingConfig
from .bank_format import FRAME_STARTS_KEY, TEACHER_VIDEOS_KEY
from .teacher_bank_data import TeacherBankSourceDataset


BANK_COMPLETE_FILENAME = "BANK_COMPLETE.json"
BANK_MANIFEST_VERSION = 1
_BANK_LOCATION_FIELDS = frozenset({"dataset_path"})


def bank_complete_path(bank_root: str | Path) -> Path:
    return Path(bank_root).expanduser().resolve() / BANK_COMPLETE_FILENAME


def remove_bank_complete_marker(bank_root: str | Path) -> None:
    """Invalidate a bank before any entry is generated or overwritten."""

    marker = bank_complete_path(bank_root)
    if marker.exists():
        marker.unlink()


def _transformer_dir(model_path: str | Path) -> Path:
    root = Path(model_path).expanduser().resolve()
    candidates = (root, root / "transformer")
    for candidate in candidates:
        if (candidate / "config.json").is_file() and any(
            candidate.glob("*.safetensors")
        ):
            return candidate
    raise FileNotFoundError(
        f"cannot find a Transformer checkpoint below {root}"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_inventory(transformer_dir: Path) -> list[dict[str, Any]]:
    files = sorted(transformer_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(
            f"no safetensors weights found in {transformer_dir}"
        )
    return [
        {"name": path.name, "size_bytes": path.stat().st_size}
        for path in files
    ]


def _paths_digest(relative_paths: list[str]) -> str:
    digest = hashlib.sha256()
    for value in relative_paths:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def expected_bank_contract(
    config: TrainingConfig,
) -> tuple[dict[str, Any], list[Path]]:
    """Resolve the exact source-to-bank mapping implied by the dataset."""

    dataset = TeacherBankSourceDataset(config)
    paths = [dataset.bank_path(index) for index in range(len(dataset))]
    relative_paths = [
        path.relative_to(config.teacher_video_bank_path).as_posix()
        for path in paths
    ]
    transformer = _transformer_dir(config.teacher_model_path)
    contract = {
        "manifest_version": BANK_MANIFEST_VERSION,
        "dataset_path": str(config.dataset_path),
        "dataset_count": len(dataset._datasets),
        "source_count": len(dataset),
        "expected_paths_sha256": _paths_digest(relative_paths),
        "teacher_model_path": str(config.teacher_model_path),
        "teacher_transformer_config_sha256": _sha256_file(
            transformer / "config.json"
        ),
        "teacher_transformer_inventory": _checkpoint_inventory(transformer),
        "teacher_candidate_count": config.teacher_candidate_count,
        "teacher_video_num_inference_steps": (
            config.teacher_video_num_inference_steps
        ),
        "teacher_video_guidance_scale": config.teacher_video_guidance_scale,
        "teacher_bank_seed": config.teacher_bank_seed,
        "frame_chunk_size": config.frame_chunk_size,
    }
    return contract, paths


def _validate_entry(path: Path, config: TrainingConfig) -> None:
    entry = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(entry, Mapping):
        raise TypeError(f"teacher bank entry must be a mapping: {path}")
    if TEACHER_VIDEOS_KEY not in entry or FRAME_STARTS_KEY not in entry:
        raise KeyError(
            f"teacher bank entry is missing required tensors: {path}"
        )

    videos = entry[TEACHER_VIDEOS_KEY]
    starts = entry[FRAME_STARTS_KEY]
    if not isinstance(videos, Tensor) or videos.ndim != 6:
        raise ValueError(
            f"teacher_videos must have shape [S,M,C,F,H,W]: {path}"
        )
    if videos.dtype != torch.bfloat16:
        raise ValueError(
            f"teacher_videos must be bfloat16, got {videos.dtype}: {path}"
        )
    if videos.shape[1] != config.teacher_candidate_count:
        raise ValueError(
            f"expected {config.teacher_candidate_count} candidates: {path}"
        )
    if videos.shape[3] != config.frame_chunk_size:
        raise ValueError(
            f"expected {config.frame_chunk_size} frames per chunk: {path}"
        )
    if not isinstance(starts, Tensor):
        raise TypeError(f"frame_starts must be a tensor: {path}")
    starts = starts.to(dtype=torch.long).reshape(-1)
    if starts.numel() == 0 or starts.numel() != videos.shape[0]:
        raise ValueError(
            f"frame_starts and stored chunk counts differ: {path}"
        )
    if starts[0].item() != 0:
        raise ValueError(f"frame_starts must begin at zero: {path}")
    if bool((starts % config.frame_chunk_size != 0).any()):
        raise ValueError(f"frame_starts are not chunk-aligned: {path}")
    if starts.numel() > 1 and not bool((starts[1:] > starts[:-1]).all()):
        raise ValueError(f"frame_starts must be strictly increasing: {path}")
    if not bool(torch.isfinite(videos).all()):
        raise ValueError(f"teacher bank contains non-finite values: {path}")


def _check_expected_counts(
    contract: Mapping[str, Any],
    config: TrainingConfig,
) -> None:
    if (
        config.expected_dataset_count is not None
        and contract["dataset_count"] != config.expected_dataset_count
    ):
        raise ValueError(
            f"expected {config.expected_dataset_count} datasets, found "
            f"{contract['dataset_count']}"
        )
    if (
        config.expected_source_count is not None
        and contract["source_count"] != config.expected_source_count
    ):
        raise ValueError(
            f"expected {config.expected_source_count} sources, found "
            f"{contract['source_count']}"
        )


def validate_teacher_bank_and_write(
    config: TrainingConfig,
    *,
    log_interval: int = 100,
) -> dict[str, Any]:
    """Fully validate every entry, then atomically publish the bank marker."""

    contract, paths = expected_bank_contract(config)
    _check_expected_counts(contract, config)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        examples = ", ".join(str(path) for path in missing[:5])
        raise FileNotFoundError(
            f"teacher bank is incomplete: {len(missing)}/{len(paths)} entries "
            f"are missing; examples: {examples}"
        )

    for index, path in enumerate(paths, start=1):
        _validate_entry(path, config)
        if log_interval > 0 and index % log_interval == 0:
            print(f"validated teacher bank entries: {index}/{len(paths)}")

    manifest = dict(contract)
    manifest.update(
        {
            "bank_path": str(config.teacher_video_bank_path),
            "validated_file_count": len(paths),
            "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    marker = bank_complete_path(config.teacher_video_bank_path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, marker)
    finally:
        if temporary.exists():
            temporary.unlink()
    return manifest


def verify_teacher_bank_marker(config: TrainingConfig) -> dict[str, Any]:
    """Verify the marker contract and the existence of every expected entry."""

    marker = bank_complete_path(config.teacher_video_bank_path)
    if not marker.is_file():
        raise FileNotFoundError(
            f"teacher bank has no {BANK_COMPLETE_FILENAME}: {marker.parent}; "
            "run `python -m train_v4.validate_teacher_bank --write-marker` "
            "after generation finishes"
        )
    with marker.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, Mapping):
        raise TypeError(f"teacher bank marker must be a JSON object: {marker}")

    expected, paths = expected_bank_contract(config)
    _check_expected_counts(expected, config)
    # A bank may be copied from shared storage to a faster local disk. Its
    # source inventory and generation contract must stay identical, but the
    # absolute dataset location is not part of the generated content.
    mismatches = {
        key: {"marker": manifest.get(key), "expected": value}
        for key, value in expected.items()
        if key not in _BANK_LOCATION_FIELDS and manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "teacher bank marker does not match the configured dataset/teacher: "
            f"{mismatches}"
        )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        examples = ", ".join(str(path) for path in missing[:5])
        raise FileNotFoundError(
            f"teacher bank marker exists but {len(missing)} entries are missing; "
            f"examples: {examples}"
        )
    if manifest.get("validated_file_count") != len(paths):
        raise ValueError(
            "teacher bank marker has the wrong validated_file_count"
        )
    return dict(manifest)
