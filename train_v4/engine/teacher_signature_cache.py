"""Persistent cache for frozen-teacher action signatures."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import torch
from torch import Tensor

from ..forwards.action_signature import (
    ActionSignatureCondition,
    ActionSignatureGenerator,
)
from ..training_config import TrainingConfig
from .inference_compatibility import inference_contract_source_hash


CACHE_CONTRACT_FILENAME = "CACHE_CONTRACT.json"
CACHE_COMPLETE_FILENAME = "CACHE_COMPLETE.json"
CACHE_FORMAT_VERSION = 2
_CACHE_LOCATION_FIELDS = frozenset(
    {"dataset_path", "teacher_video_bank_path"}
)


@dataclass(slots=True)
class TeacherSignatureBatch:
    """Cached teacher signatures and cache lookup statistics."""

    signatures: Tensor
    hits: int
    misses: int


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _transformer_dir(model_path: Path) -> Path:
    for candidate in (model_path, model_path / "transformer"):
        if (candidate / "config.json").is_file() and any(
            candidate.glob("*.safetensors")
        ):
            return candidate
    raise FileNotFoundError(
        f"cannot find a Transformer checkpoint below {model_path}"
    )


def _checkpoint_inventory(transformer_dir: Path) -> list[dict[str, Any]]:
    weights = sorted(transformer_dir.glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError(
            f"no safetensors weights found in {transformer_dir}"
        )
    return [
        {"name": path.name, "size_bytes": path.stat().st_size}
        for path in weights
    ]


def teacher_signature_cache_contract(config: TrainingConfig) -> dict[str, Any]:
    """Describe every frozen input that changes a cached signature."""

    transformer = _transformer_dir(config.teacher_model_path)
    bank_marker = config.teacher_video_bank_path / "BANK_COMPLETE.json"
    if not bank_marker.is_file():
        raise FileNotFoundError(
            f"teacher-video bank marker not found: {bank_marker}"
        )
    signature_source = (
        Path(__file__).resolve().parents[1] / "forwards" / "action_signature.py"
    )
    model_source = (
        config.lingbot_va_root / "wan_va" / "modules" / "model.py"
    )
    if not model_source.is_file():
        raise FileNotFoundError(f"LingBot-VA model source not found: {model_source}")

    return {
        "format_version": CACHE_FORMAT_VERSION,
        "dataset_path": str(config.dataset_path),
        "teacher_video_bank_path": str(config.teacher_video_bank_path),
        "teacher_video_bank_marker_sha256": _sha256_file(bank_marker),
        "teacher_model_path": str(config.teacher_model_path),
        "teacher_transformer_config_sha256": _sha256_file(
            transformer / "config.json"
        ),
        "teacher_transformer_inventory": _checkpoint_inventory(transformer),
        "action_signature_source_sha256": _sha256_file(signature_source),
        "lingbot_model_source_sha256": inference_contract_source_hash(
            _sha256_file(model_source)
        ),
        "teacher_signature_noise_seed": config.teacher_signature_noise_seed,
        "param_dtype": config.param_dtype,
        "signature_horizon": config.signature_horizon,
        "frame_chunk_size": config.frame_chunk_size,
        "action_dim": config.action_dim,
        "action_per_frame": config.action_per_frame,
        "action_teacher_num_inference_steps": (
            config.action_teacher_num_inference_steps
        ),
        "action_num_train_timesteps": config.action_num_train_timesteps,
        "action_snr_shift": config.action_snr_shift,
        "attn_window": config.attn_window,
        "teacher_candidate_count": config.teacher_candidate_count,
        "used_action_channel_ids": list(config.used_action_channel_ids),
    }


def _contract_digest(contract: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _validate_contract_payload(
    payload: Any,
    expected_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate cache identity while permitting storage-only relocation."""

    if not isinstance(payload, Mapping):
        raise ValueError("teacher-signature cache contract is not an object")
    stored_contract = payload.get("contract")
    stored_digest = payload.get("contract_sha256")
    if not isinstance(stored_contract, Mapping) or not isinstance(
        stored_digest, str
    ):
        raise ValueError("teacher-signature cache contract is malformed")
    if stored_digest != _contract_digest(stored_contract):
        raise ValueError("teacher-signature cache contract digest is invalid")

    stored_identity = {
        key: value
        for key, value in stored_contract.items()
        if key not in _CACHE_LOCATION_FIELDS
    }
    expected_identity = {
        key: value
        for key, value in expected_contract.items()
        if key not in _CACHE_LOCATION_FIELDS
    }
    if stored_identity != expected_identity:
        raise ValueError("teacher-signature cache contract differs from this run")
    return dict(payload)


def prepare_teacher_signature_cache(config: TrainingConfig) -> dict[str, Any]:
    """Create or validate the immutable cache contract on rank zero."""

    root = config.teacher_signature_cache_path
    contract = teacher_signature_cache_contract(config)
    payload = {
        "contract_sha256": _contract_digest(contract),
        "contract": contract,
    }
    marker = root / CACHE_CONTRACT_FILENAME
    if marker.is_file():
        with marker.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        try:
            existing = _validate_contract_payload(existing, contract)
        except ValueError as error:
            raise ValueError(
                "teacher-signature cache contract differs from this run; "
                f"use a new empty cache path instead of {root}"
            ) from error
        return existing

    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"teacher-signature cache is non-empty but has no valid contract: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, marker)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def write_teacher_signature_cache_complete(
    config: TrainingConfig,
    *,
    expected_entries: int,
) -> dict[str, Any]:
    """Publish a completion marker after every expected entry was resolved."""

    if expected_entries < 1:
        raise ValueError("expected_entries must be positive")
    contract_payload = prepare_teacher_signature_cache(config)
    root = config.teacher_signature_cache_path
    entry_count = sum(1 for _ in (root / "entries").rglob("*.pt"))
    if entry_count != expected_entries:
        raise ValueError(
            f"teacher-signature cache has {entry_count} entries; "
            f"expected {expected_entries}"
        )
    payload = {
        "format_version": CACHE_FORMAT_VERSION,
        "contract_sha256": contract_payload["contract_sha256"],
        "expected_entries": expected_entries,
        "validated_entries": entry_count,
    }
    marker = root / CACHE_COMPLETE_FILENAME
    temporary = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, marker)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def verify_teacher_signature_cache_complete(
    config: TrainingConfig,
) -> dict[str, Any]:
    """Verify the immutable contract, marker, and cache entry count."""

    root = config.teacher_signature_cache_path
    contract_marker = root / CACHE_CONTRACT_FILENAME
    complete_marker = root / CACHE_COMPLETE_FILENAME
    if not contract_marker.is_file() or not complete_marker.is_file():
        raise FileNotFoundError(
            "teacher-signature cache is incomplete; run "
            "`python -m train_v4.precompute_teacher_signatures` first: "
            f"{root}"
        )
    with contract_marker.open("r", encoding="utf-8") as handle:
        contract_payload = json.load(handle)
    with complete_marker.open("r", encoding="utf-8") as handle:
        complete_payload = json.load(handle)
    expected_contract = teacher_signature_cache_contract(config)
    try:
        contract_payload = _validate_contract_payload(
            contract_payload,
            expected_contract,
        )
    except ValueError as error:
        raise ValueError(
            f"teacher-signature cache contract differs from this run: {root}"
        ) from error
    stored_digest = contract_payload["contract_sha256"]
    if not isinstance(complete_payload, Mapping):
        raise TypeError(
            f"teacher-signature completion marker must be an object: "
            f"{complete_marker}"
        )
    expected_entries = complete_payload.get("expected_entries")
    if (
        complete_payload.get("format_version") != CACHE_FORMAT_VERSION
        or complete_payload.get("contract_sha256") != stored_digest
        or not isinstance(expected_entries, int)
        or expected_entries < 1
        or complete_payload.get("validated_entries") != expected_entries
    ):
        raise ValueError(
            f"invalid teacher-signature completion marker: {complete_marker}"
        )
    entry_count = sum(1 for _ in (root / "entries").rglob("*.pt"))
    if entry_count != expected_entries:
        raise ValueError(
            f"teacher-signature cache has {entry_count} entries but its "
            f"completion marker requires {expected_entries}"
        )
    return dict(complete_payload)


class TeacherSignatureCache:
    """Load teacher signatures, optionally computing persistent misses."""

    def __init__(
        self,
        config: TrainingConfig,
        generator: ActionSignatureGenerator | None = None,
        *,
        allow_misses: bool = True,
    ) -> None:
        self.config = config
        self.generator = generator
        self.allow_misses = bool(allow_misses)
        self.root = config.teacher_signature_cache_path
        marker = self.root / CACHE_CONTRACT_FILENAME
        if not marker.is_file():
            raise FileNotFoundError(
                f"teacher-signature cache contract not found: {marker}"
            )
        with marker.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        expected_contract = teacher_signature_cache_contract(config)
        try:
            payload = _validate_contract_payload(payload, expected_contract)
        except ValueError as error:
            raise ValueError(
                "teacher-signature cache contract differs from this run: "
                f"{self.root}"
            ) from error
        # Existing entries reference the digest written when the cache was
        # built, so relocation must retain that original, validated digest.
        self.contract_sha256 = payload["contract_sha256"]

    def get_or_compute(
        self,
        sample_ids: tuple[str, ...],
        teacher_videos: Tensor,
        condition: ActionSignatureCondition,
    ) -> TeacherSignatureBatch:
        """Return a batch, atomically saving each previously unseen item."""

        batch_size = teacher_videos.shape[0]
        if len(sample_ids) != batch_size:
            raise ValueError("sample_ids and teacher videos have different batches")
        frame_starts = _batch_positions(
            condition.frame_start,
            batch_size,
            name="frame_start",
        )
        expected_noise_shape = (
            self.config.action_dim,
            teacher_videos.shape[3],
            self.config.action_per_frame,
            1,
        )
        expected_signature_shape = (
            teacher_videos.shape[1],
            *expected_noise_shape,
        )

        signatures: list[Tensor | None] = [None] * batch_size
        missing_indices: list[int] = []
        for index, sample_id in enumerate(sample_ids):
            path = self._entry_path(sample_id)
            if path.is_file():
                signature = self._load_entry(
                    path,
                    sample_id=sample_id,
                    frame_start=frame_starts[index],
                    expected_signature_shape=expected_signature_shape,
                )
                signatures[index] = signature
            else:
                missing_indices.append(index)

        device = teacher_videos.device
        dtype = getattr(torch, self.config.param_dtype)

        if missing_indices:
            if not self.allow_misses:
                examples = ", ".join(
                    sample_ids[index] for index in missing_indices[:3]
                )
                raise FileNotFoundError(
                    "formal training does not compute 50-step teacher "
                    f"signatures online; missing {len(missing_indices)} cache "
                    f"entries, examples: {examples}"
                )
            if self.generator is None:
                raise RuntimeError(
                    "a signature generator is required when cache misses are allowed"
                )
            missing_noise_cpu = torch.stack(
                [
                    self._deterministic_noise(
                        sample_ids[index],
                        expected_noise_shape,
                    )
                    for index in missing_indices
                ]
            )
            missing_action_noise = missing_noise_cpu.to(
                device=device,
                dtype=dtype,
            )
            index_tensor = torch.tensor(
                missing_indices,
                device=device,
                dtype=torch.long,
            )
            missing_signatures = self.generator(
                teacher_videos.index_select(0, index_tensor),
                missing_action_noise,
                _select_condition(condition, index_tensor),
            )
            for result_index, batch_index in enumerate(missing_indices):
                signature = missing_signatures[result_index].detach().to(
                    device="cpu",
                    dtype=dtype,
                ).contiguous()
                signatures[batch_index] = signature
                self._save_entry(
                    self._entry_path(sample_ids[batch_index]),
                    sample_id=sample_ids[batch_index],
                    frame_start=frame_starts[batch_index],
                    signature=signature,
                )

        if any(value is None for value in signatures):
            raise RuntimeError("teacher-signature cache left an unresolved item")
        teacher_signatures = torch.stack(
            [value for value in signatures if value is not None]
        ).to(device=device, dtype=dtype)
        return TeacherSignatureBatch(
            signatures=teacher_signatures,
            hits=batch_size - len(missing_indices),
            misses=len(missing_indices),
        )

    def _entry_path(self, sample_id: str) -> Path:
        digest = _sha256_bytes(sample_id.encode("utf-8"))
        return self.root / "entries" / digest[:2] / digest[2:4] / f"{digest}.pt"

    def _deterministic_noise(
        self,
        sample_id: str,
        shape: tuple[int, ...],
    ) -> Tensor:
        seed_material = (
            f"{self.config.teacher_signature_noise_seed}\0{sample_id}"
        ).encode("utf-8")
        seed = int.from_bytes(
            hashlib.sha256(seed_material).digest()[:8],
            byteorder="little",
            signed=False,
        ) & ((1 << 63) - 1)
        random_generator = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn(
            shape,
            generator=random_generator,
            dtype=torch.float32,
        ).to(dtype=getattr(torch, self.config.param_dtype)).contiguous()

    def _load_entry(
        self,
        path: Path,
        *,
        sample_id: str,
        frame_start: int,
        expected_signature_shape: tuple[int, ...],
    ) -> Tensor:
        entry = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(entry, Mapping):
            raise TypeError(f"teacher-signature cache entry is not a mapping: {path}")
        expected_metadata = {
            "format_version": CACHE_FORMAT_VERSION,
            "contract_sha256": self.contract_sha256,
            "sample_id": sample_id,
            "frame_start": frame_start,
        }
        mismatches = {
            key: {"cached": entry.get(key), "expected": value}
            for key, value in expected_metadata.items()
            if entry.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"teacher-signature cache metadata differs at {path}: {mismatches}"
            )
        signature = entry.get("teacher_signatures")
        if not isinstance(signature, Tensor):
            raise TypeError(f"teacher-signature cache tensor is missing: {path}")
        if tuple(signature.shape) != expected_signature_shape:
            raise ValueError(
                f"cached teacher_signatures must have shape "
                f"{expected_signature_shape}, got {tuple(signature.shape)}: {path}"
            )
        expected_dtype = getattr(torch, self.config.param_dtype)
        if signature.dtype != expected_dtype:
            raise ValueError(
                f"cached teacher_signatures must use {expected_dtype}: {path}"
            )
        if not bool(torch.isfinite(signature).all()):
            raise ValueError(f"teacher-signature cache contains non-finite data: {path}")
        return signature.contiguous()

    def _save_entry(
        self,
        path: Path,
        *,
        sample_id: str,
        frame_start: int,
        signature: Tensor,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f".{path.name}.rank{self.config.rank}.{os.getpid()}.{uuid4().hex}.tmp"
        )
        payload = {
            "format_version": CACHE_FORMAT_VERSION,
            "contract_sha256": self.contract_sha256,
            "sample_id": sample_id,
            "frame_start": frame_start,
            "teacher_signatures": signature,
        }
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _batch_positions(
    value: int | Tensor,
    batch_size: int,
    *,
    name: str,
) -> list[int]:
    if isinstance(value, Tensor):
        positions = value.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        if positions.numel() == 1 and batch_size != 1:
            positions = positions.expand(batch_size)
        if positions.numel() != batch_size:
            raise ValueError(f"{name} must contain one value per condition")
        return [int(item) for item in positions.tolist()]
    return [int(value)] * batch_size


def _select_condition(
    condition: ActionSignatureCondition,
    indices: Tensor,
) -> ActionSignatureCondition:
    def select(value: Tensor | None) -> Tensor | None:
        return None if value is None else value.index_select(0, indices)

    def select_positions(value: int | Tensor) -> int | Tensor:
        return value if isinstance(value, int) else value.index_select(0, indices)

    return ActionSignatureCondition(
        text_emb=condition.text_emb.index_select(0, indices),
        frame_start=select_positions(condition.frame_start),
        history_video=select(condition.history_video),
        history_action=select(condition.history_action),
        history_frame_start=select_positions(condition.history_frame_start),
    )
