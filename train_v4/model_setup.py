"""Model loading for action-response-guided video distillation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import DistillationConfig
from .runtime import activate_lingbot_va


_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _dtype(name: str) -> torch.dtype:
    try:
        return _DTYPES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(_DTYPES))
        raise ValueError(f"unsupported param_dtype {name!r}; choose {choices}") from exc


def _transformer_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    nested = path / "transformer"
    return nested if nested.is_dir() else path


def _load_transformer(
    config: DistillationConfig,
    path: Path,
    *,
    device: str | torch.device,
    storage_dtype: torch.dtype | None = None,
) -> Any:
    activate_lingbot_va(config.lingbot_va_root)
    from wan_va.modules.utils import load_transformer

    transformer_dir = _transformer_dir(path)
    if not transformer_dir.is_dir():
        raise FileNotFoundError(
            f"Transformer checkpoint directory not found: {transformer_dir}"
        )
    return load_transformer(
        transformer_dir,
        torch_dtype=storage_dtype or _dtype(config.param_dtype),
        torch_device=device,
        attn_mode="torch",
    )


def load_frozen_action_teacher(
    config: DistillationConfig,
    *,
    device: str | torch.device = "cpu",
) -> Any:
    """Load the frozen LingBot-VA teacher used for action signatures."""

    teacher = _load_transformer(
        config,
        config.teacher_model_path,
        device=device,
    )
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


def configure_frozen_teacher(
    config: DistillationConfig,
    *,
    device: str | torch.device,
) -> Any:
    """Load one frozen teacher replica per rank for feature/response probes."""

    activate_lingbot_va(config.lingbot_va_root)
    import torch.distributed as dist

    teacher = load_frozen_action_teacher(config, device="cpu")
    if dist.is_initialized():
        dist.barrier()
    # The same replica serves causal action responses and video features.
    # Keeping it unsharded also avoids FSDP cache-lifetime constraints.
    teacher.to(device=device)
    teacher.eval().requires_grad_(False)
    return teacher


def configure_lingbot_video_teacher(
    config: DistillationConfig,
    *,
    device: str | torch.device,
) -> Any:
    """Load and shard the original LingBot-VA teacher for offline videos."""

    activate_lingbot_va(config.lingbot_va_root)
    from wan_va.distributed.fsdp import shard_model
    from wan_va.distributed.util import _configure_model
    from wan_va.modules.utils import load_transformer

    transformer_dir = _transformer_dir(config.teacher_model_path)
    if not transformer_dir.is_dir():
        raise FileNotFoundError(
            f"Transformer checkpoint directory not found: {transformer_dir}"
        )
    dtype = _dtype(config.param_dtype)
    teacher = load_transformer(
        transformer_dir,
        torch_dtype=dtype,
        torch_device="cpu",
        attn_mode="torch",
    )
    teacher.requires_grad_(False)
    teacher.eval()
    teacher = _configure_model(
        teacher,
        shard_fn=lambda model: shard_model(model, param_dtype=dtype),
        param_dtype=dtype,
        device=device,
        eval_mode=True,
    )
    teacher.eval().requires_grad_(False)
    return teacher


def load_online_student(
    config: DistillationConfig,
    *,
    device: str | torch.device = "cpu",
) -> Any:
    """Load the trainable student from the configured initialization source."""

    student = _load_transformer(
        config,
        config.selected_student_init_path,
        device=device,
        storage_dtype=torch.float32,
    )
    student.requires_grad_(True)
    student.train()
    return student


def load_target_student(
    config: DistillationConfig,
    *,
    device: str | torch.device = "cpu",
) -> Any:
    """Load the frozen EMA target from the same configured initialization."""

    target = _load_transformer(
        config,
        config.selected_student_init_path,
        device=device,
        storage_dtype=torch.float32,
    )
    target.requires_grad_(False)
    target.eval()
    return target

