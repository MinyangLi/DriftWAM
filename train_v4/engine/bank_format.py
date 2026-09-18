"""Neutral file-format helpers shared by teacher-bank writers and readers."""

from __future__ import annotations

from pathlib import Path

from torch import Tensor

from ..training_config import TrainingConfig


TEACHER_VIDEOS_KEY = "teacher_videos"
FRAME_STARTS_KEY = "frame_starts"


def teacher_video_bank_file(
    bank_root: str | Path,
    dataset_root: str | Path,
    repo_id: str | Path,
    *,
    episode_index: int,
    start_frame: int,
    end_frame: int,
) -> Path:
    """Return the stable bank path for one official latent-dataset item."""

    dataset_root = Path(dataset_root).expanduser().resolve()
    repo_path = Path(repo_id).expanduser().resolve()
    try:
        relative_repo = repo_path.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(
            f"dataset repository {repo_path} is outside {dataset_root}"
        ) from exc
    filename = (
        f"episode_{int(episode_index):06d}_"
        f"{int(start_frame)}_{int(end_frame)}.pt"
    )
    return Path(bank_root).expanduser().resolve() / relative_repo / filename


def validate_latent_source(
    config: TrainingConfig,
    latents: Tensor,
    actions: Tensor,
    action_mask: Tensor,
) -> None:
    """Check the shared latent/action contract used by bank I/O."""

    if latents.ndim != 4:
        raise ValueError("source latents must have shape [C,F,H,W]")
    expected_action = (
        config.action_dim,
        latents.shape[1],
        config.action_per_frame,
        1,
    )
    if tuple(actions.shape) != expected_action:
        raise ValueError(
            f"source actions must have shape {expected_action}, got "
            f"{tuple(actions.shape)}"
        )
    if action_mask.shape != actions.shape:
        raise ValueError("source actions and actions_mask shapes differ")
