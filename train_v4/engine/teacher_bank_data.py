"""Raw LingBot-VA latent samples used to build the teacher-video bank."""

from __future__ import annotations

import logging
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..forwards.action_signature import ActionSignatureCondition
from ..runtime import activate_lingbot_va
from ..training_config import TrainingConfig
from .bank_format import teacher_video_bank_file, validate_latent_source


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TeacherBankSource:
    """One official latent item and all aligned chunk start positions."""

    source_index: int
    sample_id: str
    bank_path: Path
    latents: Tensor
    actions: Tensor
    text_emb: Tensor
    frame_starts: Tensor

    @property
    def initial_frame(self) -> Tensor:
        return self.latents[:, :1]

    def condition(self, frame_start: int) -> ActionSignatureCondition:
        """Build a batch-one condition ending exactly at frame_start."""

        if frame_start < 0 or frame_start >= self.latents.shape[1]:
            raise ValueError("frame_start is outside the source latent range")
        history_video = None
        history_action = None
        if frame_start > 0:
            history_video = self.latents[:, :frame_start].unsqueeze(0)
            history_action = self.actions[:, :frame_start].unsqueeze(0)
        return ActionSignatureCondition(
            text_emb=self.text_emb.unsqueeze(0),
            frame_start=frame_start,
            history_video=history_video,
            history_action=history_action,
            history_frame_start=0,
        )


class TeacherBankSourceDataset(Dataset[TeacherBankSource]):
    """Load official latent items through the original LingBot-VA dataset."""

    def __init__(self, config: TrainingConfig) -> None:
        if config.cfg_prob != 0:
            raise ValueError(
                "cfg_prob must be zero while generating a condition-matched "
                "teacher bank"
            )
        activate_lingbot_va(config.lingbot_va_root)
        from wan_va.dataset.lerobot_latent_dataset import LatentLeRobotDataset

        self.config = config
        repositories = sorted(
            info_path.parent.parent
            for info_path in config.dataset_path.rglob("meta/info.json")
        )
        self._datasets: list[Any] = []
        self._cumulative_ends: list[int] = []
        total_items = 0
        for repository in repositories:
            try:
                child = LatentLeRobotDataset(
                    repo_id=str(repository),
                    config=config,
                )
            except Exception as exc:
                logger.warning(
                    "Skipping incomplete latent dataset %s: %s",
                    repository.name,
                    exc,
                )
                continue
            self._datasets.append(child)
            total_items += len(child)
            self._cumulative_ends.append(total_items)

        if not self._datasets:
            raise RuntimeError(
                f"no valid LingBot-VA latent datasets found under "
                f"{config.dataset_path}"
            )
        logger.info(
            "Loaded %d/%d LingBot-VA latent sub-datasets",
            len(self._datasets),
            len(repositories),
        )

    def __len__(self) -> int:
        return self._cumulative_ends[-1]

    def _resolve(self, index: int) -> tuple[Any, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        dataset_index = bisect_right(self._cumulative_ends, index)
        child_start = (
            0 if dataset_index == 0 else self._cumulative_ends[dataset_index - 1]
        )
        return self._datasets[dataset_index], index - child_start

    def bank_path(self, index: int) -> Path:
        """Resolve an output path without loading the item's latent tensors."""

        child, local_index = self._resolve(index)
        metadata = child.new_metas[local_index]
        return teacher_video_bank_file(
            self.config.teacher_video_bank_path,
            self.config.dataset_path,
            child.repo_id,
            episode_index=metadata["episode_index"],
            start_frame=metadata["start_frame"],
            end_frame=metadata["end_frame"],
        )

    def __getitem__(self, index: int) -> TeacherBankSource:
        child, local_index = self._resolve(index)
        metadata = child.new_metas[local_index]
        source = child[local_index]

        latents = source["latents"].contiguous()
        actions = source["actions"].contiguous()
        action_mask = source["actions_mask"].to(dtype=torch.bool).contiguous()
        validate_latent_source(
            self.config,
            latents,
            actions,
            action_mask,
        )
        actions = actions.masked_fill(~action_mask, 0).contiguous()

        text_emb = source["text_emb"].contiguous()
        if text_emb.ndim == 3 and text_emb.shape[0] == 1:
            text_emb = text_emb[0]
        if text_emb.ndim != 2:
            raise ValueError("source text_emb must have shape [L,D]")

        chunk_size = self.config.frame_chunk_size
        frame_count = latents.shape[1]
        if frame_count < chunk_size:
            raise ValueError(
                f"source {index} contains fewer than {chunk_size} latent frames"
            )
        frame_starts = torch.arange(
            0,
            frame_count - chunk_size + 1,
            chunk_size,
            dtype=torch.long,
        )

        relative_repo = Path(child.repo_id).resolve().relative_to(
            self.config.dataset_path
        )
        sample_id = (
            f"{relative_repo.as_posix()}/"
            f"episode_{int(metadata['episode_index']):06d}_"
            f"{int(metadata['start_frame'])}_{int(metadata['end_frame'])}"
        )
        return TeacherBankSource(
            source_index=index,
            sample_id=sample_id,
            bank_path=self.bank_path(index),
            latents=latents,
            actions=actions,
            text_emb=text_emb,
            frame_starts=frame_starts,
        )


def source_shape(source: TeacherBankSource, frame_chunk_size: int) -> tuple[int, ...]:
    """Return the noise shape for one candidate of one current chunk."""

    return (
        source.latents.shape[0],
        frame_chunk_size,
        source.latents.shape[2],
        source.latents.shape[3],
    )
