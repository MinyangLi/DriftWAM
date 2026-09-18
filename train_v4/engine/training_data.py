"""Training samples backed by RoboTwin latents and offline teacher videos."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

from ..forwards.action_signature import ActionSignatureCondition
from .latent_dataset import SafeMultiLatentLeRobotDataset
from ..training_config import TrainingConfig
from .bank_format import (
    FRAME_STARTS_KEY,
    TEACHER_VIDEOS_KEY,
    teacher_video_bank_file,
    validate_latent_source,
)


def resolve_latent_source(base_dataset: Any, index: int) -> tuple[Any, int]:
    """Resolve a global multi-dataset index into its child and local index."""

    if index < 0:
        index += len(base_dataset)
    if index < 0 or index >= len(base_dataset):
        raise IndexError(index)
    dataset_id = base_dataset.item_id_to_dataset_id[index]
    child = base_dataset._datasets[dataset_id]
    local_index = index - base_dataset.acc_dset_num[dataset_id]
    return child, local_index


@dataclass(slots=True)
class TrainingSample:
    """One causal chunk and its offline teacher candidates on CPU."""

    sample_id: str
    teacher_videos: Tensor
    gt_video: Tensor
    gt_action: Tensor
    gt_action_mask: Tensor
    initial_frame: Tensor
    text_emb: Tensor
    frame_start: int
    history_video: Tensor | None
    history_action: Tensor | None
    history_frame_start: int
    video_valid_frames: Tensor
    action_valid_mask: Tensor


@dataclass(slots=True)
class TrainingBatch:
    """Collated causal conditions consumed by one training micro-step."""

    sample_ids: tuple[str, ...]
    teacher_videos: Tensor
    gt_video: Tensor
    gt_action: Tensor
    gt_action_mask: Tensor
    initial_frame: Tensor
    text_emb: Tensor
    frame_start: Tensor
    history_video: Tensor | None
    history_action: Tensor | None
    history_frame_start: Tensor
    video_valid_frames: Tensor
    action_valid_mask: Tensor

    @property
    def batch_size(self) -> int:
        return self.teacher_videos.shape[0]

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: torch.dtype | None = None,
    ) -> "TrainingBatch":
        """Move model inputs while preserving integer and Boolean dtypes."""

        def move(tensor: Tensor | None) -> Tensor | None:
            if tensor is None:
                return None
            target_dtype = dtype if tensor.is_floating_point() else None
            return tensor.to(device=device, dtype=target_dtype)

        return TrainingBatch(
            sample_ids=self.sample_ids,
            teacher_videos=move(self.teacher_videos),
            gt_video=move(self.gt_video),
            # Preserve GT/noise/FM-target precision until the model forward.
            gt_action=self.gt_action.to(device=device, dtype=torch.float32),
            gt_action_mask=move(self.gt_action_mask),
            initial_frame=move(self.initial_frame),
            text_emb=move(self.text_emb),
            frame_start=move(self.frame_start),
            history_video=move(self.history_video),
            history_action=move(self.history_action),
            history_frame_start=move(self.history_frame_start),
            video_valid_frames=move(self.video_valid_frames),
            action_valid_mask=move(self.action_valid_mask),
        )

    def condition(self) -> ActionSignatureCondition:
        """Build the shared causal condition for all generated candidates."""

        return ActionSignatureCondition(
            text_emb=self.text_emb,
            frame_start=self.frame_start,
            history_video=self.history_video,
            history_action=self.history_action,
            history_frame_start=self.history_frame_start,
        )


class DriftingTrainingDataset(Dataset[TrainingSample]):
    """Wrap the local LingBot-VA latent loader with an offline teacher bank.

    A bank file stores all available current chunks for one source sample:

    - ``teacher_videos``: ``[S,M,C,2,H,W]``
    - ``frame_starts``: ``[S]`` latent-frame offsets into the source sample

    ``S`` may vary by source sample. The distributed sampler selects one
    shared ``frame_start`` for every global batch. Only preceding frames
    become causal history. The matching current GT video/action chunk is
    returned separately, exclusively for action consistency/flow matching.
    """

    def __init__(self, config: TrainingConfig) -> None:
        if config.cfg_prob != 0:
            raise ValueError(
                "cfg_prob must be zero when using a condition-matched "
                "offline teacher-video bank"
            )

        self.config = config
        self.base_dataset = SafeMultiLatentLeRobotDataset(
            config,
            num_init_worker=max(config.num_workers, 1),
            unclipped_actions=True,
        )
        self._frame_starts_cache: dict[int, tuple[int, ...]] = {}

    def _bank_path(self, index: int) -> Path:
        child, local_index = resolve_latent_source(self.base_dataset, index)
        metadata = child.new_metas[local_index]
        return teacher_video_bank_file(
            self.config.teacher_video_bank_path,
            self.config.dataset_path,
            child.repo_id,
            episode_index=metadata["episode_index"],
            start_frame=metadata["start_frame"],
            end_frame=metadata["end_frame"],
        )

    def frame_starts_for_index(self, index: int) -> tuple[int, ...]:
        """Return cached valid starts without loading source latents."""

        cached = self._frame_starts_cache.get(index)
        if cached is None:
            bank = self._load_bank(self._bank_path(index))
            starts = bank[FRAME_STARTS_KEY]
            cached = tuple(int(value) for value in starts.tolist())
            if not cached:
                raise ValueError(f"teacher bank has no frame starts: {index}")
            self._frame_starts_cache[index] = cached
        return cached


    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int | tuple[int, int]) -> TrainingSample:
        requested_frame_start = None
        if isinstance(index, tuple):
            source_index, requested_frame_start = index
        else:
            source_index = index
        child, local_index = resolve_latent_source(
            self.base_dataset, source_index
        )
        metadata = child.new_metas[local_index]
        source = child[local_index]

        latents = source["latents"].contiguous()
        gt_actions = source["actions"].contiguous()
        # Keep the pre-existing cached video/probe condition unchanged. The
        # action training pair separately consumes the full unclipped GT below.
        actions = gt_actions.clamp(-1.5, 1.5)
        action_mask = source["actions_mask"].to(dtype=torch.bool).contiguous()
        text_emb = source["text_emb"].contiguous()
        validate_latent_source(self.config, latents, actions, action_mask)

        bank_path = self._bank_path(source_index)
        bank = self._load_bank(bank_path)
        teacher_videos = bank[TEACHER_VIDEOS_KEY]
        frame_starts = bank[FRAME_STARTS_KEY]

        if requested_frame_start is None:
            row = int(torch.randint(frame_starts.numel(), ()).item())
        else:
            matches = torch.nonzero(
                frame_starts.eq(requested_frame_start), as_tuple=False
            ).flatten()
            if matches.numel() != 1:
                raise ValueError(
                    f"requested frame_start is unavailable: {requested_frame_start}"
                )
            row = int(matches.item())
        frame_start = int(frame_starts[row].item())
        current_frames = teacher_videos.shape[3]
        if frame_start % self.config.frame_chunk_size != 0:
            raise ValueError(
                f"bank frame_start must align to a complete chunk: {bank_path}"
            )
        if frame_start < 0 or frame_start + current_frames > latents.shape[1]:
            raise ValueError(
                f"bank frame_start {frame_start} is outside source latents: "
                f"{bank_path}"
            )

        selected_teachers = teacher_videos[row].contiguous().clone()
        expected_video_tail = (
            latents.shape[0],
            self.config.frame_chunk_size,
            latents.shape[2],
            latents.shape[3],
        )
        if tuple(selected_teachers.shape[1:]) != expected_video_tail:
            raise ValueError(
                f"teacher bank video tail must be {expected_video_tail}, got "
                f"{tuple(selected_teachers.shape[1:])}: {bank_path}"
            )

        history_video = None
        history_action = None
        if frame_start > 0:
            history_video = latents[:, :frame_start].contiguous()
            history_action = actions[:, :frame_start].contiguous()

        current_slice = slice(frame_start, frame_start + current_frames)
        current_action_mask = action_mask[:, current_slice].clone()
        video_valid_frames = torch.ones(current_frames, dtype=torch.bool)
        if frame_start == 0:
            video_valid_frames[0] = False
            current_action_mask[:, 0] = False

        relative_repo = Path(child.repo_id).resolve().relative_to(
            self.config.dataset_path
        )
        sample_id = (
            f"{relative_repo.as_posix()}/"
            f"episode_{int(metadata['episode_index']):06d}_"
            f"{int(metadata['start_frame'])}_{int(metadata['end_frame'])}"
            f"@latent_{frame_start:04d}"
        )
        return TrainingSample(
            sample_id=sample_id,
            teacher_videos=selected_teachers,
            gt_video=latents,
            gt_action=gt_actions,
            gt_action_mask=action_mask,
            initial_frame=latents[:, :1].contiguous(),
            text_emb=text_emb,
            frame_start=frame_start,
            history_video=history_video,
            history_action=history_action,
            history_frame_start=0,
            video_valid_frames=video_valid_frames,
            action_valid_mask=current_action_mask.contiguous(),
        )

    def _load_bank(self, path: Path) -> Mapping[str, Tensor]:
        if not path.is_file():
            raise FileNotFoundError(
                f"offline teacher-video bank entry not found: {path}"
            )
        entry = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(entry, Mapping):
            raise TypeError(f"teacher bank entry must be a mapping: {path}")
        if TEACHER_VIDEOS_KEY not in entry or FRAME_STARTS_KEY not in entry:
            raise KeyError(
                f"teacher bank entry needs {TEACHER_VIDEOS_KEY!r} and "
                f"{FRAME_STARTS_KEY!r}: {path}"
            )

        videos = entry[TEACHER_VIDEOS_KEY]
        starts = entry[FRAME_STARTS_KEY]
        if not isinstance(videos, Tensor) or videos.ndim != 6:
            raise ValueError(
                "teacher_videos must have shape [S,M,C,F,H,W]: "
                f"{path}"
            )
        if videos.shape[1] != self.config.teacher_candidate_count:
            raise ValueError(
                f"teacher candidate count must be "
                f"{self.config.teacher_candidate_count}: {path}"
            )
        if videos.shape[3] != self.config.frame_chunk_size:
            raise ValueError(
                f"teacher videos must contain "
                f"{self.config.frame_chunk_size} frames: {path}"
            )
        if not isinstance(starts, Tensor):
            starts = torch.as_tensor(starts)
        starts = starts.to(dtype=torch.long).reshape(-1)
        if starts.numel() == 0 or starts.numel() != videos.shape[0]:
            raise ValueError(
                "frame_starts must contain one value per stored chunk: "
                f"{path}"
            )
        return {
            TEACHER_VIDEOS_KEY: videos,
            FRAME_STARTS_KEY: starts,
        }


class SynchronizedFrameStartSampler(Sampler[tuple[int, int]]):
    """Keep every local/global batch on one causal history length."""

    def __init__(
        self,
        dataset: DriftingTrainingDataset,
        *,
        num_replicas: int,
        rank: int,
        batch_size: int,
        seed: int,
        shuffle: bool = True,
    ) -> None:
        if num_replicas < 1:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.global_batch_size = num_replicas * batch_size
        self.num_global_batches = len(dataset) // self.global_batch_size
        if self.num_global_batches < 1:
            raise ValueError("dataset is smaller than one global batch")

    def __len__(self) -> int:
        return self.num_global_batches * self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)

        def shuffled(values: list[Any]) -> list[Any]:
            if not self.shuffle or len(values) < 2:
                return list(values)
            order = torch.randperm(len(values), generator=generator).tolist()
            return [values[index] for index in order]

        buckets: dict[tuple[int, ...], list[int]] = {}
        for source_index in range(len(self.dataset)):
            signature = self.dataset.frame_starts_for_index(source_index)
            buckets.setdefault(signature, []).append(source_index)

        global_batches: list[list[int]] = []
        leftovers: list[int] = []
        for signature in sorted(buckets):
            indices = shuffled(buckets[signature])
            full_count = len(indices) // self.global_batch_size
            split = full_count * self.global_batch_size
            for start in range(0, split, self.global_batch_size):
                global_batches.append(
                    indices[start : start + self.global_batch_size]
                )
            leftovers.extend(indices[split:])

        leftovers = shuffled(leftovers)
        split = (len(leftovers) // self.global_batch_size) * (
            self.global_batch_size
        )
        for start in range(0, split, self.global_batch_size):
            global_batches.append(
                leftovers[start : start + self.global_batch_size]
            )
        if len(global_batches) != self.num_global_batches:
            raise RuntimeError("synchronized sampler built the wrong length")
        global_batches = shuffled(global_batches)

        local_start = self.rank * self.batch_size
        local_stop = local_start + self.batch_size
        for global_batch in global_batches:
            common_starts = set(
                self.dataset.frame_starts_for_index(global_batch[0])
            )
            for source_index in global_batch[1:]:
                common_starts.intersection_update(
                    self.dataset.frame_starts_for_index(source_index)
                )
            if not common_starts:
                raise RuntimeError("global batch has no common frame_start")
            ordered_starts = sorted(common_starts)
            selected_index = int(
                torch.randint(
                    len(ordered_starts),
                    (),
                    generator=generator,
                ).item()
            )
            selected_start = ordered_starts[selected_index]
            for source_index in global_batch[local_start:local_stop]:
                yield source_index, selected_start


class HistoryStressSampler(SynchronizedFrameStartSampler):
    """Bounded diagnostic order: long history, accumulation boundary, empty.

    Source selection stays rank-synchronized and uses only available bank
    chunks. This deliberately biased sampler is never used for formal runs.
    """

    def __init__(
        self, dataset: DriftingTrainingDataset, *,
        config: TrainingConfig,
    ) -> None:
        super().__init__(
            dataset, num_replicas=config.world_size, rank=config.rank,
            batch_size=config.batch_size, seed=config.seed,
        )
        by_start: dict[int, list[int]] = {}
        for index in range(len(dataset)):
            for frame_start in set(dataset.frame_starts_for_index(index)):
                by_start.setdefault(frame_start, []).append(index)
        # Rare longest chunks may number fewer than ranks. Repeat only in
        # this diagnostic so every rank exercises the actual longest shape.
        self.sources_by_start = by_start
        if not self.sources_by_start:
            raise ValueError("history stress test needs available chunks")
        longest = max(self.sources_by_start)
        boundary = config.max_unsynced_history_frames
        required = {longest, boundary}
        if config.max_train_steps == 3:
            required.add(0)
        if longest <= boundary or not required.issubset(self.sources_by_start):
            raise ValueError(
                "history stress test needs chunks at the sync boundary "
                "and beyond it (plus zero history for a third update)"
            )
        accumulation = config.gradient_accumulation_steps
        updates = [
            [longest] + [boundary] * (accumulation - 1),
            [boundary] * (accumulation - 1) + [longest],
            [0] * accumulation,
        ][:config.max_train_steps]
        self.history_schedule = tuple(start for update in updates for start in update)
        if self.rank == 0:
            logging.getLogger(__name__).info(
                "History stress test: latent history frames per update=%s; "
                "sources at longest=%d, boundary=%d",
                updates, len(self.sources_by_start[longest]),
                len(self.sources_by_start[boundary]),
            )

    def __len__(self) -> int:
        return len(self.history_schedule) * self.batch_size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        local_start = self.rank * self.batch_size
        local_stop = local_start + self.batch_size
        for frame_start in self.history_schedule:
            sources = self.sources_by_start[frame_start]
            order = torch.randperm(len(sources), generator=generator).tolist()
            repeats = (self.global_batch_size + len(order) - 1) // len(order)
            global_order = (order * repeats)[:self.global_batch_size]
            for offset in global_order[local_start:local_stop]:
                yield sources[offset], frame_start


def collate_training_samples(samples: list[TrainingSample]) -> TrainingBatch:
    """Collate samples whose causal histories have the same tensor shape."""

    if not samples:
        raise ValueError("cannot collate an empty sample list")

    if len({sample.gt_video.shape for sample in samples}) != 1:
        raise ValueError("full GT sequences in one local batch must have equal lengths; use batch_size=1")
    has_history = [sample.history_video is not None for sample in samples]
    if len(set(has_history)) != 1:
        raise ValueError(
            "one batch cannot mix empty and non-empty causal histories; "
            "use batch_size=1 or bucket samples by frame_start"
        )

    history_video = None
    history_action = None
    if has_history[0]:
        history_shapes = {sample.history_video.shape for sample in samples}
        action_shapes = {sample.history_action.shape for sample in samples}
        if len(history_shapes) != 1 or len(action_shapes) != 1:
            raise ValueError(
                "causal histories in one batch must have equal lengths; "
                "use batch_size=1 or bucket samples by frame_start"
            )
        history_video = torch.stack(
            [sample.history_video for sample in samples]
        )
        history_action = torch.stack(
            [sample.history_action for sample in samples]
        )

    return TrainingBatch(
        sample_ids=tuple(sample.sample_id for sample in samples),
        teacher_videos=torch.stack(
            [sample.teacher_videos for sample in samples]
        ),
        gt_video=torch.stack([sample.gt_video for sample in samples]),
        gt_action=torch.stack([sample.gt_action for sample in samples]),
        gt_action_mask=torch.stack([sample.gt_action_mask for sample in samples]),
        initial_frame=torch.stack(
            [sample.initial_frame for sample in samples]
        ),
        text_emb=torch.stack([sample.text_emb for sample in samples]),
        frame_start=torch.tensor(
            [sample.frame_start for sample in samples],
            dtype=torch.long,
        ),
        history_video=history_video,
        history_action=history_action,
        history_frame_start=torch.tensor(
            [sample.history_frame_start for sample in samples],
            dtype=torch.long,
        ),
        video_valid_frames=torch.stack(
            [sample.video_valid_frames for sample in samples]
        ),
        action_valid_mask=torch.stack(
            [sample.action_valid_mask for sample in samples]
        ),
    )


def build_training_dataloader(
    config: TrainingConfig,
    *,
    dataset: Dataset[TrainingSample] | None = None,
) -> DataLoader[TrainingBatch]:
    """Build the rank-aware shuffled loader used by the later trainer."""

    training_dataset = dataset or DriftingTrainingDataset(config)
    if not hasattr(training_dataset, "frame_starts_for_index"):
        raise TypeError(
            "training dataset must expose frame_starts_for_index"
        )
    if config.history_stress_test:
        sampler = HistoryStressSampler(training_dataset, config=config)
    else:
        sampler = SynchronizedFrameStartSampler(
            training_dataset,
            num_replicas=config.world_size,
            rank=config.rank,
            batch_size=config.batch_size,
            seed=config.seed,
            shuffle=True,
        )

    generator = torch.Generator()
    generator.manual_seed(config.seed + config.rank)
    return DataLoader(
        training_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
        collate_fn=collate_training_samples,
        generator=generator,
    )
