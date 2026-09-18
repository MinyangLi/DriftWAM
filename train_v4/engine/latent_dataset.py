"""Local training wrapper around the upstream LingBot-VA latent dataset."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..runtime import activate_lingbot_va


logger = logging.getLogger(__name__)


def _dataset_type(lingbot_va_root: str | Path) -> type[Any]:
    """Return LingBot-VA's clipped latent dataset implementation."""

    activate_lingbot_va(lingbot_va_root)
    from wan_va.dataset.lerobot_latent_dataset import LatentLeRobotDataset

    return LatentLeRobotDataset


class SafeMultiLatentLeRobotDataset:
    """Load all complete latent sub-datasets and skip incomplete ones."""

    def __init__(
        self, config: Any, num_init_worker: int = 1, *, unclipped_actions: bool = False,
    ) -> None:
        del num_init_worker
        dataset_type = _dataset_type(config.lingbot_va_root)
        repositories = sorted(
            info_path.parent.parent
            for info_path in config.dataset_path.rglob("meta/info.json")
        )
        self._datasets = []
        for repository in repositories:
            try:
                child = dataset_type(repo_id=str(repository), config=config)
                child.clip_normalized_actions = not unclipped_actions
                self._datasets.append(child)
            except Exception as exc:
                logger.warning(
                    "Skipping incomplete latent dataset %s: %s",
                    repository.name,
                    exc,
                )
        if not self._datasets:
            raise RuntimeError(
                f"no valid latent sub-datasets found under {config.dataset_path}"
            )
        logger.info(
            "Loaded %d/%d latent sub-datasets",
            len(self._datasets),
            len(repositories),
        )
        self.item_id_to_dataset_id, self.acc_dset_num = (
            self._build_index_maps()
        )

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self._datasets)

    def __getitem__(self, index: int):
        dataset_id = self.item_id_to_dataset_id[index]
        local_index = index - self.acc_dset_num[dataset_id]
        return self._datasets[dataset_id][local_index]

    def _build_index_maps(self) -> tuple[dict[int, int], dict[int, int]]:
        item_to_dataset = {}
        dataset_starts = {}
        item_index = 0
        for dataset_id, dataset in enumerate(self._datasets):
            dataset_starts[dataset_id] = item_index
            for _ in range(len(dataset)):
                item_to_dataset[item_index] = dataset_id
                item_index += 1
        return item_to_dataset, dataset_starts
