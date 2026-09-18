"""Filesystem-only deletion before replacing the current run's checkpoints."""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from time import perf_counter


logger = logging.getLogger(__name__)


def checkpoints_to_remove(current: Path, keep_count: int) -> list[Path]:
    """Reserve one retention slot for the checkpoint about to be written."""

    match = re.fullmatch(r"step_(\d+)", current.name)
    if keep_count < 1 or match is None or current.is_symlink():
        raise ValueError(f"invalid checkpoint retention request: {current}")
    if current.exists():
        raise FileExistsError(f"checkpoint already exists: {current}")
    if not current.parent.exists():
        return []
    current_step = int(match[1])
    candidates = []
    interrupted = []
    for path in current.parent.iterdir():
        if path.is_symlink() or not path.is_dir():
            continue
        old = re.fullmatch(r"step_(\d+)", path.name)
        deleting = re.fullmatch(r"\.step_(\d+)\.deleting", path.name)
        if old and int(old[1]) < current_step and (
            path / "CHECKPOINT_COMPLETE"
        ).is_file():
            candidates.append((int(old[1]), path))
        elif deleting and int(deleting[1]) < current_step:
            interrupted.append(path)
    candidates.sort(key=lambda item: item[0], reverse=True)
    return interrupted + [path for _, path in candidates[keep_count - 1:]]


def reclaimable_checkpoint_bytes(current: Path, keep_count: int) -> int:
    """Conservative disk credit for old files deleted before the first save.

    Do not credit symlinks, shared hard links or unallocated sparse file extents.
    This matters on resume, where a retained checkpoint already consumes space.
    """

    total = 0
    for checkpoint in checkpoints_to_remove(current, keep_count):
        for path in checkpoint.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            stat = path.stat()
            if stat.st_nlink == 1:
                total += min(stat.st_size, stat.st_blocks * 512)
    return total


def prune_old_checkpoints(current: Path, keep_count: int) -> None:
    """Delete obsolete checkpoints synchronously BEFORE writing ``current``.

    With keep_count=1 this deliberately leaves no recovery checkpoint until the
    new save completes. Rename first so interrupted deletion is never resumable.
    Only rank 0 calls this function; it performs no distributed/GPU operations.
    """

    started = perf_counter()
    candidates = checkpoints_to_remove(current, keep_count)
    for stale in candidates:
        if stale.name.startswith("step_"):
            deleting = stale.with_name(f".{stale.name}.deleting")
            stale.rename(deleting)
        else:
            deleting = stale
        shutil.rmtree(deleting)
        logger.info("Deleted old checkpoint before saving %s: %s", current.name, stale)
    logger.info(
        "Checkpoint pre-save deletion: count=%d duration_seconds=%.3f",
        len(candidates), perf_counter() - started,
    )
