"""Detached GT video/action pair used by action consistency and flow matching."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(slots=True)
class ActionTrainingPair:
    """The aligned full ground-truth video/action sequence per condition."""

    video: Tensor
    clean_action: Tensor
    action_mask: Tensor
