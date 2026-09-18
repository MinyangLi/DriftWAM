"""Video drifting and optional action-execution objectives."""

from .action_execution_loss import (
    ActionExecutionLoss,
    ActionExecutionLossResult,
)
from .action_consistency_loss import (
    ActionConsistencyLoss,
    ActionConsistencyLossResult,
)

__all__ = [
    "ActionConsistencyLoss",
    "ActionConsistencyLossResult",
    "ActionExecutionLoss",
    "ActionExecutionLossResult",
]
