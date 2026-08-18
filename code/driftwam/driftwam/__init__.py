"""driftwam - diagnostics for drifting-based one-step distillation of LingBot-VA.

Lives alongside the Flash-WAM checkout and never writes into it. Flash-WAM's
model, dataset and scheduler code is reached through `driftwam.bootstrap`.
"""

__version__ = "0.1.0"

__all__ = ["bootstrap", "features", "metrics", "paths", "sampling", "shapes"]
