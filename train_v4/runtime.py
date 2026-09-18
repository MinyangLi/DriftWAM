"""Activation helpers for the external LingBot-VA runtime."""

from __future__ import annotations

import importlib.machinery
import sys
import types
from pathlib import Path


def _prepend_sys_path(path: Path) -> None:
    value = str(path)
    while value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)


def _install_flash_attn_stub() -> None:
    """Provide the compatibility symbols used by both upstream runtimes."""

    for module_name in ("flash_attn_interface", "flash_attn"):
        if module_name in sys.modules:
            continue
        try:
            __import__(module_name)
        except ImportError:
            stub = types.ModuleType(module_name)
            stub.__spec__ = importlib.machinery.ModuleSpec(module_name, None)
            stub.__version__ = "0.0.0"
            stub.flash_attn_func = None
            stub.flash_attn_varlen_func = None
            sys.modules[module_name] = stub


def activate_lingbot_va(lingbot_va_root: str | Path) -> Path:
    """Make the original LingBot-VA runtime importable."""

    root = Path(lingbot_va_root).expanduser().resolve()
    wan_va_root = root / "wan_va"
    if not wan_va_root.is_dir():
        raise FileNotFoundError(
            f"LingBot-VA wan_va directory not found: {wan_va_root}"
        )
    _prepend_sys_path(root)
    _install_flash_attn_stub()
    return wan_va_root
