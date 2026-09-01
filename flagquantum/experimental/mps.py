"""Unstable user-facing MPS production workflows."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PUBLIC_NAMES = ("plan_production_mps", "validate_production_mps_workload")
__all__ = _PUBLIC_NAMES


def __getattr__(name: str) -> Any:
    if name in __all__:
        return getattr(import_module("flagquantum.runtime.backends.mps"), name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
