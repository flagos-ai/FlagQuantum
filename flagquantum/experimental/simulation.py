"""Unstable specialized simulation workflows."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PUBLIC_NAMES = ("run_tebd",)
__all__ = _PUBLIC_NAMES


def __getattr__(name: str) -> Any:
    if name in __all__:
        return getattr(import_module("flagquantum.simulation.mps.tebd"), name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
