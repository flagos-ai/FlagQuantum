"""Unstable specialized simulation workflows."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PUBLIC_NAMES = ("run_tebd",)
__all__ = _PUBLIC_NAMES


def __getattr__(name: str) -> Any:
    if name in {"TEBDResult", "run_tebd"}:
        return getattr(import_module("flagquantum.simulation.tebd"), name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
