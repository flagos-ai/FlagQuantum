"""Unstable third-party adapter implementations."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_ADAPTER_NAMES = ("pennylane", "qiskit")
__all__ = _ADAPTER_NAMES


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    return import_module(f"flagquantum.interop.{name}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
