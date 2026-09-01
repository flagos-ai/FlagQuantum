"""Unstable advanced planning entry point."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PUBLIC_NAMES = ("plan_advanced",)
__all__ = _PUBLIC_NAMES


def __getattr__(name: str) -> Any:
    if name in {"plan_advanced", "plan_runtime_selection", "select_backend_by_cost"}:
        return getattr(import_module("flagquantum.compilation.planner"), name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
