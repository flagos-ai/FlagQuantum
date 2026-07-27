"""Lazy registry for reproducible benchmark runners."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
import re

RunnerMain = Callable[[], int]

_RUNNERS: dict[str, tuple[str, str]] = {
    "environment_probe": ("benchmarks.runners.environment_probe", "main"),
    "statevector_weak_scaling": (
        "benchmarks.runners.statevector_weak_scaling",
        "main",
    ),
    "statevector_strong_scaling": (
        "benchmarks.runners.statevector_strong_scaling",
        "main",
    ),
    "statevector_training_scaling": (
        "benchmarks.runners.statevector_training_scaling",
        "main",
    ),
}


def register(name: str, module: str, attribute: str = "main") -> None:
    """Register a runner exactly once; duplicate names are rejected."""
    if not name or not module or not attribute:
        raise ValueError("runner registration requires name, module, and attribute")
    if re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", name) is None:
        raise ValueError("runner name must use lowercase snake_case")
    if name in _RUNNERS:
        raise ValueError(f"benchmark runner already registered: {name}")
    _RUNNERS[name] = (module, attribute)


def names() -> tuple[str, ...]:
    return tuple(sorted(_RUNNERS))


def resolve(name: str) -> RunnerMain:
    try:
        module_name, attribute = _RUNNERS[name]
    except KeyError as exc:
        raise KeyError(f"unknown benchmark runner: {name}") from exc
    return getattr(import_module(module_name), attribute)


__all__ = ["names", "register", "resolve"]
