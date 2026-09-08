"""Stable expert interfaces for backend-native execution.

Most users should call :func:`flagquantum.run`. These interfaces are for
callers that intentionally depend on backend-native result objects.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = (
    "resolve_device",
    "run_mps",
    "run_native",
    "run_noisy_mps",
    "run_target",
    "run_tensor_network",
)

_EXPORTS = {
    "resolve_device": ("flagquantum.runtime.backend_registry", "resolve_device"),
    "run_mps": ("flagquantum.simulation.mps.entrypoints", "run_mps"),
    "run_native": ("flagquantum.runtime.execution", "run_native"),
    "run_noisy_mps": ("flagquantum.runtime.executors.mps.noisy", "run_noisy_mps"),
    "run_target": ("flagquantum.runtime.target_execution", "run_target"),
    "run_tensor_network": (
        "flagquantum.simulation.tensor_network.entrypoints",
        "run_tensor_network",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, symbol = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    return getattr(import_module(module_name), symbol)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
