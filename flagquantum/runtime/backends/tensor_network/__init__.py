"""Tensor-network backend boundary."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ("run_distributed_tensor_network", "run_tensor_network")

_EXPORTS = {
    "run_distributed_tensor_network": (
        "flagquantum.runtime.distributed.engine",
        "run_distributed_tensor_network",
    ),
    "run_tensor_network": (
        "flagquantum.simulation.tensor",
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
