"""Distributed tensor-network runtime implementation.

Import planning, execution, records, and evidence from their owning modules.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = (
    "ShardedTensorNetworkTrainingResult",
    "train_distributed_tensor_network",
)

_EXPORTS = {
    "ShardedTensorNetworkTrainingResult": (
        "flagquantum.runtime.executors.tensor_network.training",
        "ShardedTensorNetworkTrainingResult",
    ),
    "train_distributed_tensor_network": (
        "flagquantum.runtime.executors.tensor_network.training",
        "train_distributed_tensor_network",
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
