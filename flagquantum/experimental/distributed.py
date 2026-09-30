"""Unstable task-level distributed workflows.

Only user-invokable workflows are discoverable. Low-level shard records,
executors, kernel counters, and evidence objects remain implementation details.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PUBLIC_NAMES = (
    "train_distributed_mps",
    "train_distributed_statevector",
    "train_distributed_tensor_network",
)
__all__ = _PUBLIC_NAMES

_MODULES = {
    "train_distributed_mps": "flagquantum.runtime.executors.mps",
    "train_distributed_statevector": "flagquantum.runtime.executors.statevector",
    "train_distributed_tensor_network": "flagquantum.runtime.executors.tensor_network",
}


def __getattr__(name: str) -> Any:
    if name in _MODULES:
        return getattr(import_module(_MODULES[name]), name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
