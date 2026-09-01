"""Unstable task-level distributed workflows.

Only user-invokable workflows are discoverable. Low-level shard records,
executors, kernel counters, and evidence objects remain implementation details.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PUBLIC_NAMES = (
    "distributed_tensor_network_amplitude",
    "distributed_tensor_network_amplitudes",
    "distributed_tensor_network_expectation",
    "distributed_tensor_network_expectations",
    "train_distributed_mps",
    "train_distributed_statevector",
)
__all__ = _PUBLIC_NAMES


def __getattr__(name: str) -> Any:
    if name in {
        "distributed_tensor_network_amplitude",
        "distributed_tensor_network_amplitudes",
        "distributed_tensor_network_expectation",
        "distributed_tensor_network_expectations",
    }:
        return getattr(
            import_module("flagquantum.runtime.distributed.tensor_network_execution"),
            name,
        )
    if name in {"train_distributed_mps", "train_distributed_statevector"}:
        module = {
            "train_distributed_mps": "flagquantum.runtime.backends.mps",
            "train_distributed_statevector": "flagquantum.runtime.backends.statevector",
        }[name]
        return getattr(import_module(module), name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
