"""Statevector backend boundary."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = (
    "execute_torch_distributed_statevector",
    "execute_torch_distributed_statevector_reverse",
    "initialize_statevector_shard",
    "plan_distributed_statevector",
    "simulate_distributed_statevector_local",
    "StatevectorCheckpointPolicy",
    "TorchDistributedStatevectorGradientResult",
    "TorchDistributedStatevectorResult",
    "train_distributed_statevector",
)

_EXPORTS = {
    "TorchDistributedStatevectorResult": (
        "flagquantum.runtime.backends.statevector.forward",
        "TorchDistributedStatevectorResult",
    ),
    "execute_torch_distributed_statevector": (
        "flagquantum.runtime.backends.statevector.forward",
        "execute_torch_distributed_statevector",
    ),
    "initialize_statevector_shard": (
        "flagquantum.runtime.backends.statevector.forward",
        "initialize_statevector_shard",
    ),
    "plan_distributed_statevector": (
        "flagquantum.runtime.backends.statevector.state",
        "plan_distributed_statevector",
    ),
    "simulate_distributed_statevector_local": (
        "flagquantum.runtime.backends.statevector.state",
        "simulate_distributed_statevector_local",
    ),
    "StatevectorCheckpointPolicy": (
        "flagquantum.runtime.backends.statevector.reverse",
        "StatevectorCheckpointPolicy",
    ),
    "TorchDistributedStatevectorGradientResult": (
        "flagquantum.runtime.backends.statevector.reverse",
        "TorchDistributedStatevectorGradientResult",
    ),
    "execute_torch_distributed_statevector_reverse": (
        "flagquantum.runtime.backends.statevector.reverse",
        "execute_torch_distributed_statevector_reverse",
    ),
    "train_distributed_statevector": (
        "flagquantum.runtime.backends.statevector.training",
        "train_distributed_statevector",
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
