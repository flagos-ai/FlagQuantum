"""Optional JAX backend boundary.

Importing this module exposes contracts without initializing JAX devices.
The optional dependency is loaded only when an execution function is called.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = (
    "JAXDistributedQuantumPlan",
    "compile_quantum_kernel",
    "plan_jax_distributed_quantum_backend",
    "plan_jax_sharded_mps_training",
    "plan_jax_sharded_statevector_training",
    "run_jax_sharded_mps",
    "run_jax_sharded_statevector",
    "run_jax_sharded_tensor_network",
)

_EXPORTS = {
    "JAXDistributedQuantumPlan": (
        "flagquantum.runtime.executors.jax.planning_core",
        "JAXDistributedQuantumPlan",
    ),
    "compile_quantum_kernel": (
        "flagquantum.runtime.executors.jax.kernel",
        "compile_quantum_kernel",
    ),
    "plan_jax_distributed_quantum_backend": (
        "flagquantum.runtime.executors.jax.backend_dispatch",
        "plan_jax_distributed_quantum_backend",
    ),
    "plan_jax_sharded_mps_training": (
        "flagquantum.runtime.executors.jax.mps.planning",
        "plan_jax_sharded_mps_training",
    ),
    "plan_jax_sharded_statevector_training": (
        "flagquantum.runtime.executors.jax.statevector.training",
        "plan_jax_sharded_statevector_training",
    ),
    "run_jax_sharded_mps": (
        "flagquantum.runtime.executors.jax.mps.execution",
        "run_jax_sharded_mps",
    ),
    "run_jax_sharded_statevector": (
        "flagquantum.runtime.executors.jax.statevector.execution",
        "run_jax_sharded_statevector",
    ),
    "run_jax_sharded_tensor_network": (
        "flagquantum.runtime.executors.jax.tensor_network.execution",
        "run_jax_sharded_tensor_network",
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
