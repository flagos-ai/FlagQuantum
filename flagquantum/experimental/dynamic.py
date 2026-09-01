"""Unstable dynamic-circuit execution, deployment, and conformance APIs."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_RUNTIME_EXPORTS = {
    "DynamicBackendCompatibility",
    "DynamicExecutionResult",
    "assess_dynamic_backend",
    "create_dynamic_deployment_package",
    "deploy_dynamic_circuit",
    "export_braket_iqm_dynamic_qasm3",
    "export_dynamic_qasm3",
    "export_dynamic_qasm3_for_backend",
    "route_dynamic_circuit",
    "run_dynamic",
}
_PUBLIC_NAMES = (
    "assess_dynamic_backend",
    "create_dynamic_deployment_package",
    "deploy_dynamic_circuit",
    "export_dynamic_qasm3",
    "route_dynamic_circuit",
    "run_dynamic",
)
__all__ = _PUBLIC_NAMES


def __getattr__(name: str) -> Any:
    if name in __all__:
        return getattr(import_module("flagquantum.runtime.dynamic"), name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
