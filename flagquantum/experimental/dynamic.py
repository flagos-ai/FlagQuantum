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
_CONFORMANCE_EXPORTS = {
    "BRAKET_IQM_DYNAMIC_FEATURES",
    "DynamicConformanceCase",
    "DynamicConformanceResult",
    "DynamicFeatureReport",
    "DynamicFeatureSet",
    "LOCAL_TRAJECTORY_FEATURES",
    "QISKIT_AER_DYNAMIC_FEATURES",
    "assess_dynamic_features",
    "dynamic_conformance_cases",
    "run_dynamic_conformance",
    "run_qiskit_aer_dynamic",
    "run_qiskit_aer_qasm3_round_trip",
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
    if name in _RUNTIME_EXPORTS:
        return getattr(import_module("flagquantum.runtime.dynamic"), name)
    if name in _CONFORMANCE_EXPORTS:
        return getattr(import_module("flagquantum.runtime.dynamic_conformance"), name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
