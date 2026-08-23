"""FlagQuantum stable API with lazy compatibility exports."""

from __future__ import annotations

import warnings
from importlib import import_module
from typing import Any

from .version import __version__, get_version

__author__ = "FlagQuantum Team"
__license__ = "Apache-2.0"

# Stable root surface. Advanced historical exports remain available lazily from
# flagquantum.api until the compatibility removal version.
__all__ = (
    "Circuit",
    "ExecutionResult",
    "MeasurementResult",
    "NoiseModel",
    "CircuitIR",
    "Instruction",
    "IR_VERSION",
    "IRSerializationError",
    "IRValidationError",
    "HybridParallelPlan",
    "MeasurementNode",
    "Module",
    "GateInfo",
    "ObservableNode",
    "Parameter",
    "ParameterExpression",
    "RuntimePolicy",
    "TrainingResult",
    "compile_for_backend",
    "plan",
    "plan_runtime_selection",
    "select_backend_by_cost",
    "run",
    "run_native",
    "run_target",
    "run_mps",
    "run_noisy_mps",
    "run_tensor_network",
    "noisy_density_matrix",
    "resolve_device",
    "distributed_tensor_network_amplitude",
    "distributed_tensor_network_amplitudes",
    "distributed_tensor_network_expectation",
    "distributed_tensor_network_expectations",
    "tensor_network_amplitude",
    "tensor_network_amplitudes",
    "tensor_network_expectations",
    "train",
    "train_distributed_statevector",
    "train_distributed_mps",
    "MPSReverseCheckpointPolicy",
    "MPSAcceptanceGates",
    "MPSCrossoverMeasurement",
    "MPSProductionAcceptanceError",
    "MPSProductionPlan",
    "MPSProductionSupport",
    "PauliMeasurementPlan",
    "plan_production_mps",
    "validate_production_mps_workload",
    "build_mps_release_artifact",
    "create_deployment_package",
    "create_pauli_measurement_plan",
    "deploy_circuit",
    "hamiltonian_expectation_from_grouped_counts",
    "__version__",
    "get_version",
    "gate_info",
    "info",
    "hello",
    "experimental",
)

COMPATIBILITY_EXPORT_OWNER = "FlagQuantum core maintainers"
COMPATIBILITY_EXPORT_REMOVAL_VERSION = "0.3.0"

_DEPRECATED_INTERNAL_ROOT_EXPORTS = {
    "DistributedEvidenceContract",
    "DistributedTransportEvidence",
    "JAXDistributedQuantumPlan",
    "JAXMPSRankShardState",
    "JAXStatevectorShardState",
    "JAXTNSliceRankState",
    "StatevectorShard",
    "StatevectorShardState",
}


def _compat_api() -> Any:
    return import_module(".api", __name__)


def __getattr__(name: str) -> Any:
    if name == "experimental":
        return import_module(".experimental", __name__)
    if name == "Circuit":
        return getattr(import_module(".circuit", __name__), name)
    if name in {"ExecutionResult", "MeasurementResult", "Module", "RuntimePolicy"}:
        return getattr(import_module(".runtime.contracts", __name__), name)
    if name == "run":
        return getattr(import_module(".runtime.execution", __name__), name)
    if name in {"TrainingResult", "train"}:
        return getattr(import_module(".runtime.training", __name__), name)
    if name == "HybridParallelPlan":
        return getattr(import_module(".runtime.planning", __name__), name)
    if name in _DEPRECATED_INTERNAL_ROOT_EXPORTS:
        warnings.warn(
            f"flagquantum.{name} is an internal compatibility export; use "
            f"flagquantum.experimental.{name}. Root access will be removed in "
            f"version {COMPATIBILITY_EXPORT_REMOVAL_VERSION}.",
            DeprecationWarning,
            stacklevel=2,
        )
    api = _compat_api()
    try:
        return getattr(api, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_compat_api())))


def info() -> dict[str, str]:
    return {
        "name": "flagquantum",
        "version": __version__,
        "author": __author__,
        "license": __license__,
    }


def hello() -> None:
    print(f"FlagQuantum v{__version__} - Distributed Quantum Computing Framework")
