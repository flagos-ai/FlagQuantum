"""Matrix-product-state backend boundary."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = (
    "MPSAcceptanceGates",
    "MPSCrossoverMeasurement",
    "MPSForwardLifetimeError",
    "MPSFullMaterializationError",
    "MPSProductionAcceptanceError",
    "MPSProductionPlan",
    "MPSProductionSupport",
    "NonlocalMPSCompilationError",
    "build_mps_release_artifact",
    "execute_torch_distributed_mps_forward",
    "execute_torch_distributed_mps_reverse",
    "plan_production_mps",
    "run_distributed_mps",
    "select_mps_crossover_decision",
    "train_distributed_mps",
    "validate_production_mps_workload",
)

_EXPORTS = {
    "MPSAcceptanceGates": (
        "flagquantum.runtime.executors.mps.production",
        "MPSAcceptanceGates",
    ),
    "MPSCrossoverMeasurement": (
        "flagquantum.runtime.executors.mps.production",
        "MPSCrossoverMeasurement",
    ),
    "MPSForwardLifetimeError": (
        "flagquantum.runtime.executors.mps.errors",
        "MPSForwardLifetimeError",
    ),
    "MPSFullMaterializationError": (
        "flagquantum.runtime.executors.mps.errors",
        "MPSFullMaterializationError",
    ),
    "MPSProductionAcceptanceError": (
        "flagquantum.runtime.executors.mps.production",
        "MPSProductionAcceptanceError",
    ),
    "MPSProductionPlan": (
        "flagquantum.runtime.executors.mps.production",
        "MPSProductionPlan",
    ),
    "MPSProductionSupport": (
        "flagquantum.runtime.executors.mps.production",
        "MPSProductionSupport",
    ),
    "NonlocalMPSCompilationError": (
        "flagquantum.runtime.executors.mps.errors",
        "NonlocalMPSCompilationError",
    ),
    "build_mps_release_artifact": (
        "flagquantum.runtime.executors.mps.production",
        "build_mps_release_artifact",
    ),
    "execute_torch_distributed_mps_forward": (
        "flagquantum.runtime.executors.mps.forward",
        "execute_torch_distributed_mps_forward",
    ),
    "execute_torch_distributed_mps_reverse": (
        "flagquantum.runtime.executors.mps.reverse",
        "execute_torch_distributed_mps_reverse",
    ),
    "plan_production_mps": (
        "flagquantum.runtime.executors.mps.production",
        "plan_production_mps",
    ),
    "run_distributed_mps": (
        "flagquantum.runtime.executors.mps.execution",
        "run_distributed_mps",
    ),
    "select_mps_crossover_decision": (
        "flagquantum.runtime.executors.mps.production",
        "select_mps_crossover_decision",
    ),
    "train_distributed_mps": (
        "flagquantum.runtime.executors.mps.training",
        "train_distributed_mps",
    ),
    "validate_production_mps_workload": (
        "flagquantum.runtime.executors.mps.production",
        "validate_production_mps_workload",
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
