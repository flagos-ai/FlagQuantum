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
    "select_mps_crossover_decision",
    "train_distributed_mps",
    "validate_production_mps_workload",
)

_EXPORTS = {
    "MPSAcceptanceGates": (
        "flagquantum.runtime.backends.mps.production",
        "MPSAcceptanceGates",
    ),
    "MPSCrossoverMeasurement": (
        "flagquantum.runtime.backends.mps.production",
        "MPSCrossoverMeasurement",
    ),
    "MPSForwardLifetimeError": (
        "flagquantum.runtime.backends.mps.errors",
        "MPSForwardLifetimeError",
    ),
    "MPSFullMaterializationError": (
        "flagquantum.runtime.backends.mps.errors",
        "MPSFullMaterializationError",
    ),
    "MPSProductionAcceptanceError": (
        "flagquantum.runtime.backends.mps.production",
        "MPSProductionAcceptanceError",
    ),
    "MPSProductionPlan": (
        "flagquantum.runtime.backends.mps.production",
        "MPSProductionPlan",
    ),
    "MPSProductionSupport": (
        "flagquantum.runtime.backends.mps.production",
        "MPSProductionSupport",
    ),
    "NonlocalMPSCompilationError": (
        "flagquantum.runtime.backends.mps.errors",
        "NonlocalMPSCompilationError",
    ),
    "build_mps_release_artifact": (
        "flagquantum.runtime.backends.mps.production",
        "build_mps_release_artifact",
    ),
    "execute_torch_distributed_mps_forward": (
        "flagquantum.runtime.backends.mps.forward",
        "execute_torch_distributed_mps_forward",
    ),
    "execute_torch_distributed_mps_reverse": (
        "flagquantum.runtime.backends.mps.reverse",
        "execute_torch_distributed_mps_reverse",
    ),
    "plan_production_mps": (
        "flagquantum.runtime.backends.mps.production",
        "plan_production_mps",
    ),
    "select_mps_crossover_decision": (
        "flagquantum.runtime.backends.mps.production",
        "select_mps_crossover_decision",
    ),
    "train_distributed_mps": (
        "flagquantum.runtime.backends.mps.training",
        "train_distributed_mps",
    ),
    "validate_production_mps_workload": (
        "flagquantum.runtime.backends.mps.production",
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
