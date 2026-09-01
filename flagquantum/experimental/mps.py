"""Unstable user-facing MPS production workflows."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PUBLIC_NAMES = ("plan_production_mps", "validate_production_mps_workload")
__all__ = _PUBLIC_NAMES

_MPS_NAMES = {
    "MPSAcceptanceGates",
    "MPSCrossoverMeasurement",
    "MPSProductionAcceptanceError",
    "MPSProductionPlan",
    "MPSProductionSupport",
    "build_mps_release_artifact",
    "plan_production_mps",
    "validate_production_mps_workload",
}


def __getattr__(name: str) -> Any:
    if name in _MPS_NAMES:
        return getattr(import_module("flagquantum.runtime.backends.mps"), name)
    if name == "MPSReverseCheckpointPolicy":
        return getattr(import_module("flagquantum.runtime.backends.mps.records"), name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
