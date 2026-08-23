"""Compatibility facade for native MPS simulation."""

from importlib import import_module
from typing import Any

_MODULES = (
    "flagquantum.simulation.mps_execution",
    "flagquantum.simulation.mps_models",
    "flagquantum.simulation.mps_factorization",
    "flagquantum.simulation.mps_state",
)


def __getattr__(name: str) -> Any:
    for module_name in _MODULES:
        module = import_module(module_name)
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    names = set(globals())
    for module_name in _MODULES:
        names.update(dir(import_module(module_name)))
    return sorted(names)


__all__ = (  # noqa: F822
    "CompiledMPSOperation",
    "CompiledMPSProgram",
    "MPSAdaptiveBondPlan",
    "MPSAdaptiveRunResult",
    "MPSBondProfile",
    "MPSConfig",
    "MPSLocalRefinementPlan",
    "MPSMonteCarloResult",
    "MPSState",
    "MPSTruncationRecord",
    "merge_noisy_mps_results",
    "run_mps",
    "run_mps_adaptive",
    "run_noisy_mps",
    "run_noisy_mps_trajectory",
)
