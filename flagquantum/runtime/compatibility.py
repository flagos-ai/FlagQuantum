"""Private bridge for the frozen :mod:`flagquantum.api` surface.

This module is not a public runtime API. It centralizes legacy symbol lookup so
the compatibility aggregator does not depend on implementation paths. New code
must import from the explicit modules under :mod:`flagquantum.runtime`.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__: tuple[str, ...] = ()

_MODULE_ALIASES = {
    "audit": "flagquantum.runtime.audit.engine",
    "backends": "flagquantum.runtime.backend_registry",
    "distributed": "flagquantum.runtime.distributed.engine",
    "execution": "flagquantum.runtime.execution",
    "local_preflight": "flagquantum.runtime.local_preflight",
    "operator_backends": "flagquantum.runtime.operator_backends",
    "parity": "flagquantum.runtime.parity",
    "runtime": "flagquantum.runtime.configuration",
}

_SYMBOL_ALIASES = {
    "Phase4StatevectorClaimabilityGate": (
        "flagquantum.runtime.audit.schema",
        "StatevectorTrainingClaimabilityGate",
    ),
    "Phase5MPSBackwardReadinessGate": (
        "flagquantum.runtime.audit.schema",
        "MPSBackwardReadinessGate",
    ),
    "PHASE4_CLAIMABILITY_STATUSES": (
        "flagquantum.runtime.audit.vocabulary",
        "STATEVECTOR_TRAINING_CLAIMABILITY_STATUSES",
    ),
    "PHASE5_MPS_BACKWARD_READINESS_STATUSES": (
        "flagquantum.runtime.audit.vocabulary",
        "MPS_BACKWARD_READINESS_STATUSES",
    ),
    "evaluate_phase4_statevector_claimability": (
        "flagquantum.runtime.audit.engine",
        "evaluate_statevector_training_claimability",
    ),
    "evaluate_phase5_mps_backward_readiness": (
        "flagquantum.runtime.audit.engine",
        "evaluate_mps_backward_readiness",
    ),
}

# Lookup order mirrors the historical api.py imports where names overlap.
_SYMBOL_MODULES = (
    "flagquantum.runtime.audit.engine",
    "flagquantum.runtime.backend_registry",
    "flagquantum.runtime.distributed.engine",
    "flagquantum.runtime.distributed.backend_policy",
    "flagquantum.runtime.execution",
    "flagquantum.runtime.backends.jax.kernel",
    "flagquantum.runtime.backends.jax.compatibility_surface",
    "flagquantum.runtime.local_preflight",
    "flagquantum.runtime.module",
    "flagquantum.runtime.backends.mps.forward",
    "flagquantum.runtime.backends.mps.production",
    "flagquantum.runtime.backends.mps.records",
    "flagquantum.runtime.backends.mps.training",
    "flagquantum.runtime.operator_backends",
    "flagquantum.runtime.parallel",
    "flagquantum.runtime.parity",
    "flagquantum.runtime.configuration",
    "flagquantum.runtime.backends.statevector.models",
    "flagquantum.runtime.backends.statevector.planning",
    "flagquantum.runtime.backends.statevector.local_execution",
    "flagquantum.runtime.backends.statevector.training",
    "flagquantum.runtime.mps_training",
    "flagquantum.runtime.training",
    "flagquantum.runtime.training_state",
)


def __getattr__(name: str) -> Any:
    alias = _MODULE_ALIASES.get(name)
    if alias is not None:
        return import_module(alias)
    symbol_alias = _SYMBOL_ALIASES.get(name)
    if symbol_alias is not None:
        module_name, symbol = symbol_alias
        return getattr(import_module(module_name), symbol)
    for module_name in _SYMBOL_MODULES:
        module = import_module(module_name)
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_MODULE_ALIASES))
