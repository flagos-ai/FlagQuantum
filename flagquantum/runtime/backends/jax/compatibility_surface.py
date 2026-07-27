# ruff: noqa: F822
"""Compatibility surface for the decomposed optional JAX distributed runtime.

Implementation ownership lives in semantically named ``jax_runtime`` modules. This shim
retains public and private symbol identity through the documented compatibility
window without importing JAX/XLA at module import time.
"""

from __future__ import annotations

import sys
from types import ModuleType

from ._loader import load as _load_implementations
from .common import communication_tier as _communication_tier
from .common import env_int as _env_int
from .common import node_count as _node_count
from .release_policy import (
    attach_evidence_contract as _attach_distributed_evidence_contract,
)

_SYMBOLS, _LOADED_MODULES = _load_implementations()
globals().update(_SYMBOLS)
_COMPATIBILITY_ALIASES = {
    "_communication_tier": _communication_tier,
    "_env_int": _env_int,
    "_node_count": _node_count,
    "_attach_distributed_evidence_contract": _attach_distributed_evidence_contract,
}
globals().update(_COMPATIBILITY_ALIASES)
_SYMBOLS.update(_COMPATIBILITY_ALIASES)


class _CompatibilityModule(ModuleType):
    """Keep legacy monkeypatches visible to extracted function namespaces."""

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        if name in _SYMBOLS:
            _SYMBOLS[name] = value
            for module in _LOADED_MODULES:
                vars(module)[name] = value


sys.modules[__name__].__class__ = _CompatibilityModule

__all__ = [
    "JAXDistributedQuantumPlan",
    "JAXMPSRankShardState",
    "JAXShardedStatevectorResult",
    "JAXShardedStatevectorParameterGradientResult",
    "JAXShardedStatevectorTrainingPlan",
    "JAXShardedMPSResult",
    "JAXShardedMPSParameterGradientResult",
    "JAXShardedMPSTrainingPlan",
    "JAXShardedMPSParameterFlowPlan",
    "JAXShardedMPSParameterGateAssignment",
    "JAXShardedTensorNetworkResult",
    "JAXStatevectorShardState",
    "JAXSlicedTensorNetworkGradientResult",
    "JAXSlicedTensorNetworkParameterGradientResult",
    "JAXTNSliceRankState",
    "JAXTensorNetworkNode",
    "initialize_jax_distributed",
    "jax_sharded_mps_parameter_value_and_grad",
    "jax_sharded_statevector_parameter_value_and_grad",
    "jax_sliced_tensor_network_parameter_value_and_grad",
    "jax_sliced_tensor_network_value_and_grad",
    "plan_jax_sharded_mps_parameter_flow",
    "plan_jax_sharded_mps_training",
    "plan_jax_sharded_statevector_training",
    "plan_jax_distributed_quantum_backend",
    "run_jax_sharded_mps",
    "run_jax_sharded_statevector",
    "run_jax_sharded_tensor_network",
]
