"""Compatibility facade for tensor-network simulation."""

from importlib import import_module
from typing import Any

_MODULES = (
    "flagquantum.simulation.tensor_network.contraction",
    "flagquantum.simulation.tensor_network.models",
    "flagquantum.simulation.tensor_network.state",
    "flagquantum.simulation.tensor_network.entrypoints",
    "flagquantum.simulation.tensor_network.observables",
)

__all__ = (
    "CompiledTNContractionBucket",
    "CompiledTNContractionStage",
    "CompiledTNNode",
    "CompiledTNObservableProgram",
    "CompiledTNProgram",
    "CompiledTNStagePlan",
    "ContractionGraph",
    "ContractionPathStep",
    "EinsumProgram",
    "PairContractionStep",
    "TensorNetworkContractionPlan",
    "TensorNetworkContractionProfile",
    "TensorNetworkExpectationPlan",
    "TensorNetworkNode",
    "TensorNetworkSlicingPlan",
    "TensorNetworkState",
    "TensorNode",
    "build_tensor_network",
    "build_tensor_network_expectation",
    "build_tensor_network_hamiltonian_expectation",
    "build_tensor_network_hamiltonian_expectations",
    "run_tensor_network",
    "tensor_network_amplitude",
    "tensor_network_amplitudes",
    "tensor_network_expectation_ps",
    "tensor_network_expectations",
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
