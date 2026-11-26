"""Stable public namespace over the responsible implementation owners."""

from collections.abc import Callable as Callable
from collections.abc import Iterable as Iterable
from collections.abc import Mapping as Mapping
from collections.abc import Sequence as Sequence
from dataclasses import dataclass as dataclass
from functools import partial as partial
from importlib import import_module as _import_module
from time import perf_counter as perf_counter
from typing import TYPE_CHECKING, Any
from typing import Literal as Literal
from typing import Protocol as Protocol

import torch as torch

from flagquantum.algorithms.many_body.hva import heisenberg_hva as heisenberg_hva
from flagquantum.algorithms.many_body.hva import (
    heisenberg_hva_parameter_count as heisenberg_hva_parameter_count,
)
from flagquantum.algorithms.many_body.spin import (
    heisenberg_chain_hamiltonian as heisenberg_chain_hamiltonian,
)
from flagquantum.algorithms.many_body.spin import (
    transverse_field_ising as transverse_field_ising,
)
from flagquantum.algorithms.many_body.spin import (
    zz_chain_hamiltonian as zz_chain_hamiltonian,
)
from flagquantum.algorithms.optimization.qaoa_circuits import (
    qaoa_circuit as qaoa_circuit,
)
from flagquantum.algorithms.optimization.qaoa_circuits import qaoa_loss as qaoa_loss
from flagquantum.algorithms.optimization.vqe import (
    AdaptVQEIteration as AdaptVQEIteration,
)
from flagquantum.algorithms.optimization.vqe import AdaptVQEResult as AdaptVQEResult
from flagquantum.algorithms.optimization.vqe import (
    LayerwiseVQEResult as LayerwiseVQEResult,
)
from flagquantum.algorithms.optimization.vqe import OptimizerFactory as OptimizerFactory
from flagquantum.algorithms.optimization.vqe import VQEResult as VQEResult
from flagquantum.algorithms.optimization.vqe import run_adapt_vqe as run_adapt_vqe
from flagquantum.algorithms.optimization.vqe import run_hybrid_vqe as run_hybrid_vqe
from flagquantum.algorithms.optimization.vqe import (
    run_layerwise_vqe as run_layerwise_vqe,
)
from flagquantum.algorithms.optimization.vqe import run_vqe as run_vqe
from flagquantum.algorithms.optimization.vqe import vqe_loss as vqe_loss
from flagquantum.algorithms.primitives.ansatz import (
    hardware_efficient_ansatz as hardware_efficient_ansatz,
)
from flagquantum.algorithms.primitives.ansatz import (
    hardware_efficient_parameter_count as hardware_efficient_parameter_count,
)
from flagquantum.algorithms.primitives.hamiltonian import Hamiltonian as Hamiltonian
from flagquantum.algorithms.primitives.hamiltonian import (
    HamiltonianTerm as HamiltonianTerm,
)
from flagquantum.algorithms.primitives.hamiltonian import pauli_term as pauli_term

from ..circuit import Circuit as Circuit
from ..core.runtime_config import get_runtime_config as get_runtime_config
from ..observables import Observable as Observable
from ..simulation.mps.state import MPSState as MPSState
from ..simulation.pauli import (
    infer_n_qubits_from_dense_state as infer_n_qubits_from_dense_state,
)
from ..simulation.pauli import (
    pauli_product_density_expectation as pauli_product_density_expectation,
)
from ..simulation.pauli import pauli_product_operator as pauli_product_operator
from ..simulation.pauli import (
    pauli_product_statevector_expectation as pauli_product_statevector_expectation,
)
from ..simulation.tensor_network.state import TensorNetworkState as TensorNetworkState
from .optimization import HybridOptimizationResult as HybridOptimizationResult
from .optimization import OptimizationStage as OptimizationStage
from .optimization import optimize_hybrid as optimize_hybrid

if TYPE_CHECKING:
    from flagquantum.algorithms.many_body.hva import (
        heisenberg_hva as heisenberg_hva,
    )
    from flagquantum.algorithms.many_body.hva import (
        heisenberg_hva_parameter_count as heisenberg_hva_parameter_count,
    )
    from flagquantum.algorithms.many_body.spin import (
        heisenberg_chain_hamiltonian as heisenberg_chain_hamiltonian,
    )
    from flagquantum.algorithms.many_body.spin import (
        transverse_field_ising as transverse_field_ising,
    )
    from flagquantum.algorithms.many_body.spin import (
        zz_chain_hamiltonian as zz_chain_hamiltonian,
    )
    from flagquantum.algorithms.optimization.qaoa_circuits import (
        qaoa_circuit as qaoa_circuit,
    )
    from flagquantum.algorithms.optimization.qaoa_circuits import (
        qaoa_loss as qaoa_loss,
    )
    from flagquantum.algorithms.optimization.vqe import (
        AdaptVQEIteration as AdaptVQEIteration,
    )
    from flagquantum.algorithms.optimization.vqe import (
        AdaptVQEResult as AdaptVQEResult,
    )
    from flagquantum.algorithms.optimization.vqe import (
        LayerwiseVQEResult as LayerwiseVQEResult,
    )
    from flagquantum.algorithms.optimization.vqe import (
        OptimizerFactory as OptimizerFactory,
    )
    from flagquantum.algorithms.optimization.vqe import (
        VQEResult as VQEResult,
    )
    from flagquantum.algorithms.optimization.vqe import (
        run_adapt_vqe as run_adapt_vqe,
    )
    from flagquantum.algorithms.optimization.vqe import (
        run_hybrid_vqe as run_hybrid_vqe,
    )
    from flagquantum.algorithms.optimization.vqe import (
        run_layerwise_vqe as run_layerwise_vqe,
    )
    from flagquantum.algorithms.optimization.vqe import (
        run_vqe as run_vqe,
    )
    from flagquantum.algorithms.optimization.vqe import (
        vqe_loss as vqe_loss,
    )
    from flagquantum.algorithms.primitives.ansatz import (
        hardware_efficient_ansatz as hardware_efficient_ansatz,
    )
    from flagquantum.algorithms.primitives.ansatz import (
        hardware_efficient_parameter_count as hardware_efficient_parameter_count,
    )
    from flagquantum.algorithms.primitives.hamiltonian import Hamiltonian as Hamiltonian
    from flagquantum.algorithms.primitives.hamiltonian import (
        HamiltonianTerm as HamiltonianTerm,
    )
    from flagquantum.algorithms.primitives.hamiltonian import pauli_term as pauli_term

_OWNERS = {
    "Hamiltonian": "flagquantum.algorithms.primitives.hamiltonian",
    "HamiltonianTerm": "flagquantum.algorithms.primitives.hamiltonian",
    "heisenberg_chain_hamiltonian": "flagquantum.algorithms.many_body.spin",
    "pauli_term": "flagquantum.algorithms.primitives.hamiltonian",
    "transverse_field_ising": "flagquantum.algorithms.many_body.spin",
    "zz_chain_hamiltonian": "flagquantum.algorithms.many_body.spin",
    "hardware_efficient_ansatz": "flagquantum.algorithms.primitives.ansatz",
    "hardware_efficient_parameter_count": "flagquantum.algorithms.primitives.ansatz",
    "heisenberg_hva": "flagquantum.algorithms.many_body.hva",
    "heisenberg_hva_parameter_count": "flagquantum.algorithms.many_body.hva",
    "qaoa_circuit": "flagquantum.algorithms.optimization.qaoa_circuits",
    "qaoa_loss": "flagquantum.algorithms.optimization.qaoa_circuits",
    "AdaptVQEIteration": "flagquantum.algorithms.optimization.vqe",
    "AdaptVQEResult": "flagquantum.algorithms.optimization.vqe",
    "LayerwiseVQEResult": "flagquantum.algorithms.optimization.vqe",
    "OptimizerFactory": "flagquantum.algorithms.optimization.vqe",
    "VQEResult": "flagquantum.algorithms.optimization.vqe",
    "run_adapt_vqe": "flagquantum.algorithms.optimization.vqe",
    "run_hybrid_vqe": "flagquantum.algorithms.optimization.vqe",
    "run_layerwise_vqe": "flagquantum.algorithms.optimization.vqe",
    "run_vqe": "flagquantum.algorithms.optimization.vqe",
    "vqe_loss": "flagquantum.algorithms.optimization.vqe",
}

__all__ = [
    "AdaptVQEIteration",
    "AdaptVQEResult",
    "Hamiltonian",
    "HamiltonianTerm",
    "LayerwiseVQEResult",
    "OptimizerFactory",
    "VQEResult",
    "hardware_efficient_ansatz",
    "hardware_efficient_parameter_count",
    "heisenberg_chain_hamiltonian",
    "heisenberg_hva",
    "heisenberg_hva_parameter_count",
    "pauli_term",
    "qaoa_loss",
    "qaoa_circuit",
    "run_vqe",
    "run_adapt_vqe",
    "run_hybrid_vqe",
    "run_layerwise_vqe",
    "transverse_field_ising",
    "vqe_loss",
    "zz_chain_hamiltonian",
]


def __getattr__(name: str) -> Any:
    if name.startswith("_"):
        raise AttributeError(name)
    if name in _OWNERS:
        value = getattr(_import_module(_OWNERS[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Resolve published class annotations in this public module without owner side effects.
