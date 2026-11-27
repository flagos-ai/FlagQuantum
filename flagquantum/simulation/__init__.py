"""Simulation engines, tensor utilities, and numerical kernels."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from flagquantum.simulation import unitary as unitary
    from flagquantum.simulation.dynamics import adjoint

    lindblad_adjoint = adjoint
    from flagquantum.simulation.dynamics import lindblad as lindblad
    from flagquantum.simulation.dynamics.adjoint import (
        adjoint_gradient as adjoint_gradient,
    )
    from flagquantum.simulation.dynamics.lindblad import (
        CollapseOperator as CollapseOperator,
    )
    from flagquantum.simulation.dynamics.lindblad import (
        EvolutionPlan as EvolutionPlan,
    )
    from flagquantum.simulation.dynamics.lindblad import (
        EvolutionResult as EvolutionResult,
    )
    from flagquantum.simulation.dynamics.lindblad import (
        EvolutionValidationError as EvolutionValidationError,
    )
    from flagquantum.simulation.dynamics.lindblad import (
        amplitude_damping as amplitude_damping,
    )
    from flagquantum.simulation.dynamics.lindblad import (
        evolve_density_matrix as evolve_density_matrix,
    )
    from flagquantum.simulation.dynamics.lindblad import (
        plan_density_matrix_evolution as plan_density_matrix_evolution,
    )

    from .statevector import small

    small_statevector = small
    from .statevector.small import small_data_reuploading_z as small_data_reuploading_z
    from .unitary import MAX_UNITARY_QUBITS as MAX_UNITARY_QUBITS
    from .unitary import get_unitary as get_unitary

_EXPORT_MODULES = {
    "small_data_reuploading_z": ".statevector.small",
    "CollapseOperator": ".dynamics.lindblad",
    "EvolutionPlan": ".dynamics.lindblad",
    "EvolutionResult": ".dynamics.lindblad",
    "EvolutionValidationError": ".dynamics.lindblad",
    "amplitude_damping": ".dynamics.lindblad",
    "evolve_density_matrix": ".dynamics.lindblad",
    "plan_density_matrix_evolution": ".dynamics.lindblad",
    "adjoint_gradient": ".dynamics.adjoint",
    "MAX_UNITARY_QUBITS": ".unitary",
    "get_unitary": ".unitary",
}
__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    if name in {"lindblad", "lindblad_adjoint", "unitary", "small_statevector"}:
        module = {
            "small_statevector": ".statevector.small",
            "lindblad": ".dynamics.lindblad",
            "lindblad_adjoint": ".dynamics.adjoint",
            "unitary": ".unitary",
        }[name]
        return import_module(module, __name__)
    if name in _EXPORT_MODULES:
        value = getattr(import_module(_EXPORT_MODULES[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(name)
