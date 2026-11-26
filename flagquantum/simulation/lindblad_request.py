"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.dynamics.problem import (
    CollapseOperator as CollapseOperator,
)
from flagquantum.simulation.dynamics.problem import (
    EvolutionValidationError as EvolutionValidationError,
)
from flagquantum.simulation.dynamics.problem import (
    _as_complex_matrix as _as_complex_matrix,
)
from flagquantum.simulation.dynamics.problem import (
    _dense_operator_terms as _dense_operator_terms,
)
from flagquantum.simulation.dynamics.problem import (
    _descriptor_matrix as _descriptor_matrix,
)
from flagquantum.simulation.dynamics.problem import (
    _descriptor_pauli as _descriptor_pauli,
)
from flagquantum.simulation.dynamics.problem import (
    _error as _error,
)
from flagquantum.simulation.dynamics.problem import (
    _hamiltonian_terms as _hamiltonian_terms,
)
from flagquantum.simulation.dynamics.problem import (
    _named_operator as _named_operator,
)
from flagquantum.simulation.dynamics.problem import (
    _normalize_collapse_operators as _normalize_collapse_operators,
)
from flagquantum.simulation.dynamics.problem import (
    _normalize_collapse_terms as _normalize_collapse_terms,
)
from flagquantum.simulation.dynamics.problem import (
    _normalize_hamiltonian as _normalize_hamiltonian,
)
from flagquantum.simulation.dynamics.problem import (
    _normalize_hamiltonians as _normalize_hamiltonians,
)
from flagquantum.simulation.dynamics.problem import (
    _normalize_initial_state as _normalize_initial_state,
)
from flagquantum.simulation.dynamics.problem import (
    _normalize_observables as _normalize_observables,
)
from flagquantum.simulation.dynamics.problem import (
    _normalize_parameter_mapping as _normalize_parameter_mapping,
)
from flagquantum.simulation.dynamics.problem import (
    _normalize_times as _normalize_times,
)
from flagquantum.simulation.dynamics.problem import (
    _observable_matrix as _observable_matrix,
)
from flagquantum.simulation.dynamics.problem import (
    _observable_terms as _observable_terms,
)
from flagquantum.simulation.dynamics.problem import (
    _pauli_operator as _pauli_operator,
)
from flagquantum.simulation.dynamics.problem import (
    _pauli_term as _pauli_term,
)
from flagquantum.simulation.dynamics.problem import (
    _pauli_term_wires as _pauli_term_wires,
)
from flagquantum.simulation.dynamics.problem import (
    _require_density_matrices as _require_density_matrices,
)
from flagquantum.simulation.dynamics.problem import (
    _require_gradients_flag as _require_gradients_flag,
)
from flagquantum.simulation.dynamics.problem import (
    _require_hermitian_terms as _require_hermitian_terms,
)
from flagquantum.simulation.dynamics.problem import (
    _require_qubit_space as _require_qubit_space,
)
from flagquantum.simulation.dynamics.problem import (
    _resolve_batch_size as _resolve_batch_size,
)
from flagquantum.simulation.dynamics.problem import (
    _resolve_state_space as _resolve_state_space,
)
from flagquantum.simulation.dynamics.problem import (
    _solve_max_iterations as _solve_max_iterations,
)
from flagquantum.simulation.dynamics.problem import (
    _solve_tolerance as _solve_tolerance,
)
from flagquantum.simulation.dynamics.problem import (
    _statevectors_to_density as _statevectors_to_density,
)

_implementation = _import_module("flagquantum.simulation.dynamics.problem")
_sys.modules[__name__] = _implementation
