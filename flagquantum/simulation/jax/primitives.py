"""Historical numerical import facade."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.numerics.jax import (
    _ACTIVE_JAX_BATCH_INDEX as _ACTIVE_JAX_BATCH_INDEX,
)
from flagquantum.simulation.numerics.jax import (
    _ACTIVE_JAX_COMPUTE_DTYPE as _ACTIVE_JAX_COMPUTE_DTYPE,
)
from flagquantum.simulation.numerics.jax import (
    TorchCircuitBuilder as TorchCircuitBuilder,
)
from flagquantum.simulation.numerics.jax import _jax_apply_matrix as _jax_apply_matrix
from flagquantum.simulation.numerics.jax import _jax_complex_dtype as _jax_complex_dtype
from flagquantum.simulation.numerics.jax import _jax_controlled as _jax_controlled
from flagquantum.simulation.numerics.jax import _jax_cphase as _jax_cphase
from flagquantum.simulation.numerics.jax import _jax_cx as _jax_cx
from flagquantum.simulation.numerics.jax import _jax_cy as _jax_cy
from flagquantum.simulation.numerics.jax import _jax_cz as _jax_cz
from flagquantum.simulation.numerics.jax import _jax_diag2 as _jax_diag2
from flagquantum.simulation.numerics.jax import _jax_diag4 as _jax_diag4
from flagquantum.simulation.numerics.jax import _jax_h as _jax_h
from flagquantum.simulation.numerics.jax import (
    _jax_hamiltonian_expectation as _jax_hamiltonian_expectation,
)
from flagquantum.simulation.numerics.jax import (
    _jax_instruction_matrix as _jax_instruction_matrix,
)
from flagquantum.simulation.numerics.jax import _jax_pauli_matrix as _jax_pauli_matrix
from flagquantum.simulation.numerics.jax import (
    _jax_pauli_string_expectation as _jax_pauli_string_expectation,
)
from flagquantum.simulation.numerics.jax import _jax_phase as _jax_phase
from flagquantum.simulation.numerics.jax import _jax_real_dtype as _jax_real_dtype
from flagquantum.simulation.numerics.jax import _jax_rx as _jax_rx
from flagquantum.simulation.numerics.jax import _jax_rxx as _jax_rxx
from flagquantum.simulation.numerics.jax import _jax_ry as _jax_ry
from flagquantum.simulation.numerics.jax import _jax_ryy as _jax_ryy
from flagquantum.simulation.numerics.jax import _jax_rz as _jax_rz
from flagquantum.simulation.numerics.jax import _jax_rzz as _jax_rzz
from flagquantum.simulation.numerics.jax import _jax_s as _jax_s
from flagquantum.simulation.numerics.jax import _jax_scalar_param as _jax_scalar_param
from flagquantum.simulation.numerics.jax import _jax_sdg as _jax_sdg
from flagquantum.simulation.numerics.jax import _jax_swap as _jax_swap
from flagquantum.simulation.numerics.jax import _jax_sx as _jax_sx
from flagquantum.simulation.numerics.jax import _jax_sxdg as _jax_sxdg
from flagquantum.simulation.numerics.jax import _jax_t as _jax_t
from flagquantum.simulation.numerics.jax import _jax_tdg as _jax_tdg
from flagquantum.simulation.numerics.jax import _jax_u2 as _jax_u2
from flagquantum.simulation.numerics.jax import _jax_u3 as _jax_u3
from flagquantum.simulation.numerics.jax import _jax_x as _jax_x
from flagquantum.simulation.numerics.jax import _jax_y as _jax_y
from flagquantum.simulation.numerics.jax import _jax_z as _jax_z
from flagquantum.simulation.numerics.jax import _jax_z_sum as _jax_z_sum
from flagquantum.simulation.numerics.jax import _jax_z_values as _jax_z_values
from flagquantum.simulation.numerics.jax import (
    _set_active_jax_compute_dtype as _set_active_jax_compute_dtype,
)
from flagquantum.simulation.numerics.jax import (
    jax_circuit_tensor_nodes as jax_circuit_tensor_nodes,
)
from flagquantum.simulation.numerics.jax import (
    jax_contract_tensor_nodes as jax_contract_tensor_nodes,
)
from flagquantum.simulation.numerics.jax import (
    jax_evaluate_declared_batch as jax_evaluate_declared_batch,
)

_sys.modules[__name__] = _import_module("flagquantum.simulation.numerics.jax")
