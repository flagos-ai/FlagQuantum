"""Historical numerical import facade."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.mps.jax.kernels import (
    is_zz_z_chain_hamiltonian as is_zz_z_chain_hamiltonian,
)
from flagquantum.simulation.mps.jax.kernels import (
    jax_mps_adjacent_zz_with_envs as jax_mps_adjacent_zz_with_envs,
)
from flagquantum.simulation.mps.jax.kernels import (
    jax_mps_apply_adjacent_chain_scan as jax_mps_apply_adjacent_chain_scan,
)
from flagquantum.simulation.mps.jax.kernels import (
    jax_mps_apply_local_stack as jax_mps_apply_local_stack,
)
from flagquantum.simulation.mps.jax.kernels import (
    jax_mps_apply_one as jax_mps_apply_one,
)
from flagquantum.simulation.mps.jax.kernels import (
    jax_mps_apply_two_adjacent as jax_mps_apply_two_adjacent,
)
from flagquantum.simulation.mps.jax.kernels import (
    jax_mps_apply_two_adjacent_active as jax_mps_apply_two_adjacent_active,
)
from flagquantum.simulation.mps.jax.kernels import (
    jax_mps_apply_two_remote as jax_mps_apply_two_remote,
)
from flagquantum.simulation.mps.jax.kernels import (
    jax_mps_expectation_product_ops as jax_mps_expectation_product_ops,
)
from flagquantum.simulation.mps.jax.kernels import (
    jax_mps_hamiltonian_expectation as jax_mps_hamiltonian_expectation,
)
from flagquantum.simulation.mps.jax.kernels import (
    jax_mps_initial_open_boundary_tensors as jax_mps_initial_open_boundary_tensors,
)
from flagquantum.simulation.mps.jax.kernels import (
    jax_mps_initial_padded_stack as jax_mps_initial_padded_stack,
)
from flagquantum.simulation.mps.jax.kernels import (
    jax_mps_pauli_string_expectation as jax_mps_pauli_string_expectation,
)
from flagquantum.simulation.mps.jax.kernels import (
    jax_mps_project_open_boundaries as jax_mps_project_open_boundaries,
)
from flagquantum.simulation.mps.jax.kernels import (
    jax_mps_single_pauli_with_envs as jax_mps_single_pauli_with_envs,
)
from flagquantum.simulation.mps.jax.kernels import (
    jax_mps_site_observable_transfer as jax_mps_site_observable_transfer,
)
from flagquantum.simulation.mps.jax.kernels import (
    jax_mps_site_transfer as jax_mps_site_transfer,
)
from flagquantum.simulation.mps.jax.kernels import (
    jax_mps_split_pair as jax_mps_split_pair,
)
from flagquantum.simulation.mps.jax.kernels import (
    jax_mps_to_statevector as jax_mps_to_statevector,
)
from flagquantum.simulation.mps.jax.kernels import (
    jax_mps_transfer_identity as jax_mps_transfer_identity,
)
from flagquantum.simulation.mps.jax.kernels import (
    jax_mps_transfer_identity_right as jax_mps_transfer_identity_right,
)
from flagquantum.simulation.mps.jax.kernels import (
    jax_mps_transfer_op as jax_mps_transfer_op,
)
from flagquantum.simulation.mps.jax.kernels import jax_mps_z_sum as jax_mps_z_sum
from flagquantum.simulation.mps.jax.kernels import jax_mps_z_values as jax_mps_z_values
from flagquantum.simulation.mps.jax.kernels import (
    jax_mps_zz_z_chain_expectation_padded_scan as jax_mps_zz_z_chain_expectation_padded_scan,
)
from flagquantum.simulation.mps.jax.kernels import (
    jax_sharded_mps_z_sum as jax_sharded_mps_z_sum,
)
from flagquantum.simulation.mps.jax.kernels import (
    parse_zz_z_chain_hamiltonian as parse_zz_z_chain_hamiltonian,
)

_sys.modules[__name__] = _import_module("flagquantum.simulation.mps.jax.kernels")
