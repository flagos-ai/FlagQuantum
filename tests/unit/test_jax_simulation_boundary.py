"""JAX numerical ownership checks that do not require the optional JAX package."""

import pytest

from flagquantum.runtime.backends.jax import kernel, mps_kernel
from flagquantum.simulation import jax_gate_primitives, jax_mps

pytestmark = pytest.mark.unit


def test_runtime_reuses_simulation_owned_jax_gate_primitives():
    assert kernel._jax_apply_matrix is jax_gate_primitives._jax_apply_matrix
    assert kernel._jax_complex_dtype is jax_gate_primitives._jax_complex_dtype
    assert kernel._jax_instruction_matrix is jax_gate_primitives._jax_instruction_matrix
    assert (
        kernel._jax_statevector_from_circuit
        is jax_gate_primitives._jax_statevector_from_circuit
    )


def test_runtime_reuses_simulation_owned_jax_mps_operations():
    assert mps_kernel._jax_mps_apply_one is jax_mps.jax_mps_apply_one
    assert mps_kernel._jax_mps_split_pair is jax_mps.jax_mps_split_pair
    assert mps_kernel._jax_mps_to_statevector is jax_mps.jax_mps_to_statevector
    assert mps_kernel._jax_mps_transfer_identity is jax_mps.jax_mps_transfer_identity
    assert kernel._jax_mps_z_values is jax_mps.jax_mps_z_values
    assert kernel._jax_mps_z_sum is jax_mps.jax_mps_z_sum
    assert (
        mps_kernel._jax_mps_pauli_string_expectation
        is jax_mps.jax_mps_pauli_string_expectation
    )
