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
    assert (
        mps_kernel._jax_mps_single_pauli_with_envs
        is jax_mps.jax_mps_single_pauli_with_envs
    )
    assert (
        mps_kernel._jax_mps_adjacent_zz_with_envs
        is jax_mps.jax_mps_adjacent_zz_with_envs
    )


def test_zz_z_chain_parser_accepts_and_accumulates_supported_terms():
    terms = (
        (1.5, ()),
        (2.0, ((0, "X"),)),
        (-0.5, ((1, "z"),)),
        (0.25, ((1, "Z"), (0, "Z"))),
        (0.75, ((0, "z"), (1, "z"))),
    )

    parsed = jax_mps.parse_zz_z_chain_hamiltonian(terms, 3)

    assert parsed == (
        {"x": (2.0, 0.0, 0.0), "y": (0.0, 0.0, 0.0), "z": (0.0, -0.5, 0.0)},
        (1.0, 0.0),
        1.5,
    )
    assert jax_mps.is_zz_z_chain_hamiltonian(terms, 3) is True


@pytest.mark.parametrize(
    "terms",
    (
        ((1.0, ((0, "z"), (2, "z"))),),
        ((1.0, ((0, "x"), (1, "x"))),),
        ((1.0, ((3, "z"),)),),
    ),
)
def test_zz_z_chain_parser_rejects_unsupported_terms(terms):
    assert jax_mps.parse_zz_z_chain_hamiltonian(terms, 3) is None
