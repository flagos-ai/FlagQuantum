"""JAX numerical ownership checks that do not require the optional JAX package."""

import pytest

from flagquantum.runtime.backends.jax import (
    array_conversions,
    kernel,
    mps_canonicalization,
    mps_execution,
    mps_kernel,
    mps_pullbacks,
    statevector_kernels,
)
from flagquantum.simulation import (
    jax_gate_primitives,
    jax_mps,
    jax_mps_batched,
    jax_mps_pullbacks,
    jax_statevector,
    jax_tensor_network,
)

pytestmark = pytest.mark.unit


def test_runtime_reuses_simulation_owned_jax_gate_primitives():
    assert (
        kernel._jax_statevector_from_circuit
        is jax_gate_primitives._jax_statevector_from_circuit
    )
    assert (
        kernel._set_active_jax_compute_dtype
        is jax_gate_primitives._set_active_jax_compute_dtype
    )
    assert (
        statevector_kernels._jax_sharded_statevector_rank_loss_from_local_amplitudes
        is jax_statevector.jax_sharded_statevector_rank_loss
    )
    assert (
        array_conversions._jax_basis_indices_for_wires
        is jax_statevector.jax_basis_indices_for_wires
    )
    assert (
        statevector_kernels.jax_apply_local_statevector_gate
        is jax_statevector.jax_apply_local_statevector_gate
    )
    assert (
        statevector_kernels.jax_rank_mask_for_touched_delta
        is jax_statevector.jax_rank_mask_for_touched_delta
    )
    assert (
        statevector_kernels.jax_local_positions_for_gate_input
        is jax_statevector.jax_local_positions_for_gate_input
    )
    assert (
        statevector_kernels.jax_gate_basis_in_for_delta_and_local_input
        is jax_statevector.jax_gate_basis_in_for_delta_and_local_input
    )
    assert (
        statevector_kernels.jax_combine_pair_exchanged_statevector
        is jax_statevector.jax_combine_pair_exchanged_statevector
    )


def test_runtime_reuses_simulation_owned_jax_mps_operations():
    assert mps_kernel._jax_mps_apply_one is jax_mps.jax_mps_apply_one
    assert mps_kernel._jax_mps_apply_local_stack is jax_mps.jax_mps_apply_local_stack
    assert (
        mps_kernel._jax_mps_apply_adjacent_chain_scan
        is jax_mps.jax_mps_apply_adjacent_chain_scan
    )
    assert (
        mps_kernel._jax_mps_initial_padded_stack is jax_mps.jax_mps_initial_padded_stack
    )
    assert (
        mps_kernel._jax_mps_project_open_boundaries
        is jax_mps.jax_mps_project_open_boundaries
    )
    assert kernel._jax_mps_z_values is jax_mps.jax_mps_z_values
    assert kernel._jax_mps_z_sum is jax_mps.jax_mps_z_sum
    assert (
        kernel._jax_mps_hamiltonian_expectation
        is jax_mps.jax_mps_hamiltonian_expectation
    )
    assert (
        mps_execution._apply_one_jax_mps_tensor
        is jax_mps_batched.jax_mps_apply_one_batched
    )
    assert (
        mps_execution._apply_two_jax_mps_tensors
        is jax_mps_batched.jax_mps_apply_two_batched
    )
    assert (
        mps_pullbacks.jax_mps_owner_local_vjp
        is jax_mps_pullbacks.jax_mps_owner_local_vjp
    )
    assert (
        mps_pullbacks.jax_mps_boundary_rxx_pullback
        is jax_mps_pullbacks.jax_mps_boundary_rxx_pullback
    )
    assert (
        mps_canonicalization.jax_mps_canonicalization_pullback
        is jax_mps_pullbacks.jax_mps_canonicalization_pullback
    )


def test_runtime_reuses_simulation_owned_jax_tensor_network_observables():
    assert (
        kernel._jax_tensor_network_z_values
        is jax_tensor_network.jax_tensor_network_z_values
    )
    assert (
        kernel._jax_tensor_network_z_sum is jax_tensor_network.jax_tensor_network_z_sum
    )
    assert (
        kernel._jax_tensor_network_hamiltonian_expectation
        is jax_tensor_network.jax_tensor_network_hamiltonian_expectation
    )


def test_statevector_rank_mask_uses_sharded_wire_order():
    sharded_wires = (0, 3, 5)

    assert (
        jax_statevector.jax_rank_mask_for_touched_delta(sharded_wires, (0, 5), 0b01)
        == 0b001
    )
    assert (
        jax_statevector.jax_rank_mask_for_touched_delta(sharded_wires, (0, 5), 0b10)
        == 0b100
    )
    assert (
        jax_statevector.jax_rank_mask_for_touched_delta(sharded_wires, (0, 5), 0b11)
        == 0b101
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
