# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 dev_tools/qualtran_dev_tools/notebook_specs.py.

"""Explicit native library notebook catalog in source registration order."""

from pathlib import Path

from tools.development.repository import get_git_root

# Authored notebook assets own documentation and domain-specific example code.
# Keep registration explicit so new files cannot silently enter publication.
NOTEBOOKS_BY_SECTION: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Basic Gates",
        (
            "register_t_gate.ipynb",
            "register_toffoli.ipynb",
            "register_hadamard.ipynb",
            "register_cnot.ipynb",
            "register_diagonal_gates.ipynb",
            "register_y_gate.ipynb",
            "register_and.ipynb",
            "register_states_effects.ipynb",
            "register_measurement.ipynb",
            "register_swap.ipynb",
            "swap_network.ipynb",
            "register_global_phase.ipynb",
            "register_identity.ipynb",
            "control_predicate.ipynb",
            "controlled_via_and.ipynb",
        ),
    ),
    (
        "Chemistry",
        (
            "register_sparse_chemistry.ipynb",
            "register_single_factorization.ipynb",
            "register_double_factorization.ipynb",
            "register_thc_chemistry.ipynb",
            "register_first_quantization.ipynb",
            "register_first_quantization_projectile.ipynb",
            "register_grid_trotter.ipynb",
            "register_trotter_product.ipynb",
            "register_ising_trotter.ipynb",
            "register_hubbard_trotter.ipynb",
            "register_hubbard_qubitization.ipynb",
            "register_hubbard_select.ipynb",
            "register_gradient_givens.ipynb",
        ),
    ),
    (
        "Arithmetic",
        (
            "register_addition.ipynb",
            "register_controlled_addition.ipynb",
            "register_negation.ipynb",
            "register_subtraction.ipynb",
            "register_conditional_arithmetic.ipynb",
            "register_multiplication.ipynb",
            "register_comparison.ipynb",
            "register_conversions.ipynb",
            "register_sorting.ipynb",
            "triangular_register_index.ipynb",
            "register_permutation.ipynb",
            "register_bitwise.ipynb",
            "register_trigonometric.ipynb",
            "register_lists.ipynb",
        ),
    ),
    (
        "Modular Arithmetic",
        (
            "modular_addition.ipynb",
            "modular_subtraction.ipynb",
            "modular_multiplication.ipynb",
            "modular_inverse.ipynb",
            "register_rsa_phase_estimation.ipynb",
            "register_elliptic_curve_addition.ipynb",
            "register_elliptic_curve_cryptography.ipynb",
        ),
    ),
    (
        "GF Arithmetic",
        (
            "binary_field_multiplication.ipynb",
            "binary_field_addition.ipynb",
            "binary_field_add_constant.ipynb",
            "binary_field_square.ipynb",
            "binary_field_inverse.ipynb",
        ),
    ),
    (
        "Polynomials over Galois Fields",
        (
            "field_polynomial_views.ipynb",
            "binary_field_polynomial_add_constant.ipynb",
            "binary_field_polynomial_addition.ipynb",
        ),
    ),
    (
        "Rotations",
        (
            "register_rotation.ipynb",
            "register_su2_rotation.ipynb",
            "quantum_variable_rotation.ipynb",
            "phasing_via_cost_function.ipynb",
            "phase_gradient.ipynb",
            "hamming_weight_phasing.ipynb",
            "zpow_via_phase_gradient.ipynb",
            "rz_via_phase_gradient.ipynb",
            "programmable_rotation_gate_array.ipynb",
            "two_bit_fermionic_fourier.ipynb",
            "approximate_qft.ipynb",
            "textbook_qft.ipynb",
            "phase_gradient_qft.ipynb",
            "luis_perina_phase_window.ipynb",
            "textbook_phase_estimation.ipynb",
            "kaiser_phase_window.ipynb",
            "qubitization_walk.ipynb",
            "qubitization_phase_estimation.ipynb",
        ),
    ),
    (
        "Block Encoding",
        (
            "block_encoding_interface.ipynb",
            "register_unitary_block_encoding.ipynb",
            "register_tensor_product_block_encoding.ipynb",
            "register_product_block_encoding.ipynb",
            "register_phase_block_encoding.ipynb",
            "register_linear_combination.ipynb",
            "register_sparse_matrix_block_encoding.ipynb",
            "register_hermitian_sparse_matrix.ipynb",
            "register_chebyshev.ipynb",
            "register_lcu_block_encoding.ipynb",
        ),
    ),
    (
        "Optimization",
        (
            "kxor_instance_loading.ipynb",
            "kikuchi_guiding_state.ipynb",
            "kikuchi_adjacency_list.ipynb",
            "kikuchi_adjacency_matrix.ipynb",
            "kikuchi_block_encoding.ipynb",
        ),
    ),
    (
        "Bookkeeping",
        (
            "register_split_join.ipynb",
            "register_split2_join2.ipynb",
            "register_allocate_free.ipynb",
            "register_partition.ipynb",
            "register_cast.ipynb",
            "register_always.ipynb",
        ),
    ),
    (
        "Other",
        (
            "prepare_uniform_superposition.ipynb",
            "apply_gate_to_lth_target.ipynb",
            "apply_lth_bloq.ipynb",
            "register_qrom.ipynb",
            "register_select_swap_qrom.ipynb",
            "register_qroam_clean.ipynb",
            "prepared_state_reflections.ipynb",
            "multi_paulis.ipynb",
            "select_pauli_lcu.ipynb",
            "state_preparation_alias_sampling.ipynb",
            "state_preparation_via_rotation.ipynb",
            "generalized_signal_processing.ipynb",
            "hamiltonian_simulation_by_gqsp.ipynb",
            "guided_hamiltonian.ipynb",
        ),
    ),
)


def notebook_sections(
    notebook_root: Path | None = None,
) -> tuple[tuple[str, tuple[Path, ...]], ...]:
    """Resolve the explicit library catalog against the native notebook root.

    The default root belongs to the current checkout. An explicit root supports
    documentation builds and fixtures. Returned paths retain registration order,
    even when files are missing, so the consuming workflow can report each gap.
    """
    root = (
        get_git_root() / "examples/notebooks"
        if notebook_root is None
        else notebook_root
    )
    return tuple(
        (section, tuple(root / name for name in names))
        for section, names in NOTEBOOKS_BY_SECTION
    )
