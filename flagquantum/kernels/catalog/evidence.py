"""Test and benchmark evidence associated with kernel implementations."""

from __future__ import annotations

from .schema import KernelEvidence


def _evidence(
    implementation_id: str,
    *correctness_tests: str,
    gradient_tests: tuple[str, ...] = (),
    capability_tests: tuple[str, ...] = (),
    benchmark_artifacts: tuple[str, ...] = (),
) -> KernelEvidence:
    return KernelEvidence(
        evidence_id=implementation_id.replace("FQKI-", "FQKE-", 1),
        implementation_id=implementation_id,
        correctness_tests=correctness_tests,
        gradient_tests=gradient_tests,
        capability_tests=capability_tests,
        benchmark_artifacts=benchmark_artifacts,
    )


EVIDENCE: tuple[KernelEvidence, ...] = (
    _evidence(
        "FQKI-TRITON-SV-001-A",
        "tests/unit/test_statevector_triton.py::test_generic_local_1q_matches_pytorch",
    ),
    _evidence(
        "FQKI-TRITON-SV-001-B",
        "tests/test_statevector_triton_gates.py::test_generic_constant_single_qubit_regions_preserve_input_gradient",
        gradient_tests=(
            "tests/unit/test_statevector_gate_autograd_boundary.py::test_matrix_wrapper_preserves_state_and_matrix_gradients",
        ),
    ),
    _evidence(
        "FQKI-TRITON-SV-002-A",
        "tests/test_statevector_triton_gates.py::test_local_cx_inplace_matches_index_reference",
    ),
    _evidence(
        "FQKI-TRITON-SV-003-A",
        "tests/test_statevector_triton_gates.py::test_local_cx_segment_matches_reverse_source_permutation",
    ),
    _evidence(
        "FQKI-TRITON-SV-003-B",
        "tests/test_statevector_triton_gates.py::test_constant_ry_rz_triton_path_with_cx_matches_cpu",
        gradient_tests=(
            "tests/unit/test_statevector_gate_autograd_boundary.py::test_cx_wrapper_reverses_noncommuting_gate_sequence",
        ),
    ),
    _evidence(
        "FQKI-TRITON-SV-004-A",
        "tests/test_statevector_triton_gates.py::test_constant_ry_rz_triton_path_with_cx_matches_cpu",
    ),
    _evidence(
        "FQKI-TRITON-SV-005-A",
        "tests/unit/test_single_qubit_loop_triton.py::test_repeated_rx_rz_cuda_matches_reference",
        gradient_tests=(
            "tests/unit/test_single_qubit_loop_triton.py::test_repeated_rx_rz_cuda_matches_reference",
        ),
        capability_tests=(
            "tests/unit/test_single_qubit_loop_triton.py::test_repeated_rx_rz_cpu_fallback_matches_reference",
        ),
    ),
    _evidence(
        "FQKI-TRITON-SV-006-A",
        "tests/unit/test_statevector_triton.py::test_fused_transpose_1q_matches_unpack_then_gate",
    ),
    _evidence(
        "FQKI-TRITON-SV-007-A",
        "tests/test_statevector_triton_gates.py::test_control_one_pack_unpack_matches_index_reference",
    ),
    _evidence(
        "FQKI-TRITON-SV-008-A",
        "tests/test_statevector_triton_gates.py::test_control_one_pack_unpack_matches_index_reference",
    ),
    _evidence(
        "FQKI-TRITON-GR-001-A",
        "tests/unit/test_statevector_triton.py::test_fused_vjp_and_adjoint_match_pytorch",
        gradient_tests=(
            "tests/unit/test_statevector_triton.py::test_fused_vjp_and_adjoint_match_pytorch",
        ),
    ),
    _evidence(
        "FQKI-TRITON-GR-002-A",
        "tests/unit/test_statevector_triton.py::test_fused_reversible_vjp_matches_pytorch",
        gradient_tests=(
            "tests/unit/test_statevector_triton.py::test_fused_reversible_vjp_matches_pytorch",
        ),
    ),
    _evidence(
        "FQKI-TRITON-GR-003-A",
        "tests/unit/test_statevector_triton.py::test_fused_sharded_vjp_and_adjoint_matches_global_pair",
        gradient_tests=(
            "tests/unit/test_statevector_triton.py::test_fused_sharded_vjp_and_adjoint_matches_global_pair",
        ),
    ),
    _evidence(
        "FQKI-TRITON-GR-004-A",
        "tests/unit/test_single_qubit_loop_triton.py::test_repeated_rx_rz_cuda_tangents_match_cpu_reference",
        gradient_tests=(
            "tests/unit/test_single_qubit_loop_triton.py::test_repeated_rx_rz_cuda_tangents_match_cpu_reference",
        ),
        capability_tests=(
            "tests/unit/test_single_qubit_loop_triton.py::test_repeated_rx_rz_cpu_tangents_match_jacobian",
        ),
    ),
    _evidence(
        "FQKI-TRITON-GR-005-A",
        "tests/unit/test_two_qubit_pauli_tangent_triton.py::test_two_qubit_pauli_tangent_cuda_matches_reference",
        gradient_tests=(
            "tests/unit/test_two_qubit_pauli_tangent_triton.py::test_two_qubit_pauli_tangent_cuda_matches_reference",
        ),
        capability_tests=(
            "tests/unit/test_two_qubit_pauli_tangent_triton.py::test_two_qubit_pauli_tangent_cpu_fallback_matches_reference",
        ),
    ),
    _evidence(
        "FQKI-TRITON-GR-006-A",
        "tests/unit/test_hva_forward_tangent_triton.py::test_hva_forward_tangents_match_statevector_jacobian",
        gradient_tests=(
            "tests/unit/test_hva_forward_tangent_triton.py::test_hva_forward_tangents_match_statevector_jacobian",
        ),
    ),
    _evidence(
        "FQKI-TRITON-MPS-001-A",
        "tests/unit/test_mps_two_site_triton.py::test_fused_mps_two_site_forward_and_backward",
        gradient_tests=(
            "tests/unit/test_mps_two_site_triton.py::test_fused_mps_two_site_forward_and_backward",
        ),
        capability_tests=(
            "tests/unit/test_mps_two_site_triton.py::test_fused_mps_two_site_cpu_fallback_matches_reference",
        ),
    ),
    _evidence(
        "FQKI-TRITON-MPS-002-A",
        "tests/unit/test_mps_two_site_triton.py::test_fused_mps_range_projection_avoids_full_matrix_with_correct_values",
        capability_tests=(
            "tests/unit/test_mps_two_site_triton.py::test_fused_mps_range_projection_cpu_fallback_matches_reference",
        ),
    ),
    _evidence(
        "FQKI-TRITON-MPS-003-A",
        "tests/unit/test_mps_one_site_triton.py::test_fused_mps_one_site_forward_and_backward",
        "tests/unit/test_mps_one_site_catalog_dispatch.py::test_mps_one_site_bucket_uses_catalog_and_preserves_gradients",
        gradient_tests=(
            "tests/unit/test_mps_one_site_triton.py::test_fused_mps_one_site_forward_and_backward",
            "tests/unit/test_mps_one_site_catalog_dispatch.py::test_mps_one_site_bucket_uses_catalog_and_preserves_gradients",
        ),
        capability_tests=(
            "tests/unit/test_mps_one_site_triton.py::test_fused_mps_one_site_cpu_fallback_matches_reference",
        ),
    ),
    _evidence(
        "FQKI-TRITON-MPS-004-A",
        "tests/unit/test_mps_environment_triton.py::test_fused_mps_environment_cuda_matches_reference",
        "tests/unit/test_mps_environment_catalog_dispatch.py::test_mps_environment_product_path_uses_catalog",
        capability_tests=(
            "tests/unit/test_mps_environment_triton.py::test_fused_mps_environment_cpu_fallback_matches_reference_and_gradients",
            "tests/unit/test_mps_environment_triton.py::test_fused_mps_environment_large_shape_uses_fallback",
            "tests/unit/test_mps_environment_catalog_dispatch.py::test_mps_environment_route_enforces_evidenced_window",
            "tests/unit/test_mps_environment_catalog_dispatch.py::test_mps_environment_route_rejects_gradients",
        ),
    ),
    _evidence(
        "FQKI-TRITON-MPS-005-A",
        "tests/unit/test_mps_environment_channels_triton.py::test_fused_mps_environment_channels_cuda_matches_reference",
        "tests/unit/test_mps_environment_channels_catalog_dispatch.py::test_mps_environment_channels_product_path_uses_catalog",
        capability_tests=(
            "tests/unit/test_mps_environment_channels_triton.py::test_fused_mps_environment_channels_cpu_fallback_preserves_gradients",
            "tests/unit/test_mps_environment_channels_triton.py::test_fused_mps_environment_channels_outside_window_uses_fallback",
            "tests/unit/test_mps_environment_channels_catalog_dispatch.py::test_mps_environment_channels_route_enforces_evidenced_window",
            "tests/unit/test_mps_environment_channels_catalog_dispatch.py::test_mps_environment_channels_route_rejects_gradients",
        ),
    ),
    _evidence(
        "FQKI-TRITON-NUM-001-A",
        "tests/unit/test_triton_complex_bmm.py::test_fused_complex_bmm_forward_and_backward_match_torch",
        gradient_tests=(
            "tests/unit/test_triton_complex_bmm.py::test_fused_complex_bmm_forward_and_backward_match_torch",
        ),
        capability_tests=(
            "tests/unit/test_triton_complex_bmm.py::test_fused_complex_bmm_cpu_fallback_matches_torch",
        ),
    ),
    _evidence(
        "FQKI-TRITON-NUM-002-A",
        "tests/unit/test_real_imag_kernels.py::test_fused_layout_route_does_not_materialize_canonical_inputs",
        gradient_tests=(
            "tests/unit/test_real_imag_kernels.py::test_fused_layout_route_does_not_materialize_canonical_inputs",
        ),
        capability_tests=(
            "tests/unit/test_triton_complex_bmm.py::test_fused_complex_layout_bmm_cpu_fallback_matches_torch",
        ),
    ),
)


__all__ = ["EVIDENCE"]
