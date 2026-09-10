from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contracts" / "hybrid-compilation-private-v0-candidate.json"


def _contract() -> dict[str, object]:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_private_hybrid_contract_preserves_current_authorities() -> None:
    contract = _contract()

    assert contract["public_api_change"] is False
    assert contract["default_path_change"] is False
    assert contract["implementation_language"] == "python"
    assert contract["external_compiler"] is None
    assert contract["first_execution_classification"] == "single_device_fast_path"
    assert contract["existing_authorities"] == {
        "public_program": "flagquantum.core.ir.CircuitIR",
        "circuit_compiler": "flagquantum.compiler",
        "execution": "flagquantum.runtime",
        "numerics": "flagquantum.simulation",
        "plugin_registry": "flagquantum.ecosystem.extensions.ExtensionRegistry",
    }


def test_private_hybrid_contract_has_structured_control_and_quantum_effects() -> None:
    contract = _contract()
    operations = set(contract["phase1_operations"])
    invariants = set(contract["phase1_invariants"])

    assert {"program.return", "scf.if", "scf.for", "scf.yield"} <= operations
    assert {"tensor.extract", "arith.cmp"} <= operations
    assert {"quantum.rx", "quantum.ry", "quantum.cx"} <= operations
    assert "quantum.expectation" in operations
    assert "linear_ordered_quantum_effect" in invariants
    assert "matching_branch_signatures" in invariants
    assert "matching_loop_carried_signatures" in invariants


def test_phase1_cannot_claim_execution_or_public_integration() -> None:
    contract = _contract()
    exclusions = set(contract["phase1_exclusions"])
    rules = contract["rules"]

    assert {
        "public_exports",
        "default_compiler_integration",
        "runtime_execution",
        "gradient_execution",
        "measurement_control_flow",
        "catalyst",
        "flagquantum_authored_cpp",
    } <= exclusions
    assert rules["candidate_is_public_contract"] is False
    assert rules["may_update_public_api_snapshots"] is False
    assert rules["may_restore_removed_private_tree_wholesale"] is False
    assert rules["may_copy_historical_layer_hierarchy"] is False
    assert rules["may_duplicate_circuit_or_runtime_authority"] is False
    assert rules["unsupported_behavior_fails_closed"] is True


def test_phase1_migrates_semantics_into_current_vnext_authorities() -> None:
    contract = _contract()
    gates = contract["entry_gates"]

    assert contract["migration_strategy"] == "semantic_vertical_slice"
    assert contract["migration_mapping"] == {
        "program_control_and_effects": "flagquantum.compiler._hybrid",
        "static_quantum_region": "flagquantum.core.ir.CircuitIR",
        "circuit_transformation": "flagquantum.compiler",
        "execution_lifecycle": "flagquantum.runtime",
        "numerical_execution": "flagquantum.simulation",
    }
    assert gates == {
        "authoritative_integration_baseline_identified": True,
        "historical_tree_excluded_as_migration_target": True,
        "team_scope_preflight": True,
    }
    assert contract["implementation_started"] is True
    assert contract["phase1_completed"] is True


def test_phase2_capture_is_private_structural_and_non_executing() -> None:
    phase2 = _contract()["phase2"]

    assert phase2["private_entry_points"] == [
        "flagquantum.compiler._hybrid.capture_source",
        "flagquantum.compiler._hybrid.capture_function",
    ]
    assert {"if_elif_else", "for_range", "for_enumerate_tensor_axis"} <= set(
        phase2["supported_statements"]
    )
    assert {"tensor_extract", "comparison", "jax_numpy_mod_call"} <= set(
        phase2["supported_expressions"]
    )
    assert phase2["capture_executes_user_code"] is False
    assert phase2["runtime_tensor_predicates_evaluated_during_capture"] is False
    assert phase2["public_api_change"] is False
    assert phase2["default_path_change"] is False
    assert _contract()["phase2_completed"] is True


def test_phase3_reuses_circuit_ir_and_separates_values_from_structure() -> None:
    phase3 = _contract()["phase3"]

    assert phase3["semantic_flow"] == [
        "verified_hybrid_program",
        "runtime_path_specialization",
        "ephemeral_quantum_trace",
        "parameterized_circuit_ir_template",
        "late_bound_circuit_ir",
    ]
    assert phase3["parameter_values_excluded_from_structure_identity"] is True
    assert phase3["tensor_parameters_preserved_by_reference"] is True
    assert phase3["angle_embedding_lowering"] == "rx_per_wire"
    assert phase3["cache_policy"] == "bounded_visible_lru"
    assert phase3["public_api_change"] is False
    assert phase3["default_path_change"] is False
    assert phase3["runtime_execution"] is False
    assert _contract()["phase3_completed"] is True


def test_phase4_reuses_existing_execution_and_result_contracts() -> None:
    contract = _contract()
    phase4 = contract["phase4"]

    assert phase4["compiler_handoff_artifact"] == "flagquantum.core.ir.CircuitIR"
    assert (
        phase4["expectation_request_artifact"] == "flagquantum.core.ir.MeasurementNode"
    )
    assert phase4["runtime_entry_point"] == "flagquantum.runtime.execution.run"
    assert phase4["result_artifact"] == "flagquantum.runtime.result.ExecutionResult"
    assert phase4["result_accessor"] == "ExecutionResult.expectation"
    assert phase4["execution_mode"] == "statevector"
    assert phase4["execution_device"] == "cpu"
    assert phase4["distribution_semantics"] == "single_device_fast_path"
    assert phase4["expectation_shape"] == "one_scalar_per_batch_item"
    assert phase4["term_grouping"] == "shared_fq_output_index"
    assert phase4["compiler_emits_existing_measurement_contract"] is True
    assert phase4["runtime_imports_hybrid_compiler"] is False
    assert phase4["runtime_source_change_required"] is False
    assert phase4["simulation_source_change_required"] is False
    assert phase4["new_execution_entry_point"] is False
    assert phase4["new_result_type"] is False
    assert phase4["gradient_claim"] is False
    assert phase4["performance_claim"] is False
    assert phase4["public_api_change"] is False
    assert phase4["default_path_change"] is False
    assert contract["phase4_completed"] is True


def test_phase5_reuses_adjoint_and_preserves_branchwise_tensor_vjp() -> None:
    contract = _contract()
    phase5 = contract["phase5"]

    assert phase5["compiler_handoff_artifact"] == "flagquantum.core.ir.CircuitIR"
    assert phase5["gradient_executor"].endswith(
        ".execute_torch_distributed_statevector_reverse"
    )
    assert phase5["gradient_method"] == "statevector_adjoint"
    assert phase5["execution_scope"] == "local_cpu_single_device_fast_path"
    assert phase5["observable_profile"] == "sum_of_single_wire_pauli_z_terms"
    assert phase5["forward_execution_count"] == 1
    assert phase5["adjoint_seed"] == "sum_of_observable_term_adjoints"
    assert phase5["source_tensor_views_preserved"] is True
    assert phase5["control_flow_gradient"] == "selected_branch_only"
    assert phase5["predicate_gradient"] is False
    assert phase5["comparison_boundary"] == "fail_closed_when_smoothness_is_required"
    assert phase5["reference_methods"] == [
        "dense_pytorch_autograd",
        "central_finite_difference",
    ]
    assert phase5["parameter_shift_execution_fallback"] is False
    assert phase5["optimizer_acceptance"] == "seeded_objective_decreases"
    assert phase5["higher_order_gradient_claim"] is False
    assert phase5["finite_shot_gradient_claim"] is False
    assert phase5["noisy_gradient_claim"] is False
    assert phase5["accelerator_gradient_claim"] is False
    assert phase5["distributed_gradient_claim"] is False
    assert phase5["performance_claim"] is False
    assert phase5["public_api_change"] is False
    assert phase5["default_path_change"] is False
    assert contract["phase5_completed"] is True


def test_phase6_is_functional_explicit_and_fullgraph_verified() -> None:
    contract = _contract()
    phase6 = contract["phase6"]

    assert phase6["compiler_handoff_artifact"] == "flagquantum.core.ir.CircuitIR"
    assert phase6["parameter_order"] == "explicit_hybrid_parameter_order_metadata"
    assert phase6["operator_schema"] == (
        "(Tensor[] parameters, str circuit_ir_json) -> Tensor"
    )
    assert phase6["operator_semantics"] == "functional"
    assert phase6["mutates_inputs"] is False
    assert phase6["static_program_is_explicit_input"] is True
    assert phase6["hidden_program_registry"] is False
    assert phase6["fake_tensor_registered"] is True
    assert phase6["autograd_registered"] is True
    assert phase6["backward_is_separate_custom_operator"] is True
    assert phase6["backward_method"] == "statevector_adjoint"
    assert phase6["compile_scope"] == "specialized_quantum_region"
    assert phase6["evidence_report"] == (
        "docs/development/HYBRID_COMPILATION_PHASE6_EVIDENCE.md"
    )
    assert phase6["general_dynamic_python_graph_claim"] is False
    assert phase6["opcheck_required"] is True
    assert phase6["gradcheck_required"] is True
    assert phase6["fullgraph_required"] is True
    assert phase6["higher_order_gradient_claim"] is False
    assert phase6["execution_scope"] == "local_cpu_single_device_fast_path"
    assert phase6["performance_claim"] is False
    assert phase6["public_api_change"] is False
    assert phase6["default_path_change"] is False
    assert contract["phase6_completed"] is True


def test_phase7_reuses_dynamic_runtime_and_rejects_stochastic_gradients() -> None:
    contract = _contract()
    phase7 = contract["phase7"]

    assert {"quantum.h", "quantum.x", "quantum.measure", "scf.if"} <= set(
        phase7["phase7_operations"]
    )
    assert phase7["compiler_handoff_artifact"] == "flagquantum.core.ir.CircuitIR"
    assert phase7["runtime_result"].endswith(".DynamicExecutionResult")
    assert phase7["execution_scope"] == "local_cpu_statevector_trajectory"
    assert phase7["source_measurement_value"] == "bool_ssa_value"
    assert phase7["conditional_continuation"] == ("direct_measurement_bool_then_else")
    assert phase7["supported_gates"] == ["h", "x", "cx"]
    assert phase7["runtime_inputs_supported"] is False
    assert phase7["conditional_measurement_supported"] is False
    assert phase7["shot_semantics"] == "independent_trajectories"
    assert phase7["seeded_reproducibility_required"] is True
    assert phase7["stochastic_gradient_policy"] == "unsupported_fail_closed"
    assert phase7["durable_session_claim"] is False
    assert phase7["torch_compile_claim"] is False
    assert phase7["accelerator_claim"] is False
    assert phase7["distributed_claim"] is False
    assert phase7["performance_claim"] is False
    assert phase7["public_api_change"] is False
    assert phase7["default_path_change"] is False
    assert phase7["evidence_report"] == (
        "docs/development/HYBRID_COMPILATION_PHASE7_EVIDENCE.md"
    )
    assert contract["phase7_completed"] is True


def test_phase8_separates_dynamic_parameters_from_specialized_structure() -> None:
    contract = _contract()
    phase8 = contract["phase8"]

    assert phase8["template_artifact"] == "flagquantum.core.ir.CircuitIR"
    assert phase8["runtime_input_types"] == [
        "scalar_float32_or_float64",
        "index",
        "bool",
    ]
    assert phase8["tensor_runtime_inputs_supported"] is False
    assert phase8["parameterized_gates"] == ["rx", "ry"]
    assert phase8["parameter_binding"] == "explicit_ordered_core_parameter_slots"
    assert phase8["parameter_values_excluded_from_template_identity"] is True
    assert phase8["source_tensor_identity_preserved_by_binding"] is True
    assert phase8["classical_loop_profile"] == ("positive_bounded_range_specialization")
    assert phase8["default_max_unrolled_iterations"] == 10_000
    assert phase8["loop_measurement_supported"] is False
    assert phase8["input_dependent_structure_requires_respecialization"] is True
    assert phase8["trainable_inputs"] == "unsupported_fail_closed"
    assert phase8["stochastic_gradient_policy"] == "unsupported_fail_closed"
    assert phase8["runtime_execution"] == "existing_local_dynamic_trajectory"
    assert phase8["torch_compile_claim"] is False
    assert phase8["accelerator_claim"] is False
    assert phase8["distributed_claim"] is False
    assert phase8["performance_claim"] is False
    assert phase8["public_api_change"] is False
    assert phase8["default_path_change"] is False
    assert phase8["evidence_report"] == (
        "docs/development/HYBRID_COMPILATION_PHASE8_EVIDENCE.md"
    )
    assert contract["phase8_completed"] is True


def test_phase9_represents_loop_carried_classical_state_explicitly() -> None:
    contract = _contract()
    phase9 = contract["phase9"]

    assert phase9["control_operation"] == "scf.for"
    assert phase9["carried_value_types"] == [
        "scalar_float32_or_float64",
        "index",
        "bool",
    ]
    assert phase9["carried_value_representation"] == (
        "explicit_region_arguments_results_and_yields"
    )
    assert phase9["assignment_profile"] == ("direct_loop_body_local_name_rebinding")
    assert phase9["post_loop_values"] == "explicit_scf_for_results"
    assert phase9["loop_target_shadowing"] == "unsupported_fail_closed"
    assert phase9["tensor_carried_state"] == "unsupported_fail_closed"
    assert phase9["conditional_carried_state"] == "unsupported_fail_closed"
    assert phase9["nested_carried_state"] == "unsupported_fail_closed"
    assert phase9["loop_measurement_supported"] is False
    assert phase9["runtime_change_required"] is False
    assert phase9["simulation_change_required"] is False
    assert phase9["stochastic_gradient_policy"] == "unsupported_fail_closed"
    assert phase9["public_api_change"] is False
    assert phase9["default_path_change"] is False
    assert phase9["evidence_report"] == (
        "docs/development/HYBRID_COMPILATION_PHASE9_EVIDENCE.md"
    )
    assert contract["phase9_completed"] is True


def test_phase10_merges_branch_carried_state_without_runtime_mutation() -> None:
    contract = _contract()
    phase10 = contract["phase10"]

    assert phase10["control_operation"] == "scf.if"
    assert phase10["carried_value_types"] == [
        "scalar_float32_or_float64",
        "index",
        "bool",
    ]
    assert phase10["carried_value_representation"] == (
        "explicit_operands_region_arguments_yields_and_results"
    )
    assert phase10["assignment_profile"] == ("direct_branch_body_local_name_rebinding")
    assert phase10["one_sided_assignment"] == ("unchanged_block_argument_passthrough")
    assert phase10["post_branch_values"] == "explicit_scf_if_results"
    assert phase10["runtime_lowering_profile"] == (
        "specialization_resolved_predicates_only"
    )
    assert phase10["measurement_dependent_carried_state"] == ("unsupported_fail_closed")
    assert phase10["tensor_carried_state"] == "unsupported_fail_closed"
    assert phase10["nested_carried_state"] == "unsupported_fail_closed"
    assert phase10["runtime_change_required"] is False
    assert phase10["simulation_change_required"] is False
    assert phase10["stochastic_gradient_policy"] == "unsupported_fail_closed"
    assert phase10["public_api_change"] is False
    assert phase10["default_path_change"] is False
    assert phase10["evidence_report"] == (
        "docs/development/HYBRID_COMPILATION_PHASE10_EVIDENCE.md"
    )
    assert contract["phase10_completed"] is True


def test_phase11_threads_state_through_nested_structured_regions() -> None:
    contract = _contract()
    phase11 = contract["phase11"]

    assert phase11["assignment_discovery"] == ("recursive_structured_if_for_analysis")
    assert phase11["supported_nesting"] == [
        "for_contains_if",
        "if_contains_for",
    ]
    assert phase11["carried_value_types"] == [
        "scalar_float32_or_float64",
        "index",
        "bool",
    ]
    assert phase11["state_transport"] == (
        "explicit_ssa_at_every_enclosing_structured_region"
    )
    assert phase11["quantum_effect_transport"] == ("linear_region_argument_and_result")
    assert phase11["nested_loop_unroll_accounting"] == ("cumulative_configured_ceiling")
    assert phase11["measurement_dependent_carried_state"] == ("unsupported_fail_closed")
    assert phase11["loop_measurement_supported"] is False
    assert phase11["tensor_carried_state"] == "unsupported_fail_closed"
    assert phase11["runtime_change_required"] is False
    assert phase11["simulation_change_required"] is False
    assert phase11["public_api_change"] is False
    assert phase11["default_path_change"] is False
    assert phase11["performance_claim"] is False
    assert phase11["evidence_report"] == (
        "docs/development/HYBRID_COMPILATION_PHASE11_EVIDENCE.md"
    )
    assert contract["phase11_completed"] is True


def test_phase12_uses_existing_conjunctive_runtime_conditions() -> None:
    contract = _contract()
    phase12 = contract["phase12"]

    assert phase12["program_ir_operations"] == [
        "arith.not",
        "arith.and",
        "arith.cmp",
    ]
    assert phase12["measurement_predicate_profile"] == [
        "direct_measurement_bool",
        "single_literal_negation",
        "single_literal_bool_equality_or_inequality",
        "conjunction_of_distinct_measurement_literals",
    ]
    assert phase12["runtime_representation"] == (
        "existing_instruction_conditions_conjunction"
    )
    assert phase12["multiple_measurements_supported"] is True
    assert phase12["conjunction_else_branch"] == "must_have_no_quantum_work"
    assert phase12["disjunction_supported"] is False
    assert phase12["negated_conjunction_supported"] is False
    assert phase12["measurement_to_measurement_comparison_supported"] is False
    assert phase12["inline_measurement_composition"] == "unsupported_fail_closed"
    assert phase12["measurement_dependent_carried_state"] == ("unsupported_fail_closed")
    assert phase12["runtime_change_required"] is False
    assert phase12["simulation_change_required"] is False
    assert phase12["stochastic_gradient_policy"] == "unsupported_fail_closed"
    assert phase12["public_api_change"] is False
    assert phase12["default_path_change"] is False
    assert phase12["performance_claim"] is False
    assert phase12["evidence_report"] == (
        "docs/development/HYBRID_COMPILATION_PHASE12_EVIDENCE.md"
    )
    assert contract["phase12_completed"] is True


def test_phase13_uses_bounded_canonical_dnf_conditions() -> None:
    contract = _contract()
    phase13 = contract["phase13"]

    assert phase13["program_ir_operations"] == [
        "arith.not",
        "arith.and",
        "arith.or",
        "arith.cmp",
    ]
    assert phase13["measurement_predicate_profile"] == [
        "boolean_disjunction",
        "negated_conjunction",
        "measurement_bool_equality_or_inequality",
        "measurement_to_measurement_equality_or_inequality",
        "exact_quantum_if_else",
    ]
    assert phase13["runtime_representation"] == "canonical_dnf_condition_clauses"
    assert phase13["simple_conjunction_compatibility"] == (
        "existing_instruction_conditions"
    )
    assert phase13["canonicalization"] == [
        "sort_literals_and_clauses",
        "remove_contradictory_and_duplicate_clauses",
        "remove_subsumed_clauses",
    ]
    assert phase13["default_max_condition_clauses"] == 64
    assert phase13["clause_limit_policy"] == "unsupported_fail_closed"
    assert phase13["inline_measurement_composition"] == "unsupported_fail_closed"
    assert phase13["conditional_measurement"] == "unsupported_fail_closed"
    assert phase13["measurement_dependent_carried_state"] == ("unsupported_fail_closed")
    assert phase13["provider_dialect_complex_predicates"] == ("unsupported_fail_closed")
    assert phase13["stochastic_gradient_policy"] == "unsupported_fail_closed"
    assert phase13["public_api_change"] is False
    assert phase13["default_path_change"] is False
    assert phase13["performance_claim"] is False
    assert phase13["evidence_report"] == (
        "docs/development/HYBRID_COMPILATION_PHASE13_EVIDENCE.md"
    )
    assert contract["phase13_completed"] is True


def test_phase14_merges_measurement_dependent_ssa_values() -> None:
    contract = _contract()
    phase14 = contract["phase14"]

    assert phase14["value_representation"] == (
        "bounded_condition_partitioned_ssa_cases"
    )
    assert phase14["carried_value_types"] == [
        "scalar_float32_or_float64",
        "index",
        "bool",
    ]
    assert phase14["supported_consumers"] == [
        "arithmetic_add_or_remainder",
        "boolean_and_or_not",
        "scalar_or_bool_comparison",
        "quantum_gate_parameter",
        "quantum_gate_wire",
        "subsequent_structured_if_or_bounded_for",
    ]
    assert phase14["lowering_strategy"] == (
        "split_consumers_over_mutually_exclusive_canonical_conditions"
    )
    assert phase14["equal_case_coalescing"] is True
    assert phase14["case_limit"] == "max_condition_clauses"
    assert phase14["case_limit_policy"] == "unsupported_fail_closed"
    assert phase14["conditional_measurement"] == "unsupported_fail_closed"
    assert phase14["measurement_dependent_loop_bounds"] == ("unsupported_fail_closed")
    assert phase14["measurement_dependent_program_return"] == (
        "unsupported_fail_closed"
    )
    assert phase14["stochastic_gradient_policy"] == "unsupported_fail_closed"
    assert phase14["public_api_change"] is False
    assert phase14["default_path_change"] is False
    assert phase14["core_ir_schema_change"] is False
    assert phase14["performance_claim"] is False
    assert phase14["evidence_report"] == (
        "docs/development/HYBRID_COMPILATION_PHASE14_EVIDENCE.md"
    )
    assert contract["phase14_completed"] is True


def test_phase15_supports_fixed_round_syndrome_feedback() -> None:
    contract = _contract()
    phase15 = contract["phase15"]

    assert phase15["program_ir_operations_added"] == ["quantum.reset"]
    assert phase15["execution_profile"] == (
        "statically_bounded_repeated_syndrome_feedback"
    )
    assert phase15["loop_measurement_supported"] is True
    assert phase15["loop_reset_supported"] is True
    assert phase15["per_iteration_feedback_supported"] is True
    assert phase15["measurement_allocation"] == (
        "dense_monotonic_classical_bits_after_unrolling"
    )
    assert phase15["loop_carried_syndrome"] is True
    assert phase15["budgets"] == {
        "max_unrolled_iterations_default": 10_000,
        "max_dynamic_measurements_default": 4_096,
        "max_condition_clauses_default": 64,
    }
    assert phase15["conditional_measurement"] == "unsupported_fail_closed"
    assert phase15["conditional_reset"] == "unsupported_fail_closed"
    assert phase15["measurement_dependent_loop_termination"] == (
        "unsupported_fail_closed"
    )
    assert phase15["stochastic_gradient_policy"] == "unsupported_fail_closed"
    assert phase15["provider_execution"] == "unsupported_fail_closed"
    assert phase15["public_api_change"] is False
    assert phase15["default_path_change"] is False
    assert phase15["core_ir_schema_change"] is False
    assert phase15["performance_claim"] is False
    assert phase15["evidence_report"] == (
        "docs/development/HYBRID_COMPILATION_PHASE15_EVIDENCE.md"
    )
    assert contract["phase15_completed"] is True


def test_phase16_defines_bounded_repetition_memory_workflow() -> None:
    contract = _contract()
    phase16 = contract["phase16"]

    assert phase16["domain"] == "flagquantum.qec"
    assert phase16["execution_profile"] == (
        "three_data_qubit_repetition_bit_flip_memory"
    )
    assert phase16["checks"] == ["z0_z1_parity", "z1_z2_parity"]
    assert phase16["syndrome_bits_per_round"] == 2
    assert phase16["error_profile"] == ("zero_or_one_deterministic_x_on_data_wire")
    assert phase16["typed_records"] == [
        "DetectionEvent",
        "SyndromeRound",
        "Correction",
        "RepetitionMemoryShot",
        "RepetitionMemoryResult",
    ]
    assert phase16["decoder_contract"] == "flagquantum.qec.Decoder"
    assert phase16["reference_decoder"] == ("flagquantum.qec.RepetitionLookupDecoder")
    assert phase16["detection_event_semantics"] == (
        "temporal_syndrome_xor_from_initial_zero"
    )
    assert phase16["compiled_feedback_policy"] == "reference_lookup"
    assert phase16["replaceable_decoder_role"] == ("post_execution_syndrome_analysis")
    assert phase16["realtime_decoder_integration"] is False
    assert phase16["realistic_noise"] is False
    assert phase16["logical_error_suppression_claim"] is False
    assert phase16["threshold_claim"] is False
    assert phase16["fault_tolerance_claim"] is False
    assert phase16["public_root_export"] is False
    assert phase16["public_api_change"] is False
    assert phase16["default_path_change"] is False
    assert phase16["performance_claim"] is False
    assert phase16["evidence_report"] == (
        "docs/development/HYBRID_COMPILATION_PHASE16_EVIDENCE.md"
    )
    assert contract["phase16_completed"] is True


def test_phase17_separates_timed_errors_feedback_and_pauli_frames() -> None:
    contract = _contract()
    phase17 = contract["phase17"]

    assert phase17["domain"] == "flagquantum.qec"
    assert phase17["execution_profile"] == "bounded_timed_error_repetition_memory"
    assert phase17["error_schedule"] == (
        "canonical_deterministic_x_events_by_round_and_data_wire"
    )
    assert phase17["error_event_limit"] == 64
    assert phase17["error_injection_point"] == ("start_of_round_before_parity_checks")
    assert phase17["feedback_modes"] == [
        "compiled_lookup",
        "offline_pauli_frame",
    ]
    assert phase17["executed_feedback_recorded_separately"] is True
    assert phase17["decoder_input"] == "complete_ordered_syndrome_history"
    assert phase17["reference_decoder"] == "terminal_syndrome_lookup"
    assert phase17["decode_result"] == ("corrections_plus_parity_reduced_pauli_frame")
    assert phase17["offline_frame_application"] == "final_data_readout_only"
    assert phase17["single_error_any_round_verified"] is True
    assert phase17["double_error_logical_failure_recorded"] is True
    assert phase17["stochastic_noise"] is False
    assert phase17["measurement_error"] is False
    assert phase17["realtime_decoder_integration"] is False
    assert phase17["logical_error_suppression_claim"] is False
    assert phase17["threshold_claim"] is False
    assert phase17["fault_tolerance_claim"] is False
    assert phase17["public_root_export"] is False
    assert phase17["public_api_change"] is False
    assert phase17["default_path_change"] is False
    assert phase17["performance_claim"] is False
    assert phase17["evidence_report"] == (
        "docs/development/HYBRID_COMPILATION_PHASE17_EVIDENCE.md"
    )
    assert contract["phase17_completed"] is True


def test_phase18_bridges_bounded_noise_into_dynamic_execution() -> None:
    contract = _contract()
    phase18 = contract["phase18"]

    assert phase18["noise_authority"] == "flagquantum.noise.NoiseModel"
    assert phase18["runtime_entry_points"] == [
        "flagquantum.runtime.dynamic.run_dynamic",
        "flagquantum.runtime.dynamic.hybrid_session.execute_hybrid_dynamic_session",
    ]
    assert phase18["simulation_kernels"] == (
        "flagquantum.simulation.statevector.dynamic_noise"
    )
    assert phase18["supported_channels"] == [
        "independent_one_wire_bit_flip_after_matching_executed_gate"
    ]
    assert phase18["readout_semantics"] == (
        "independent_true_to_observed_confusion_on_explicit_measurement_and_final_sampling"
    )
    assert phase18["collapse_semantics"] == "physical_state_collapses_on_true_bit"
    assert phase18["feedback_semantics"] == (
        "classical_conditions_consume_observed_bit"
    )
    assert phase18["execution_strategies"] == ["trajectory", "batched"]
    assert phase18["seeded_reproducibility"] is True
    assert phase18["statistics"] == [
        "noise_model_identity",
        "noise_channel_application_count",
        "bit_flip_event_count",
        "readout_error_count",
    ]
    assert phase18["qec_middle_wire_noise_opportunities_per_round"] == 2
    assert phase18["qec_edge_wire_noise_opportunities_per_round"] == 1
    assert phase18["finite_shot_sweep"] == (
        "flagquantum.qec.run_repetition_memory_noise_sweep"
    )
    assert phase18["general_kraus_channels"] == "unsupported_fail_closed"
    assert phase18["correlated_readout"] == "unsupported_fail_closed"
    assert phase18["device_profile_timing_noise"] == "unsupported_fail_closed"
    assert phase18["noisy_gradients"] == "unsupported_fail_closed"
    assert phase18["realtime_decoder_integration"] is False
    assert phase18["logical_error_suppression_claim"] is False
    assert phase18["threshold_claim"] is False
    assert phase18["fault_tolerance_claim"] is False
    assert phase18["provider_execution"] == "unsupported_fail_closed"
    assert phase18["public_root_export"] is False
    assert phase18["default_path_change"] is False
    assert phase18["performance_claim"] is False
    assert phase18["evidence_report"] == (
        "docs/development/HYBRID_COMPILATION_PHASE18_EVIDENCE.md"
    )
    assert contract["phase18_completed"] is True


def test_phase19_runs_replaceable_decoders_inside_trajectory_execution() -> None:
    contract = _contract()
    phase19 = contract["phase19"]

    assert phase19["runtime_authority"] == "flagquantum.runtime.dynamic"
    assert phase19["qec_authority"] == "flagquantum.qec"
    assert phase19["feedback_contract"] == (
        "private_bounded_measurement_decision_points"
    )
    assert phase19["decoder_contract"] == "flagquantum.qec.StreamingDecoder"
    assert phase19["decoder_input"] == (
        "complete_available_ordered_syndrome_history_per_round"
    )
    assert phase19["feedback_modes"] == [
        "runtime_decoder",
        "runtime_pauli_frame",
    ]
    assert phase19["action_profile"] == ["none", "physical_x", "frame_x"]
    assert phase19["trace_fields"] == [
        "true_measurement_bits",
        "observed_measurement_bits",
        "decoder_action",
        "frame_before",
        "frame_after",
    ]
    assert phase19["execution_strategy"] == "trajectory"
    assert phase19["auto_strategy_resolution"] == (
        "trajectory_when_feedback_plan_present"
    )
    assert phase19["batched_feedback"] == "unsupported_fail_closed"
    assert phase19["replaceable_decoder_changes_execution"] is True
    assert phase19["compiled_runtime_frame_outcome_crosscheck"] is True
    assert phase19["stable_plugin_type"] is False
    assert phase19["hard_realtime_claim"] is False
    assert phase19["provider_execution"] == "unsupported_fail_closed"
    assert phase19["general_code_support"] is False
    assert phase19["logical_error_suppression_claim"] is False
    assert phase19["threshold_claim"] is False
    assert phase19["fault_tolerance_claim"] is False
    assert phase19["public_root_export"] is False
    assert phase19["default_path_change"] is False
    assert phase19["performance_claim"] is False
    assert phase19["evidence_report"] == (
        "docs/development/HYBRID_COMPILATION_PHASE19_EVIDENCE.md"
    )
    assert contract["phase19_completed"] is True


def test_phase20_adds_bounded_detection_event_temporal_decoding() -> None:
    contract = _contract()
    phase20 = contract["phase20"]

    assert phase20["decoder"] == "flagquantum.qec.RepetitionTemporalDecoder"
    assert phase20["input"] == (
        "complete_available_syndrome_and_detection_event_history"
    )
    assert phase20["confirmation_rounds"] == 2
    assert phase20["data_error_rule"] == (
        "same_nonzero_syndrome_in_two_consecutive_rounds_with_no_second_boundary_event"
    )
    assert phase20["isolated_readout_rule"] == (
        "paired_temporal_detection_events_returning_to_prior_syndrome_produce_no_correction"
    )
    assert phase20["feedback_modes"] == [
        "runtime_temporal_decoder",
        "runtime_temporal_pauli_frame",
    ]
    assert phase20["confirmed_error_window"] == "rounds_zero_through_n_minus_two"
    assert phase20["terminal_round_error"] == "visible_but_unconfirmed"
    assert phase20["malformed_detection_history"] == "unsupported_fail_closed"
    assert phase20["seeded_readout_noise_observation"] == (
        "fewer_spurious_feedback_actions_than_immediate_lookup_in_checked_profile"
    )
    assert phase20["maximum_likelihood_decoder"] is False
    assert phase20["measurement_error_tolerance_claim"] == (
        "bounded_isolated_readout_oracle_only"
    )
    assert phase20["logical_error_suppression_claim"] is False
    assert phase20["threshold_claim"] is False
    assert phase20["general_code_support"] is False
    assert phase20["hard_realtime_claim"] is False
    assert phase20["provider_execution"] == "unsupported_fail_closed"
    assert phase20["public_root_export"] is False
    assert phase20["default_path_change"] is False
    assert phase20["performance_claim"] is False
    assert phase20["evidence_report"] == (
        "docs/development/HYBRID_COMPILATION_PHASE20_EVIDENCE.md"
    )
    assert contract["phase20_completed"] is True


def test_phase21_adds_a_verified_program_normalization_stage() -> None:
    contract = _contract()
    phase21 = contract["phase21"]

    assert phase21["stage"] == "verified_hybrid_program_normalization"
    assert phase21["input_ir"] == phase21["output_ir"]
    assert phase21["analyses"] == [
        "constant_values",
        "whole_program_ssa_use_counts",
        "operation_count",
    ]
    assert phase21["passes"] == [
        "constant_fold",
        "dead_constant_elimination",
    ]
    assert phase21["dead_operation_scope"] == "unused_arith_constants_only"
    assert phase21["verification"] == "before_pipeline_and_after_every_pass"
    assert phase21["maximum_passes"] == 32
    assert phase21["audit_fields"] == [
        "source_program_identity",
        "optimized_program_identity",
        "pass_name",
        "input_operation_count",
        "output_operation_count",
    ]
    assert phase21["static_specialization_integration"] is True
    assert phase21["dynamic_lowering_integration"] is True
    assert phase21["unoptimized_differential_oracle"] is True
    assert phase21["target_ir_added"] is False
    assert phase21["target_dialect_added"] is False
    assert phase21["stable_pass_extension_api"] is False
    assert phase21["public_root_export"] is False
    assert phase21["stable_api_change"] is False
    assert phase21["performance_claim"] is False
    assert phase21["evidence_report"] == (
        "docs/development/HYBRID_COMPILATION_PHASE21_EVIDENCE.md"
    )
    assert contract["phase21_completed"] is True


def test_phase22_simplifies_only_compile_time_structured_control() -> None:
    contract = _contract()
    phase22 = contract["phase22"]

    assert phase22["pass"] == "structured_control_flow_simplification"
    assert phase22["constant_if"] == (
        "inline_selected_region_and_rewire_results_to_yield_values"
    )
    assert phase22["empty_for"] == (
        "remove_loop_and_rewire_results_to_initial_carried_values"
    )
    assert phase22["rewired_values"] == [
        "classical_ssa",
        "linear_quantum_effect",
    ]
    assert phase22["preserved_dynamic_control"] == (
        "nonconstant_if_and_nonempty_or_dynamic_for"
    )
    assert phase22["verification"] == "after_control_flow_transformation"
    assert phase22["static_lowering_differential_oracle"] is True
    assert phase22["dynamic_lowering_differential_oracle"] is True
    assert phase22["representative_operation_count"] == {
        "before": 19,
        "after": 5,
    }
    assert phase22["general_loop_unrolling"] is False
    assert phase22["loop_invariant_code_motion"] is False
    assert phase22["target_ir_added"] is False
    assert phase22["stable_pass_extension_api"] is False
    assert phase22["public_root_export"] is False
    assert phase22["stable_api_change"] is False
    assert phase22["performance_claim"] is False
    assert phase22["evidence_report"] == (
        "docs/development/HYBRID_COMPILATION_PHASE22_EVIDENCE.md"
    )
    assert contract["phase22_completed"] is True


def test_phase23_unrolls_only_small_constant_entry_loops() -> None:
    contract = _contract()
    phase23 = contract["phase23"]

    assert phase23["pass"] == "bounded_loop_unroll"
    assert phase23["scope"] == "direct_entry_block_constant_bound_loops"
    assert phase23["maximum_iterations_per_loop"] == 8
    assert phase23["maximum_expanded_operations_per_pipeline"] == 256
    assert phase23["bound_requirements"] == (
        "compile_time_integer_lower_upper_and_nonzero_step"
    )
    assert phase23["iteration_semantics"] == [
        "fresh_ssa_definitions_per_iteration",
        "explicit_induction_constant_per_iteration",
        "classical_carried_value_chaining",
        "linear_quantum_effect_chaining",
    ]
    assert phase23["over_budget_behavior"] == "preserve_structured_loop"
    assert phase23["preexpanded_iteration_audit"] is True
    assert phase23["specialization_unroll_limit_conserved"] is True
    assert phase23["dynamic_lowering_unroll_limit_conserved"] is True
    assert phase23["parameter_binding_preserved"] is True
    assert phase23["autograd_edge_preserved"] is True
    assert phase23["optimized_unoptimized_circuit_equivalence"] is True
    assert phase23["nested_loop_unroll"] is False
    assert phase23["general_loop_unrolling"] is False
    assert phase23["target_ir_added"] is False
    assert phase23["public_root_export"] is False
    assert phase23["stable_api_change"] is False
    assert phase23["performance_claim"] is False
    assert phase23["evidence_report"] == (
        "docs/development/HYBRID_COMPILATION_PHASE23_EVIDENCE.md"
    )
    assert contract["phase23_completed"] is True


def test_phase24_adds_pass_audit_and_seeded_differential_verification() -> None:
    contract = _contract()
    phase24 = contract["phase24"]

    assert phase24["module_boundaries"] == {
        "analysis": "flagquantum.compiler._hybrid.analysis",
        "ssa_rewrites": "flagquantum.compiler._hybrid.rewrites",
        "transformations": "flagquantum.compiler._hybrid.transforms",
        "pipeline_and_audit": "flagquantum.compiler._hybrid.passes",
    }
    assert phase24["pass_outcome"] == "immutable_program_statistics_and_remarks"
    assert phase24["statistics"] == [
        "folded_operations",
        "constant_branches_inlined",
        "empty_loops_removed",
        "constants_removed",
        "loops_unrolled",
        "iterations_unrolled",
        "expanded_operations",
        "loops_preserved_by_reason",
    ]
    assert phase24["budget_skip_remarks"] == ("deterministic_loop_identity_and_reason")
    assert phase24["seeded_differential_cases"] == 12
    assert phase24["differential_oracles"] == [
        "circuit_instruction_structure",
        "bound_parameter_values",
        "statevector_expectation",
        "statevector_adjoint_vjp",
        "optimization_fixed_point",
    ]
    assert phase24["negative_step_profile"] == (
        "unsupported_equally_with_optimization_on_or_off"
    )
    assert phase24["pass_statistics_affect_program_identity"] is False
    assert phase24["new_optimization_semantics"] is False
    assert phase24["target_ir_added"] is False
    assert phase24["public_root_export"] is False
    assert phase24["stable_api_change"] is False
    assert phase24["performance_claim"] is False
    assert phase24["evidence_report"] == (
        "docs/development/HYBRID_COMPILATION_PHASE24_EVIDENCE.md"
    )
    assert contract["phase24_completed"] is True


def test_phase25_adds_capability_driven_target_legality_without_target_ir() -> None:
    contract = _contract()
    phase25 = contract["phase25"]

    assert phase25["input_ir"] == "flagquantum.core.ir.CircuitIR"
    assert phase25["output_ir"] == "same_flagquantum.core.ir.CircuitIR"
    assert phase25["derived_mandatory_requirements"] == [
        "qubits.logical_capacity",
        "limits.maximum_program_operations",
        "precision.effective_dtype",
        "measurements.results_when_present",
        "limits.maximum_shots_when_finite",
    ]
    assert phase25["operator_legality"] == (
        "existing_compiler_operator_lowering_registry"
    )
    assert phase25["target_evidence"] == (
        "caller_supplied_core_target_capabilities_v1_snapshot"
    )
    assert phase25["fallback_authorizations"] == "all_false"
    assert phase25["circuit_mutation"] is False
    assert phase25["live_parameter_reference_preserved"] is True
    assert phase25["native_gate_descriptor_matching"] is False
    assert phase25["decomposition"] is False
    assert phase25["topology_routing"] is False
    assert phase25["scheduling"] is False
    assert phase25["target_emission"] is False
    assert phase25["target_selection"] is False
    assert phase25["target_ir_added"] is False
    assert phase25["public_root_export"] is False
    assert phase25["stable_api_change"] is False
    assert phase25["default_path_change"] is False
    assert phase25["performance_claim"] is False
    assert phase25["evidence_report"] == (
        "docs/development/HYBRID_COMPILATION_PHASE25_EVIDENCE.md"
    )
    assert contract["phase25_completed"] is True


def test_phase26_adds_evidenced_bounded_native_gate_decomposition() -> None:
    contract = _contract()
    phase26 = contract["phase26"]

    assert phase26["input_ir"] == "flagquantum.core.ir.CircuitIR"
    assert phase26["output_ir"] == "flagquantum.core.ir.CircuitIR"
    assert phase26["native_gate_source"] == (
        "verified_target_capability_snapshot_gates_native"
    )
    assert phase26["separate_variants_preserved"] is True
    assert phase26["exact_decompositions"] == {
        "x": ["h", "z", "h"],
        "rx": ["h", "rz", "h"],
        "ry": ["sdg", "h", "rz", "h", "s"],
    }
    assert phase26["maximum_added_operations"] == 256
    assert phase26["parameter_object_identity_preserved"] is True
    assert phase26["dynamic_condition_metadata_preserved"] is True
    assert phase26["post_decomposition_target_requirements"] is True
    assert phase26["arbitrary_synthesis"] is False
    assert phase26["approximation"] is False
    assert phase26["parameter_domain_matching"] is False
    assert phase26["topology_routing"] is False
    assert phase26["scheduling"] is False
    assert phase26["target_emission"] is False
    assert phase26["target_ir_added"] is False
    assert phase26["public_root_export"] is False
    assert phase26["stable_api_change"] is False
    assert phase26["default_path_change"] is False
    assert phase26["performance_claim"] is False
    assert phase26["evidence_report"] == (
        "docs/development/HYBRID_COMPILATION_PHASE26_EVIDENCE.md"
    )
    assert contract["phase26_completed"] is True


def test_phase27_adds_bounded_topology_legality_without_target_ir() -> None:
    contract = _contract()
    phase27 = contract["phase27"]

    assert contract["status"] == (
        "phase28_dependency_preserving_logical_scheduling_verified"
    )
    assert phase27["input_ir"] == "flagquantum.core.ir.CircuitIR"
    assert phase27["output_ir"] == "flagquantum.core.ir.CircuitIR"
    assert phase27["topology_source"] == ("caller_supplied_compiler_coupling_map")
    assert phase27["target_snapshot_binding"] is True
    assert phase27["snapshot_proves_topology_provenance"] is False
    assert phase27["coupling_direction"] == "undirected"
    assert phase27["strategies"] == [
        "restore_after_each_gate",
        "persistent_layout",
        "auto_deterministic_cost_selection",
    ]
    assert phase27["path_cache_history_affects_identity"] is False
    assert phase27["postconditions"] == [
        "all_two_wire_instructions_are_coupling_edges",
        "identity_output_layout_restored",
        "non_negative_inserted_swap_count",
    ]
    assert phase27["maximum_added_operations"] == 256
    assert phase27["swap_decomposition"] == [
        "cx_forward",
        "cx_reverse",
        "cx_forward",
    ]
    assert phase27["pipeline_order"] == [
        "topology_routing",
        "native_gate_legalization",
        "backend_lowering_validation",
        "target_resource_matching",
    ]
    assert phase27["parameter_object_identity_preserved"] is True
    assert phase27["physical_ancilla_allocation"] is False
    assert phase27["directed_coupling"] is False
    assert phase27["calibration_aware_routing"] is False
    assert phase27["scheduling"] is False
    assert phase27["target_ir_added"] is False
    assert phase27["public_root_export"] is False
    assert phase27["stable_api_change"] is False
    assert phase27["default_path_change"] is False
    assert phase27["performance_claim"] is False
    assert phase27["evidence_report"] == (
        "docs/development/HYBRID_COMPILATION_PHASE27_EVIDENCE.md"
    )
    assert contract["phase27_completed"] is True


def test_phase28_adds_dependency_preserving_logical_schedule_evidence() -> None:
    contract = _contract()
    phase28 = contract["phase28"]

    assert contract["status"] == (
        "phase28_dependency_preserving_logical_scheduling_verified"
    )
    assert phase28["input_ir"] == "flagquantum.core.ir.CircuitIR"
    assert phase28["output_artifact"] == (
        "immutable_instruction_index_schedule_evidence"
    )
    assert phase28["scheduling_policy"] == ("deterministic_asap_logical_unit_layers")
    assert phase28["dependency_types"] == [
        "per_wire_source_order",
        "classical_measurement_producer_to_condition_consumer",
        "conservative_global_barrier",
    ]
    assert phase28["global_barriers"] == [
        "dynamic_operations",
        "conditioned_operations",
        "channels",
    ]
    assert phase28["condition_policy"] == (
        "binary_pairs_and_prior_measurement_required"
    )
    assert phase28["maximum_depth_policy"] == ("optional_non_negative_fail_closed")
    assert phase28["pipeline_position"] == (
        "after_routing_native_legalization_and_resource_matching"
    )
    assert phase28["gate_duration_semantics"] is False
    assert phase28["measurement_feedback_latency"] is False
    assert phase28["crosstalk_or_fidelity_scheduling"] is False
    assert phase28["pulse_scheduling"] is False
    assert phase28["schedule_execution"] is False
    assert phase28["target_ir_added"] is False
    assert phase28["public_root_export"] is False
    assert phase28["stable_api_change"] is False
    assert phase28["default_path_change"] is False
    assert phase28["performance_claim"] is False
    assert phase28["evidence_report"] == (
        "docs/development/HYBRID_COMPILATION_PHASE28_EVIDENCE.md"
    )
    assert contract["phase28_completed"] is True
