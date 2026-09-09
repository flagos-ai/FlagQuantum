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
