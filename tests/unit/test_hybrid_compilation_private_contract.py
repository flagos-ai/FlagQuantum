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
