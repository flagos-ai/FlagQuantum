from __future__ import annotations

from pathlib import Path

import pytest

from tools.check_cirq_interop_contract import contract_errors, load_toml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def _inputs() -> tuple[dict, dict]:
    return (
        load_toml(ROOT / "contracts" / "cirq-interop-contract.toml"),
        load_toml(ROOT / "dependency-policy.toml"),
    )


def test_cirq_interop_contract_is_current() -> None:
    assert contract_errors(*_inputs()) == ()


def test_contract_rejects_public_api_and_version_drift() -> None:
    contract, policy = _inputs()
    contract["public_api_available"] = False
    contract["cirq_core_versions"] = ["1.7.0"]
    errors = contract_errors(contract, policy)
    assert "Cirq public API must remain available after implementation" in errors
    assert "Cirq certification lanes must be 1.6.1 and 1.7.0" in errors


def test_contract_rejects_incomplete_opcode_partition() -> None:
    contract, policy = _inputs()
    contract["unsupported"]["flagquantum_opcodes"].remove("sdg")
    errors = contract_errors(contract, policy)
    assert any("Cirq opcode coverage drifted" in error for error in errors)


def test_symbolic_parameter_extension_is_implemented() -> None:
    contract, policy = _inputs()
    symbolic_parameters = contract["symbolic_parameter_contract"]
    assert symbolic_parameters["implementation_status"] == "implemented"
    assert symbolic_parameters["public_api_change"] is False
    assert symbolic_parameters["supported_nodes"] == [
        "symbol",
        "real_constant",
        "add",
        "mul",
        "neg",
    ]
    assert symbolic_parameters["external_object_retention"] is False
    assert "symbolic_parameters" not in contract["unsupported"]["cirq_features"]
    assert contract_errors(contract, policy) == ()


def test_contract_rejects_symbolic_parameter_scope_drift() -> None:
    contract, policy = _inputs()
    symbolic_parameters = contract["symbolic_parameter_contract"]
    symbolic_parameters["implementation_status"] = "contract_only"
    symbolic_parameters["supported_nodes"].append("power")
    symbolic_parameters["external_object_retention"] = True
    errors = contract_errors(contract, policy)
    assert "Cirq symbolic-parameter extension contract drifted" in errors


def test_contract_requires_symbolic_parameter_implementation_evidence() -> None:
    contract, policy = _inputs()
    contract["unsupported"]["cirq_features"].append("symbolic_parameters")
    contract["unsupported"]["issue_codes"].remove("unsupported_parameter_expression")
    errors = contract_errors(contract, policy)
    assert "Cirq symbolic parameters must not remain globally unsupported" in errors
    assert "Cirq must declare its symbolic-expression rejection code" in errors


def test_measurement_extension_is_implemented_for_static_conversion() -> None:
    contract, policy = _inputs()
    measurement = contract["measurement_contract"]
    assert measurement["implementation_status"] == "implemented"
    assert measurement["runtime_execution"] == "out_of_scope"
    assert measurement["capability_gap_status"] == "partial"
    assert measurement["current_rejection_issue_code"] == "measurement_not_represented"
    assert measurement["metadata_fields"] == ["classical_bit", "measurement_key"]
    assert measurement["external_object_retention"] is False
    assert "measurements" not in contract["unsupported"]["cirq_features"]
    assert contract_errors(contract, policy) == ()


def test_measurement_contract_rejects_requested_unsupported_forms() -> None:
    contract, _ = _inputs()
    measurement = contract["measurement_contract"]
    assert {
        "mid_circuit_measurement",
        "measurement_invert_mask",
        "measurement_confusion_map",
        "duplicate_measurement_key",
    } <= set(measurement["rejected_forms"])
    assert {
        "measurement_not_terminal",
        "measurement_invert_mask_not_representable",
        "measurement_confusion_map_not_representable",
        "duplicate_measurement_key",
    } <= set(measurement["rejection_issue_codes"])
    assert measurement["terminal_requirement"] == (
        "every_measurement_follows_all_non_measurement_operations"
    )
    assert measurement["classical_bit_owner"] == "flagquantum_ecosystem_cirq_adapter"
    assert measurement["metadata_owner"] == "flagquantum_ir_instruction_metadata"


def test_contract_rejects_measurement_scope_drift() -> None:
    contract, policy = _inputs()
    measurement = contract["measurement_contract"]
    measurement["implementation_status"] = "contract_only"
    measurement["runtime_execution"] = "in_scope"
    measurement["capability_gap_status"] = "supported"
    measurement["rejected_forms"].remove("mid_circuit_measurement")
    errors = contract_errors(contract, policy)
    assert "Cirq measurement extension contract drifted" in errors


def test_contract_rejects_measurement_remaining_globally_unsupported() -> None:
    contract, policy = _inputs()
    contract["unsupported"]["cirq_features"].append("measurements")
    errors = contract_errors(contract, policy)
    assert "Implemented Cirq measurements must not remain unsupported" in errors


def test_custom_unitary_extension_is_contract_only_and_unsupported() -> None:
    contract, policy = _inputs()
    custom_unitary = contract["custom_unitary_contract"]
    assert custom_unitary["implementation_status"] == "contract_only"
    assert custom_unitary["runtime_execution"] == "out_of_scope"
    assert custom_unitary["capability_gap_status"] == "unsupported"
    assert custom_unitary["current_rejection_issue_code"] == "unsupported_operation"
    assert custom_unitary["source_artifact"] == "cirq.MatrixGate"
    assert custom_unitary["source_matrix_protocol"] == "cirq.unitary"
    assert custom_unitary["supported_width_qubits"] == [1, 2, 3]
    assert custom_unitary["matrix_basis_order"] == (
        "identity_first_operation_qubit_is_most_significant"
    )
    assert custom_unitary["external_object_retention"] is False
    assert "matrix_gates" in contract["unsupported"]["cirq_features"]
    assert contract_errors(contract, policy) == ()


def test_custom_unitary_contract_rejects_requested_unsupported_forms() -> None:
    contract, _ = _inputs()
    custom_unitary = contract["custom_unitary_contract"]
    assert {
        "custom_unitary_above_three_qubits",
        "custom_unitary_qid_dimension_above_two",
        "invalid_custom_unitary_shape",
        "invalid_custom_unitary_values",
        "non_unitary_custom_matrix",
        "unavailable_custom_unitary_matrix",
    } == set(custom_unitary["rejected_forms"])
    assert {
        "custom_unitary_width_exceeds_limit",
        "custom_unitary_qid_dimension_not_representable",
        "invalid_custom_unitary_shape",
        "invalid_custom_unitary_values",
        "non_unitary_custom_matrix",
        "custom_unitary_not_representable",
    } == set(custom_unitary["rejection_issue_codes"])
    assert custom_unitary["matrix_target"] == (
        "detached_cpu_torch_complex128_owned_copy"
    )
    assert custom_unitary["unitarity_check"] == "u_dagger_u_equals_identity"
    assert custom_unitary["finite_values_required"] is True


def test_contract_rejects_custom_unitary_scope_drift() -> None:
    contract, policy = _inputs()
    custom_unitary = contract["custom_unitary_contract"]
    custom_unitary["implementation_status"] = "implemented"
    custom_unitary["supported_width_qubits"].append(4)
    custom_unitary["matrix_basis_order"] = "unspecified"
    custom_unitary["external_object_retention"] = True
    errors = contract_errors(contract, policy)
    assert "Cirq custom-unitary extension contract drifted" in errors


def test_contract_requires_matrix_gates_to_stay_unsupported() -> None:
    contract, policy = _inputs()
    contract["unsupported"]["cirq_features"].remove("matrix_gates")
    errors = contract_errors(contract, policy)
    assert "Cirq matrix gates must remain unsupported until implemented" in errors
