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
