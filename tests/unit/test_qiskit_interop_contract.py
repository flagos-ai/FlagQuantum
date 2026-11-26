from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from tools.check_qiskit_interop_contract import contract_errors, load_toml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def _inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        load_toml(ROOT / "contracts" / "qiskit-interop-contract.toml"),
        load_toml(ROOT / "dependency-policy.toml"),
    )


def test_qiskit_interop_contract_is_current() -> None:
    assert contract_errors(*_inputs()) == ()


def test_approved_acos_contract_preserves_symbolic_floor_export_boundary() -> None:
    contract, policy = _inputs()
    symbolic = contract["symbolic_parameter_contract"]
    assert symbolic["proposal"] == "API_CHANGE_PROPOSAL_128_PARAMETER_ACOS_FLOOR"
    assert "acos" in symbolic["supported_nodes"]
    assert "floor" not in symbolic["supported_nodes"]
    assert (
        "symbolic_floor_export_without_numeric_binding"
        in contract["unsupported"]["qiskit_features"]
    )
    symbolic["supported_nodes"].append("floor")
    assert (
        "Qiskit symbolic parameter contract drifted from Proposal 128"
        in contract_errors(contract, policy)
    )


def test_contract_rejects_operator_coverage_drift() -> None:
    contract, policy = _inputs()
    contract["operations"] = copy.deepcopy(contract["operations"][:-1])
    errors = contract_errors(contract, policy)
    assert any("opcode coverage drifted" in error for error in errors)


def test_contract_rejects_parameter_order_drift() -> None:
    contract, policy = _inputs()
    operation = next(
        item for item in contract["operations"] if item["flagquantum"] == "u3"
    )
    operation["flagquantum_parameters"] = ["theta", "lbd", "phi"]
    errors = contract_errors(contract, policy)
    assert "FlagQuantum parameter order drifted for 'u3'" in errors


def test_contract_rejects_version_policy_drift() -> None:
    contract, policy = _inputs()
    policy["tested"]["qiskit"] = ["2.5"]
    errors = contract_errors(contract, policy)
    assert "Qiskit version lanes must match dependency-policy.toml" in errors


def test_contract_rejects_wire_semantic_drift() -> None:
    contract, policy = _inputs()
    contract["semantics"]["qiskit_statevector_order"] = "unspecified"
    errors = contract_errors(contract, policy)
    assert any("semantics drifted" in error for error in errors)


def test_contract_rejects_issue_code_drift() -> None:
    contract, policy = _inputs()
    contract["unsupported"]["issue_codes"].pop()
    errors = contract_errors(contract, policy)
    assert any("issue-code coverage drifted" in error for error in errors)
