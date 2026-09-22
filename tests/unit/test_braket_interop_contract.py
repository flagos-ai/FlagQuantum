from __future__ import annotations

from pathlib import Path

import pytest

from tools.check_braket_interop_contract import contract_errors, load_toml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def _inputs() -> tuple[dict, dict]:
    return (
        load_toml(ROOT / "contracts" / "braket-circuit-interop-contract.toml"),
        load_toml(ROOT / "dependency-policy.toml"),
    )


def test_braket_circuit_interop_contract_is_current() -> None:
    assert contract_errors(*_inputs()) == ()


def test_contract_rejects_premature_api_and_execution_claims() -> None:
    contract, policy = _inputs()
    contract["public_api_available"] = True
    contract["runtime_execution_allowed"] = True
    contract["cloud_submission_allowed"] = True
    errors = contract_errors(contract, policy)
    assert "Braket circuit API must remain unavailable before implementation" in errors
    assert (
        "Braket contract field 'runtime_execution_allowed' must remain false" in errors
    )
    assert (
        "Braket contract field 'cloud_submission_allowed' must remain false" in errors
    )


def test_contract_rejects_dependency_and_version_drift() -> None:
    contract, policy = _inputs()
    policy["extras"]["braket"] = ["amazon-braket-sdk>=1.120,<2"]
    contract["amazon_braket_sdk_versions"] = ["1.127.1"]
    errors = contract_errors(contract, policy)
    assert "Braket optional dependency range drifted" in errors
    assert "Braket candidate lanes must be 1.117.0 and 1.127.1" in errors


def test_contract_rejects_incomplete_or_ambiguous_opcode_partition() -> None:
    contract, policy = _inputs()
    contract["unsupported"]["flagquantum_opcodes"].remove("u1")
    contract["operations"][1]["braket_gate_type"] = "I"
    errors = contract_errors(contract, policy)
    assert any("Braket opcode coverage drifted" in error for error in errors)
    assert "Braket gate mappings must be unambiguous" in errors


def test_contract_keeps_implementation_evidence_deferred() -> None:
    contract, policy = _inputs()
    contract["verification"]["sdk_lane_deferred_until_implementation"] = False
    contract["verification"]["planned_conformance"] = "tests/changed.py"
    errors = contract_errors(contract, policy)
    assert "Braket SDK lane must remain explicitly deferred" in errors
    assert "Braket planned_conformance path drifted" in errors


def test_ci_checks_contract_without_claiming_an_sdk_lane() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "python tools/check_braket_interop_contract.py" in workflow
    assert "braket-optional:" not in workflow
