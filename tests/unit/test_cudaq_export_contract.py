from __future__ import annotations

from pathlib import Path

import pytest

from tools.check_cudaq_export_contract import contract_errors, load_toml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def _inputs() -> tuple[dict, dict]:
    return (
        load_toml(ROOT / "contracts" / "cudaq-export-contract.toml"),
        load_toml(ROOT / "dependency-policy.toml"),
    )


def test_cudaq_export_contract_is_current() -> None:
    assert contract_errors(*_inputs()) == ()


def test_contract_rejects_hidden_public_api_and_bidirectional_claim() -> None:
    contract, policy = _inputs()
    contract["public_api_available"] = False
    contract["semantics"]["direction"] = "bidirectional"
    errors = contract_errors(contract, policy)
    assert "CUDA-Q public API must remain available after implementation" in errors
    assert "CUDA-Q semantic 'direction' drifted" in errors


def test_contract_rejects_dependency_and_version_drift() -> None:
    contract, policy = _inputs()
    policy["extras"]["cudaq"] = ["cudaq>=0.16,<1"]
    contract["cudaq_versions"] = ["0.16.0.post1"]
    errors = contract_errors(contract, policy)
    assert "CUDA-Q optional dependency range drifted" in errors
    assert "CUDA-Q candidate lanes must be 0.15.1 and 0.16.0.post1" in errors


def test_contract_rejects_portable_aggregate_contamination() -> None:
    contract, policy = _inputs()
    policy["aggregates"]["interop-all"].append("cudaq")
    errors = contract_errors(contract, policy)
    assert "CUDA-Q must remain outside the portable interop aggregate" in errors


def test_contract_rejects_incomplete_opcode_partition() -> None:
    contract, policy = _inputs()
    contract["unsupported"]["flagquantum_opcodes"].remove("i")
    errors = contract_errors(contract, policy)
    assert any("CUDA-Q opcode coverage drifted" in error for error in errors)


def test_ci_checks_contract_and_real_sdk_lane() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "python tools/check_cudaq_export_contract.py" in workflow
    assert "cudaq-optional:" in workflow
    assert 'OPTIONAL_VERSIONS: "0.15.1 0.16.0.post1"' in workflow
    assert "for version in ${OPTIONAL_VERSIONS}" in workflow
    assert '"cudaq==${version}"' in workflow
