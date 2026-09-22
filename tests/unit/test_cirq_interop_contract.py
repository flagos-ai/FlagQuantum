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
