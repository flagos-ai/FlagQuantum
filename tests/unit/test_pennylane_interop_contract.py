from __future__ import annotations

from pathlib import Path

import pytest

from tools.check_pennylane_interop_contract import contract_errors, load_toml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_pennylane_interop_contract_is_current() -> None:
    assert (
        contract_errors(
            load_toml(ROOT / "contracts" / "pennylane-interop-contract.toml"),
            load_toml(ROOT / "dependency-policy.toml"),
        )
        == ()
    )


def test_contract_rejects_runtime_scope_and_version_drift() -> None:
    contract = load_toml(ROOT / "contracts" / "pennylane-interop-contract.toml")
    policy = load_toml(ROOT / "dependency-policy.toml")
    contract["runtime_execution_allowed"] = True
    contract["pennylane_versions"] = ["0.45.1"]
    errors = contract_errors(contract, policy)
    assert "PennyLane runtime execution must remain outside v1" in errors
    assert "PennyLane certification lanes must be 0.44.1 and 0.45.1" in errors
