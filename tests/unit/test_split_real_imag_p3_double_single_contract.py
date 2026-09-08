from __future__ import annotations

import copy
from pathlib import Path

import pytest

from flagquantum.runtime.executors.statevector.split_real_imag_double_single import (
    split_real_imag_p3_accuracy_envelope,
    split_real_imag_p3_precision_plan,
)
from tools.check_split_real_imag_p3_double_single_contract import (
    contract_errors,
    load_toml,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
CONTRACT = (
    ROOT / "contracts" / "split-real-imag-statevector-p3-double-single-contract.toml"
)


def test_checked_in_split_p3_contract_passes() -> None:
    assert contract_errors(load_toml(CONTRACT)) == ()


def test_split_p3_contract_rejects_hidden_host_or_claim_promotion() -> None:
    contract = copy.deepcopy(load_toml(CONTRACT))
    contract["host_gate_encoding_required"] = False
    contract["state_host_fallback_allowed"] = True
    contract["convergence_claim_allowed"] = True
    errors = contract_errors(contract)
    assert any("host_gate_encoding_required" in error for error in errors)
    assert any("state_host_fallback_allowed" in error for error in errors)
    assert any("convergence_claim_allowed" in error for error in errors)


def test_split_p3_executable_plan_and_envelope_match_contract() -> None:
    contract = load_toml(CONTRACT)
    plan = split_real_imag_p3_precision_plan().to_dict()
    plan.pop("kind")
    plan.pop("contract_version")
    assert plan == contract["precision_plan"]
    envelope = split_real_imag_p3_accuracy_envelope()
    assert envelope.max_state_infidelity == pytest.approx(
        contract["thresholds"]["max_state_infidelity"]
    )
    assert envelope.max_expectation_abs_error == pytest.approx(
        contract["thresholds"]["max_expectation_absolute_error"]
    )
    assert envelope.max_gradient_rel_error == pytest.approx(
        contract["thresholds"]["max_gradient_relative_error"]
    )
