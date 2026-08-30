from __future__ import annotations

import copy
from pathlib import Path

import pytest

from flagquantum.runtime.backends.statevector.split_real_imag_precision import (
    split_real_imag_p2_accuracy_envelope,
    split_real_imag_p2_precision_plan,
)
from tools.check_split_real_imag_p2_precision_contract import (
    contract_errors,
    load_toml,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_checked_in_split_p2_precision_contract_passes() -> None:
    contract = load_toml(
        ROOT / "contracts" / "split-real-imag-statevector-p2-precision-contract.toml"
    )
    assert contract_errors(contract) == ()


def test_split_p2_precision_contract_rejects_claim_promotion() -> None:
    contract = copy.deepcopy(
        load_toml(
            ROOT
            / "contracts"
            / "split-real-imag-statevector-p2-precision-contract.toml"
        )
    )
    contract["runtime_default"] = True
    contract["full_state_double_single_claim_allowed"] = True
    contract["convergence_claim_allowed"] = True
    errors = contract_errors(contract)
    assert any("runtime_default" in error for error in errors)
    assert any("full_state_double_single_claim_allowed" in error for error in errors)
    assert any("convergence_claim_allowed" in error for error in errors)


def test_split_p2_executable_plan_and_accuracy_match_machine_contract() -> None:
    contract = load_toml(
        ROOT / "contracts" / "split-real-imag-statevector-p2-precision-contract.toml"
    )
    plan = split_real_imag_p2_precision_plan().to_dict()
    plan.pop("kind")
    plan.pop("contract_version")
    assert plan == contract["precision_plan"]

    envelope = split_real_imag_p2_accuracy_envelope()
    assert envelope.max_expectation_abs_error == pytest.approx(
        contract["thresholds"]["max_expectation_absolute_error"]
    )
    assert envelope.max_gradient_rel_error == pytest.approx(
        contract["thresholds"]["max_gradient_relative_error"]
    )
