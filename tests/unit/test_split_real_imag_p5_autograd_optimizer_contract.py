from __future__ import annotations

from pathlib import Path

import pytest

from tools.check_split_real_imag_p5_autograd_optimizer_contract import (
    contract_errors,
    load_toml,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
CONTRACT = (
    ROOT
    / "contracts"
    / "split-real-imag-statevector-p5-autograd-optimizer-contract.toml"
)


def test_checked_in_split_p5_contract_passes() -> None:
    assert contract_errors(load_toml(CONTRACT)) == ()


def test_split_p5_contract_rejects_missing_optimizer_implementation_claim() -> None:
    contract = load_toml(CONTRACT)
    contract["optimizer_available"] = False
    assert "optimizer_available" in " ".join(contract_errors(contract))


def test_split_p5_contract_rejects_an_unnamed_difference_authority() -> None:
    """The diagnostic's authority is a recorded fact, not a convention.

    Without the key the gate cannot tell a shared quotient from a local one, so
    dropping it has to fail rather than silently stop checking.
    """

    contract = load_toml(CONTRACT)
    del contract["verification"]["difference_implementation"]
    assert "one finite-difference authority" in " ".join(contract_errors(contract))


def test_split_p5_contract_rejects_a_diagnostic_call_site_that_is_gone() -> None:
    contract = load_toml(CONTRACT)
    contract["verification"]["difference_conformance"] = "flagquantum/core/ir.py"
    assert "reuse the shared quotient" in " ".join(contract_errors(contract))


def test_split_p5_contract_rejects_single_word_double_single_gradient_claim() -> None:
    contract = load_toml(CONTRACT)
    precision = contract["precision_boundary"]
    precision["single_tensor_grad_double_single_label_allowed"] = True
    assert "precision boundary" in " ".join(contract_errors(contract))


def test_split_p5_contract_rejects_standard_optimizer_equivalence() -> None:
    contract = load_toml(CONTRACT)
    optimizer = contract["precision_optimizer"]
    optimizer["standard_torch_optimizer_compatibility_claim_allowed"] = True
    assert "standard torch optimizer" in " ".join(contract_errors(contract))
