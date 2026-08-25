from __future__ import annotations

from pathlib import Path

import pytest

from tools.check_split_real_imag_p4_device_double_single_contract import (
    contract_errors,
    load_toml,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "split-real-imag-statevector-p4-device-double-single-contract.toml"


def test_checked_in_split_p4_contract_passes() -> None:
    assert contract_errors(load_toml(CONTRACT)) == ()


def test_split_p4_contract_rejects_host_gate_encoding() -> None:
    contract = load_toml(CONTRACT)
    contract["host_gate_encoding_allowed"] = True
    assert "host_gate_encoding_allowed" in " ".join(contract_errors(contract))


def test_split_p4_contract_rejects_claim_promotion() -> None:
    contract = load_toml(CONTRACT)
    contract["convergence_claim_allowed"] = True
    assert "convergence_claim_allowed" in " ".join(contract_errors(contract))
