from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools" / "check_split_real_imag_p1_contract.py"
SPEC = importlib.util.spec_from_file_location(
    "split_real_imag_p1_contract", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_split_real_imag_p1_contract_passes() -> None:
    contract = MODULE._load_toml(MODULE.CONTRACT)
    assert MODULE.contract_errors(contract) == ()


def test_split_real_imag_p1_contract_rejects_claim_promotion() -> None:
    contract = MODULE._load_toml(MODULE.CONTRACT)
    contract["hardware_certification"] = True
    contract["native_autograd_claim_allowed"] = True
    errors = MODULE.contract_errors(contract)
    assert any("hardware_certification" in error for error in errors)
    assert any("native_autograd_claim_allowed" in error for error in errors)
