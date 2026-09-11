from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "check_split_real_imag_contract",
    ROOT / "tools/check_split_real_imag_contract.py",
)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)

pytestmark = pytest.mark.unit


def test_checked_in_split_real_imag_contract_passes() -> None:
    contract = CHECKER._load_toml(CHECKER.CONTRACT)
    assert CHECKER.contract_errors(contract) == ()


def test_contract_rejects_automatic_runtime_selection() -> None:
    contract = CHECKER._load_toml(CHECKER.CONTRACT)
    contract["runtime_default"] = True
    assert (
        "split real/imag contract runtime_default must be False"
        in CHECKER.contract_errors(contract)
    )
