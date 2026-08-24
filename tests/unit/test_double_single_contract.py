from __future__ import annotations

from pathlib import Path

import pytest

from flagquantum.core.numerics import PrecisionPlanContract, coerce_precision_plan
from tools.check_double_single_contract import contract_errors, load_toml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_double_single_contract_is_current() -> None:
    assert contract_errors(load_toml(ROOT / "double-single-contract.toml")) == ()


def test_contract_rejects_runtime_and_default_selection_claims() -> None:
    contract = load_toml(ROOT / "double-single-contract.toml")
    contract["runtime_integration"] = True
    contract["default_selection_allowed"] = True
    errors = contract_errors(contract)
    assert "Double-Single contract runtime_integration must be False" in errors
    assert "Double-Single contract default_selection_allowed must be False" in errors


def test_local_statevector_still_rejects_double_single_plan() -> None:
    plan = PrecisionPlanContract(
        complex_representation="double_single_fp32",
        parameter_dtype="float32",
        gate_generation_dtype="float32",
        state_storage_dtype="float32",
        kernel_compute_dtype="float32",
        reduction_dtype="float32",
        decomposition_dtype="float32",
        gradient_dtype="float32",
        optimizer_master_dtype="float32",
        communication_dtype="float32",
        checkpoint_dtype="float32",
    )
    with pytest.raises(NotImplementedError, match="cannot honestly execute"):
        coerce_precision_plan(plan, dtype="complex64")
