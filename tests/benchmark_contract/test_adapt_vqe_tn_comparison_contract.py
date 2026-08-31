from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).parents[2] / "benchmarks/runners/tn/adapt_vqe_tn_contract.py"
SPEC = importlib.util.spec_from_file_location("adapt_vqe_tn_contract", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.benchmark_contract
def test_48q_mean_field_initial_state_beats_zero_product_state() -> None:
    angles = MODULE.mean_field_angles(6, 8)
    energy = MODULE.product_state_energy(6, 8, angles)

    assert len(angles) == 48
    assert energy < -82.0


@pytest.mark.benchmark_contract
def test_mean_field_initial_state_is_deterministic() -> None:
    assert MODULE.mean_field_angles(2, 3) == MODULE.mean_field_angles(2, 3)
