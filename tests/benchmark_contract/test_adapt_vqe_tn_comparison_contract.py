from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture(scope="module")
def tn_contract() -> ModuleType:
    """Load numerical benchmark dependencies only when this tier is selected."""
    path = Path(__file__).parents[2] / "benchmarks/runners/tn/adapt_vqe_tn_contract.py"
    spec = importlib.util.spec_from_file_location("adapt_vqe_tn_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.benchmark_contract
def test_48q_mean_field_initial_state_beats_zero_product_state(
    tn_contract: ModuleType,
) -> None:
    angles = tn_contract.mean_field_angles(6, 8)
    energy = tn_contract.product_state_energy(6, 8, angles)

    assert len(angles) == 48
    assert energy < -82.0


@pytest.mark.benchmark_contract
def test_mean_field_initial_state_is_deterministic(tn_contract: ModuleType) -> None:
    assert tn_contract.mean_field_angles(2, 3) == tn_contract.mean_field_angles(2, 3)
