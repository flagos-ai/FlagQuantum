from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

# The fixture loads a benchmark contract module that imports JAX, so the file
# belongs to the `jax` suites too and skips rather than erroring when JAX is
# absent.
if importlib.util.find_spec("jax") is None:  # pragma: no cover
    pytest.skip("jax is not installed", allow_module_level=True)

# Each test also carries `benchmark_contract`; the module-level marker records
# the JAX dependency the fixture reaches through the benchmark contract module.
pytestmark = pytest.mark.jax


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
