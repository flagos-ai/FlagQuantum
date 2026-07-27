import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.benchmark_contract

RESULT = Path("benchmarks/results/local/jax_jit_crossover_a800.json")


def test_jax_jit_crossover_result_is_honest_and_complete():
    if not RESULT.exists():
        pytest.skip("local JAX JIT benchmark has not been collected")
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    assert payload["schema"] == "flagquantum.jax_jit_crossover.v1"
    assert payload["distribution_semantics"] == "single_device_fast_path"
    assert payload["scalability_claim_allowed"] is False
    assert payload["methodology"]["cold_process_isolation"] is True
    assert payload["methodology"]["persistent_compilation_cache_disabled"] is True
    assert payload["correctness_passed"] is True
    assert len(payload["measurements"]) >= 3
    for row in payload["measurements"]:
        assert row["jax_cold_total_seconds_median"] > 0
        assert row["jax_steady_seconds_median"] > 0
        assert row["pytorch_steady_seconds_median"] > 0
        assert row["loss_abs_error_max"] <= 1e-4
        assert row["grad_max_abs_error_max"] <= 1e-4
