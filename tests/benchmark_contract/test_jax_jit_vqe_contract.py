import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.benchmark_contract

RESULTS = (
    Path("benchmarks/results/local/jax_jit_vqe_multidepth_a800.json"),
    Path("benchmarks/results/local/jax_jit_vqe_20q_a800.json"),
)


def test_jax_jit_vqe_grid_is_correct_and_single_device_only():
    if not all(path.exists() for path in RESULTS):
        pytest.skip("multi-depth TFIM VQE benchmark has not been collected")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in RESULTS]
    payload = payloads[0]
    assert payload["schema"] == "flagquantum.jax_jit_crossover.v1"
    assert payload["workload"]["task"] == "tfim_vqe"
    assert payload["workload"]["observable"] == "transverse_field_ising_energy"
    assert payload["distribution_semantics"] == "single_device_fast_path"
    assert payload["scalability_claim_allowed"] is False
    assert payload["correctness_passed"] is True
    rows = [row for item in payloads for row in item["measurements"]]
    assert len({row["layers"] for row in rows}) >= 3
    assert len({row["n_wires"] for row in rows}) >= 5
    for item in payloads:
        assert item["correctness_passed"] is True
        assert item["distribution_semantics"] == "single_device_fast_path"
    for row in rows:
        assert row["loss_abs_error_max"] <= 1e-4
        assert row["grad_max_abs_error_max"] <= 1e-4
        assert row["jax_cold_total_seconds_median"] > 0
        assert row["jax_steady_seconds_median"] > 0
