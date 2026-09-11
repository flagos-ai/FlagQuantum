import pytest

from benchmarks.compiler_routing_cache import run_benchmark

pytestmark = pytest.mark.benchmark_contract


def test_routing_cache_benchmark_reports_deterministic_cache_evidence() -> None:
    payload = run_benchmark(
        n_wires=24,
        gate_count=240,
        path_cache_capacity=64,
    )

    assert payload["schema"] == ("flagquantum_compiler_routing_cache_benchmark_v1")
    assert payload["artifact_classification"] == "local_compiler_microbenchmark"
    assert payload["distribution_semantics"] == "single_device_fast_path"
    assert payload["scalability_claim_allowed"] is False
    assert payload["cold_path_cache_delta"]["misses"] > 0
    assert payload["warm_path_cache_delta"]["misses"] == 0
    assert payload["warm_path_cache_delta"]["hits"] == payload["gate_count"]
    assert payload["cold_instruction_count"] == payload["warm_instruction_count"]
    assert payload["estimate_matches_materialized"] is True
    assert (
        payload["routing_estimate"]["estimated_instruction_count"]
        == payload["cold_instruction_count"]
    )
    assert payload["planning_seconds"] > 0
    assert payload["cold_seconds"] > 0
    assert payload["warm_seconds"] > 0
