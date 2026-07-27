import json
from pathlib import Path


def test_issue105_retained_evidence_contract():
    path = Path("benchmarks/development/issue105_dynamic_bond_compile_cache/8gpu.json")
    if not path.exists():
        return
    payload = json.loads(path.read_text())
    assert payload["schema"] == "flagquantum.issue105.dynamic_bond_compile_cache.v1"
    assert payload["distribution_semantics"] == "sharded_across_ranks"
    assert not payload["scalability_claim_allowed"]
    assert payload["world_size"] == len(payload["ranks"]) == 8
    assert payload["workload"]["steps"] == 100
    assert payload["post_warmup_compile_count"] == 0
    assert (
        payload["cache_entries_max"]
        <= payload["workload"]["cache_policy"]["max_entries"]
    )
    assert (
        payload["cache_accounted_input_bytes_max"]
        <= payload["workload"]["cache_policy"]["max_accounted_input_bytes"]
    )
    assert payload["max_correctness_error"] <= 2e-6
    assert payload["warm_regression_passed"]
    assert payload["maximum_observed_warm_ratio"] <= payload["maximum_warm_ratio"]
    assert payload["unsupported_fallbacks"] == 8
    assert payload["fallback_preserved_cache"]
    assert all(len(rank["warm_seconds"]) == 100 for rank in payload["ranks"])
    assert all(
        any(event["event"] == "compile" for event in rank["prewarm_events"])
        for rank in payload["ranks"]
    )
