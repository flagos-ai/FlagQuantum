import json
from pathlib import Path


def test_issue100_retained_evidence_contract():
    path = Path("benchmarks/development/issue100_mps_probe_pipeline/8gpu.json")
    if not path.exists():
        return
    payload = json.loads(path.read_text())
    assert payload["distribution_semantics"] == "sharded_across_ranks"
    assert payload["scalability_claim_allowed"] is False
    assert payload["throughput_improvement"] >= 0.20
    assert payload["timeline_audit"]["adjacent_rank_concurrent_useful_work"]
    assert payload["optimizer_update_error"] <= 2e-6
    assert (
        payload["bounded_live_objective_graphs_per_rank"]
        <= payload["max_pipeline_slots"]
    )
    assert len(payload["ranks"]) == 8
