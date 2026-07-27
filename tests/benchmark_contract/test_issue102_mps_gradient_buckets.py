import json
from pathlib import Path


def test_issue102_retained_evidence_contract():
    path = Path("benchmarks/development/issue102_mps_gradient_buckets/8gpu.json")
    if not path.exists():
        return
    p = json.loads(path.read_text())
    assert (
        p["distribution_semantics"] == "sharded_across_ranks"
        and not p["scalability_claim_allowed"]
    )
    assert p["parameter_count"] >= 1000
    assert p["collective_count_reduction"] >= 0.9
    assert p["bucket_collective_count"] <= p["owner_bucket_count"]
    assert p["nonowner_materialized_gradients"] == 0
    assert (
        max(
            p["loss_error"],
            p["max_gradient_error"],
            p["max_sgd_update_error"],
            p["max_adam_update_error"],
        )
        <= 2e-6
    )
    assert len(p["ranks"]) == 8 and all(Path(r["trace"]).exists() for r in p["ranks"])
