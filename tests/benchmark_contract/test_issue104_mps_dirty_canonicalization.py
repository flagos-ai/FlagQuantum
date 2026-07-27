import json
from pathlib import Path


def test_issue104_retained_evidence_contract():
    path = Path("benchmarks/development/issue104_mps_dirty_canonicalization/8gpu.json")
    if not path.exists():
        return
    p = json.loads(path.read_text())
    assert (
        p["distribution_semantics"] == "sharded_across_ranks"
        and not p["scalability_claim_allowed"]
    )
    assert (
        p["certified_clean_planned_qr_bonds"] == 0
        and p["dirty_planned_qr_bonds"] < p["full_planned_qr_bonds"]
    )
    assert p["factorization_time_reduction"] >= 0.3
    assert max(p["max_value_error"], p["max_gradient_error"]) <= 2e-6
    assert p["approximate_gradients_finite"]
    assert p["exact_discarded_weight"] == 0
    assert p["approximate_discarded_weight"] <= p["approximate_tolerance"]
    assert p["minimum_truncation_gap"] > 1e-7
    assert len(p["ranks"]) == 8
