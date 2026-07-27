import json
from pathlib import Path


def test_issue103_retained_evidence_contract():
    path = Path("benchmarks/development/issue103_mps_tensor_metadata/8gpu.json")
    if not path.exists():
        return
    p = json.loads(path.read_text())
    assert (
        p["distribution_semantics"] == "sharded_across_ranks"
        and not p["scalability_claim_allowed"]
    )
    assert p["broadcast_object_list_calls"] == 0
    assert p["object_collective_trace_events"] == 0
    assert p["cached_descriptor_messages"] < p["baseline_descriptor_messages"]
    assert p["cached_static_payload_messages"] > 0
    assert max(p["max_value_error"], p["max_gradient_error"]) <= 2e-6
    assert len(p["ranks"]) == 8 and all(Path(r["trace"]).exists() for r in p["ranks"])
