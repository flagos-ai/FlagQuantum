import json
from pathlib import Path


def test_issue101_retained_evidence_contract():
    path = Path("benchmarks/development/issue101_mps_fused_reverse/8gpu.json")
    if not path.exists():
        return
    payload = json.loads(path.read_text())
    assert payload["distribution_semantics"] == "sharded_across_ranks"
    assert payload["scalability_claim_allowed"] is False
    assert payload["autograd_invocation_reduction"] >= 0.5
    assert payload["python_dispatch_reduction"] >= 0.5
    assert payload["reverse_time_improvement"] >= 0.2
    assert payload["peak_live_memory_growth"] <= 0.1
    assert payload["max_gradient_error"] <= 2e-6
