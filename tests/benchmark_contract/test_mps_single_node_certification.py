import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.benchmark_contract
ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "benchmarks/results/local/mps_single_node_certification.json"


def test_mps_single_node_certification_is_fail_closed() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["schema"] == "flagquantum.mps_single_node_certification.v1"
    assert payload["scope"] == "development_evidence"
    assert payload["hardware"]["accelerator_count"] == 8
    assert all(
        payload["checks"][name] == "passed"
        for name in (
            "two_gpu_accelerator_backward",
            "two_gpu_rank_owned_forward",
            "four_gpu_rank_owned_forward",
            "eight_gpu_rank_owned_forward",
            "eight_gpu_sgd_100_step_soak",
            "checkpoint_restart_equivalence",
            "all_ranks_useful_work",
            "shape_mismatch_bounded_cleanup",
        )
    )
    assert payload["checks"]["maximum_memory_growth_bytes"] == 0
    assert payload["checks"]["maximum_restart_parameter_error"] == 0.0
    assert payload["checks"]["maximum_restart_loss_error"] == 0.0
    assert payload["stability"] == {
        "world_size": 8,
        "optimizer": "sgd",
        "steps": 100,
        "max_bond": 16,
        "gradient_tolerance": 0.1,
        "full_mps_materialization": False,
    }
    claims = payload["claim_policy"]
    assert not claims["production_supported"]
    assert not claims["release_certified"]
    assert not claims["multi_node_claim_allowed"]
    assert not claims["scalability_claim_allowed"]
