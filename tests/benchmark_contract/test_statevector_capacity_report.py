from copy import deepcopy

import pytest

from benchmarks.statevector_capacity_report import build_report


def _rank(rank: int, world: int, *, oom: bool) -> dict:
    workload = {"steps": 2, "n_wires": 31}
    return {
        "schema": "flagquantum.single_node_training_acceptance.v2",
        "rank": rank,
        "local_rank": 0,
        "hostname": f"node-{rank}" if world > 1 else "node-0",
        "world_size": world,
        "status": "expected_oom" if oom else "passed",
        "expectation_met": True,
        "single_device_oom_observed": oom,
        "workload": workload,
        "device": {
            "peak_allocated_bytes": 10,
            "peak_reserved_bytes": 12,
            "total_memory_bytes": 16,
        },
        "training": None
        if oom
        else {
            "distribution_semantics": "sharded_across_ranks",
            "completed_steps": 2,
            "losses": [0.9, 0.8],
            "communication_bytes": 100,
            "node_count": world,
        },
        "parameters": [0.1, -0.2],
        "oom_type": "OutOfMemoryError" if oom else None,
    }


def _documents() -> tuple[dict, dict]:
    return (
        {"workload_sha256": "same", "ranks": [_rank(0, 1, oom=True)]},
        {
            "workload_sha256": "same",
            "ranks": [_rank(0, 2, oom=False), _rank(1, 2, oom=False)],
        },
    )


def test_matched_capacity_pair_is_development_only() -> None:
    report = build_report(*_documents())
    assert report["development_capacity_expansion_observed"] is True
    assert report["capacity_claim_allowed"] is False
    assert report["distributed"]["world_size"] == 2
    assert report["distributed"]["node_count"] == 2
    assert report["multi_node_capacity_completion_observed"] is True
    assert "multi_node_capacity_completion_not_measured" not in report["blockers"]


@pytest.mark.parametrize("fault", ["hash", "baseline", "rank", "loss"])
def test_capacity_report_fails_closed(fault: str) -> None:
    baseline, distributed = deepcopy(_documents())
    if fault == "hash":
        distributed["workload_sha256"] = "different"
    elif fault == "baseline":
        baseline["ranks"][0]["status"] = "passed"
    elif fault == "rank":
        distributed["ranks"][1]["rank"] = 0
    else:
        distributed["ranks"][1]["training"]["losses"] = [0.9, 0.7]
    with pytest.raises(ValueError):
        build_report(baseline, distributed)
