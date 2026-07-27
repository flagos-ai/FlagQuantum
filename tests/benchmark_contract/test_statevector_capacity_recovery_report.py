from copy import deepcopy

import pytest

from benchmarks.statevector_capacity_recovery_report import build_report


def _inputs() -> tuple[dict, dict, dict, dict, dict]:
    workload = {"n_wires": 31}
    baseline = {
        "workload_sha256": "same",
        "ranks": [
            {
                "status": "expected_oom",
                "single_device_oom_observed": True,
                "expectation_met": True,
            }
        ],
    }
    reference_rank = {
        "parameters": [0.3, -0.4],
        "training": {"losses": [0.9, 0.8]},
    }
    uninterrupted = {
        "workload_sha256": "same",
        "ranks": [deepcopy(reference_rank), deepcopy(reference_rank)],
    }
    crash = {
        "status": "failed_closed",
        "reason": "peer_launcher_failure",
        "cleanup_verified": True,
        "all_launchers_exited": True,
        "failure_detected_seconds": 5.0,
        "cleanup_elapsed_seconds": 1.0,
    }
    resume = {
        "status": "passed",
        "reason": "completed",
        "cleanup_verified": True,
        "all_launchers_exited": True,
        "elapsed_seconds": 10.0,
    }
    recovered_ranks = []
    for rank, hostname in enumerate(("node-a", "node-b")):
        recovered_ranks.append(
            {
                "rank": rank,
                "hostname": hostname,
                "status": "passed",
                "resume_requested": True,
                "checkpoint_enabled": True,
                "parameters": [0.3, -0.4],
                "workload": workload,
                "device": {"peak_allocated_bytes": 100},
                "training": {
                    "start_step": 1,
                    "completed_steps": 2,
                    "losses": [0.8],
                    "distribution_semantics": "sharded_across_ranks",
                },
            }
        )
    recovered = {"workload_sha256": "same", "ranks": recovered_ranks}
    return baseline, uninterrupted, crash, resume, recovered


def test_accepts_matched_capacity_recovery() -> None:
    report = build_report(*_inputs())
    assert report["development_capacity_recovery_observed"] is True
    assert report["recovery_claim_allowed"] is False
    assert report["node_count"] == 2


@pytest.mark.parametrize("fault", ["hash", "oom", "cleanup", "parameter", "host"])
def test_capacity_recovery_fails_closed(fault: str) -> None:
    values = list(deepcopy(_inputs()))
    baseline, _, crash, _, recovered = values
    if fault == "hash":
        recovered["workload_sha256"] = "different"
    elif fault == "oom":
        baseline["ranks"][0]["single_device_oom_observed"] = False
    elif fault == "cleanup":
        crash["cleanup_verified"] = False
    elif fault == "parameter":
        recovered["ranks"][1]["parameters"] = [0.3, -0.5]
    else:
        recovered["ranks"][1]["hostname"] = "node-a"
    with pytest.raises(ValueError):
        build_report(*values)
