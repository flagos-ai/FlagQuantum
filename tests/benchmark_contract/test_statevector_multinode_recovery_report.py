from copy import deepcopy

import pytest

from benchmarks.statevector_multinode_recovery_report import build_report


def _watchdogs() -> tuple[dict, dict]:
    return (
        {
            "schema": "flagquantum.multinode_watchdog.v1",
            "status": "failed_closed",
            "reason": "peer_launcher_failure",
            "cleanup_verified": True,
            "all_launchers_exited": True,
            "cleanup_failures": [],
            "failure_detected_seconds": 10.0,
            "cleanup_elapsed_seconds": 1.0,
        },
        {
            "schema": "flagquantum.multinode_watchdog.v1",
            "status": "passed",
            "reason": "completed",
            "cleanup_verified": True,
            "all_launchers_exited": True,
            "elapsed_seconds": 2.0,
        },
    )


def _rank(rank: int) -> dict:
    ownership = [{"parameter": f"parameter:{rank}", "owner_rank": rank}]
    return {
        "cleanup_verified": True,
        "resumed_parameters": [0.1, -0.2],
        "reference_parameters": [0.1, -0.2],
        "reference_tail_losses": [0.8, 0.7],
        "summary": {
            "rank": rank,
            "world_size": 2,
            "node_count": 2,
            "start_step": 2,
            "completed_steps": 4,
            "losses": [0.8, 0.7],
            "distribution_semantics": "sharded_across_ranks",
            "optimizer_ownership": ownership,
        },
    }


def test_accepts_exact_multinode_recovery_campaign() -> None:
    report = build_report(*_watchdogs(), [_rank(0), _rank(1)])
    assert report["development_multinode_recovery_observed"] is True
    assert report["recovery_claim_allowed"] is False
    assert report["checkpoint_generation"] == 2


@pytest.mark.parametrize("fault", ["cleanup", "resume", "parameter", "rank"])
def test_recovery_report_fails_closed(fault: str) -> None:
    crash, resume = deepcopy(_watchdogs())
    ranks = [_rank(0), _rank(1)]
    if fault == "cleanup":
        crash["cleanup_verified"] = False
    elif fault == "resume":
        resume["status"] = "failed_closed"
    elif fault == "parameter":
        ranks[1]["resumed_parameters"] = [0.1, -0.3]
    else:
        ranks[1]["summary"]["rank"] = 0
    with pytest.raises(ValueError):
        build_report(crash, resume, ranks)
