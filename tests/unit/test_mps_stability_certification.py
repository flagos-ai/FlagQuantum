import copy

import pytest

from flagquantum.testing import MPSStabilityCertificationError, require_mps_stability

pytestmark = pytest.mark.unit


def payload():
    timeline = [
        {
            "step": step,
            "allocated": 100,
            "reserved": 200,
            "peak": 220,
            "tape": 10,
            "optimizer": 10,
            "communication": 10,
        }
        for step in range(100)
    ]

    def soak(optimizer):
        return {
            "optimizer": optimizer,
            "world_size": 8,
            "steps": 100,
            "warmup_steps": 5,
            "memory_growth_tolerance_bytes": 0,
            "restart_generation": 5,
            "rank_records": [
                {
                    "rank": rank,
                    "memory_timeline": copy.deepcopy(timeline),
                    "memory_growth_bytes": 0,
                    "rank_useful_work": True,
                    "restart_parameter_error": 0.0,
                    "restart_loss_error": 0.0,
                    "restart_start_step": 5,
                    "checkpoint_contract_fingerprint": "f" * 64,
                }
                for rank in range(8)
            ],
        }

    fault_names = (
        "topology_mismatch",
        "generation_mismatch",
        "interrupted_checkpoint",
        "rank_exception",
        "cuda_oom",
        "collective_timeout",
    )
    return {
        "schema": "flagquantum.issue093.mps_stability_matrix.v2",
        "soaks": [soak("adam"), soak("sgd")],
        "capacity_multistep": {
            "steps": 3,
            "rank_records": [
                {"rank": rank, "memory_timeline": [{}, {}, {}]} for rank in range(8)
            ],
        },
        "fault_matrix": [
            {
                "fault": name,
                "passed": True,
                "cleanup_verified": True,
                "exit_code": 1,
                "elapsed_seconds": 1.0,
                "timeout_seconds": 10.0,
                "log_sha256": "a" * 64,
            }
            for name in fault_names
        ],
        "source_artifacts": [
            {"kind": kind, "path": kind, "sha256": "b" * 64}
            for kind in (
                "adam_soak",
                "sgd_soak",
                "capacity_multistep",
                "fault_matrix",
                "workload",
            )
        ],
        "command": "aggregate issue093",
        "commit": "c" * 40,
        "full_mps_materialization": False,
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
    }


def test_v2_accepts_complete_evidence():
    require_mps_stability(payload())


@pytest.mark.parametrize("fault", ("timeline", "generation", "timeout", "source"))
def test_v2_rejects_invalid_evidence(fault):
    invalid = payload()
    if fault == "timeline":
        invalid["soaks"][0]["rank_records"][0]["memory_timeline"][-1]["allocated"] = 101
    elif fault == "generation":
        invalid["soaks"][0]["rank_records"][0]["restart_start_step"] = 4
    elif fault == "timeout":
        invalid["fault_matrix"][0]["elapsed_seconds"] = 11
    else:
        invalid["source_artifacts"].pop()
    with pytest.raises(MPSStabilityCertificationError):
        require_mps_stability(invalid)
