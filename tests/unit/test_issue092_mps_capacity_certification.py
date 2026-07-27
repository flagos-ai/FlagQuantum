import copy

import pytest

from flagquantum.testing import (
    MPSCapacityCertificationError,
    require_general_mps_capacity,
)

pytestmark = pytest.mark.unit


def payload():
    return {
        "schema": "flagquantum.issue092.general_mps_capacity.v1",
        "batch_size": 1,
        "single_gpu_capacity_failure": True,
        "sharded_completion": True,
        "world_size": 8,
        "distribution_semantics": "sharded_across_ranks",
        "full_mps_materialization": False,
        "boundary_evidence": [
            {
                "bond": 100 + rank,
                "ranks": [rank, rank + 1],
                "forward": True,
                "reverse": True,
                "bond_update": {
                    "owner_ranks": [rank, rank + 1],
                    "original_rank": 512,
                    "kept_rank": 511,
                    "forward_transport": "batched_isend_irecv",
                    "reverse_transport": "batched_isend_irecv",
                },
            }
            for rank in range(7)
        ],
        "bond_dimension_changed": True,
        "discarded_weight": 0.01,
        "truncation_error_budget": 0.1,
        "rank_records": [
            {
                "rank": rank,
                "status": "passed",
                "useful_work": True,
                "peak_memory_bytes": 1,
                "boundary_bytes": 1,
                "cleanup_verified": True,
            }
            for rank in range(8)
        ],
        "single_gpu_peak_memory_bytes": 41_000_000_000,
        "topology_fingerprint": "a" * 64,
        "command": "torchrun benchmark.py",
        "source_artifacts": [
            {"kind": kind, "path": f"{kind}.json", "sha256": "a" * 64}
            for kind in ("single_gpu_failure", "workload", "raw_log", "gpu_samples")
        ],
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
    }


def test_issue092_capacity_contract_accepts_complete_evidence():
    require_general_mps_capacity(payload())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("single_gpu_capacity_failure", False),
        ("full_mps_materialization", True),
        ("bond_dimension_changed", False),
        ("discarded_weight", 0.2),
    ],
)
def test_issue092_capacity_contract_rejects_fault(field, value):
    invalid = copy.deepcopy(payload())
    invalid[field] = value
    with pytest.raises(MPSCapacityCertificationError):
        require_general_mps_capacity(invalid)


def test_issue092_capacity_contract_rejects_idle_rank_and_missing_boundary():
    invalid = payload()
    invalid["rank_records"][7]["useful_work"] = False
    invalid["boundary_evidence"].pop()
    with pytest.raises(MPSCapacityCertificationError):
        require_general_mps_capacity(invalid)
