import copy
import hashlib

import pytest

from flagquantum.testing import (
    MPSCapacityCertificationError,
    finalize_capacity_source_integrity,
    require_capacity_source_integrity,
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
        "single_gpu_device_total_memory_bytes": 80_000_000_000,
        "topology_fingerprint": "a" * 64,
        "command": "torchrun benchmark.py",
        "source_artifacts": [
            {"kind": kind, "path": f"{kind}.json", "sha256": "a" * 64}
            for kind in ("single_gpu_failure", "workload", "raw_log", "gpu_samples")
        ],
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
    }


def test_capacity_contract_accepts_complete_evidence():
    require_general_mps_capacity(payload())


def test_capacity_sources_are_finalized_and_verified(tmp_path):
    value = payload()
    for index, source in enumerate(value["source_artifacts"]):
        path = tmp_path / f"source-{index}.bin"
        path.write_bytes(f"evidence-{index}".encode())
        source["path"] = path.name
    finalized = finalize_capacity_source_integrity(value, base_dir=tmp_path)
    require_capacity_source_integrity(finalized, base_dir=tmp_path)
    assert finalized["source_integrity_finalized"] is True
    assert (
        finalized["source_artifacts"][0]["sha256"]
        == hashlib.sha256((tmp_path / "source-0.bin").read_bytes()).hexdigest()
    )
    (tmp_path / "source-0.bin").write_bytes(b"tampered")
    with pytest.raises(MPSCapacityCertificationError, match="integrity mismatch"):
        require_capacity_source_integrity(finalized, base_dir=tmp_path)


def test_capacity_contract_accepts_exact_untruncated_capacity():
    value = payload()
    value["gradient_policy"] = "exact"
    value["bond_dimension_changed"] = False
    value["discarded_weight"] = 0.0
    value["truncation_error_budget"] = 0.0
    for boundary in value["boundary_evidence"]:
        boundary["bond_update"]["kept_rank"] = boundary["bond_update"]["original_rank"]
    require_general_mps_capacity(value)


def test_capacity_contract_accepts_complete_sixteen_rank_evidence():
    value = payload()
    value["world_size"] = 16
    value["boundary_evidence"] = [
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
        for rank in range(15)
    ]
    value["rank_records"] = [
        {
            "rank": rank,
            "status": "passed",
            "useful_work": True,
            "peak_memory_bytes": 1,
            "boundary_bytes": 1,
            "cleanup_verified": True,
        }
        for rank in range(16)
    ]
    require_general_mps_capacity(value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("single_gpu_capacity_failure", False),
        ("full_mps_materialization", True),
        ("bond_dimension_changed", False),
        ("discarded_weight", 0.2),
    ],
)
def test_capacity_contract_rejects_fault(field, value):
    invalid = copy.deepcopy(payload())
    invalid[field] = value
    with pytest.raises(MPSCapacityCertificationError):
        require_general_mps_capacity(invalid)


def test_capacity_contract_rejects_idle_rank_and_missing_boundary():
    invalid = payload()
    invalid["rank_records"][7]["useful_work"] = False
    invalid["boundary_evidence"].pop()
    with pytest.raises(MPSCapacityCertificationError):
        require_general_mps_capacity(invalid)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("peak_memory_bytes", 0, "memory evidence.*ranks=\\[3\\]"),
        ("boundary_bytes", 0, "communication evidence.*ranks=\\[3\\]"),
        ("cleanup_verified", False, "cleanup evidence.*'rank': 3"),
    ],
)
def test_capacity_contract_diagnoses_rank_evidence(field, value, message):
    invalid = payload()
    invalid["rank_records"][3][field] = value
    with pytest.raises(MPSCapacityCertificationError, match=message):
        require_general_mps_capacity(invalid)
