from benchmarks.runners.tn.tn_sliced_reverse_nccl import _audit_evidence_fields
from flagquantum.runtime.audit import audit_distributed_scalability


def _rank(rank: int) -> dict[str, object]:
    return {
        "rank": rank,
        "local_rank": rank,
        "hostname": "node0",
        "local_task_count": 32,
        "task_plan_identity": "fixed-path",
        "cuda_peak_allocated_bytes": 1024,
        "collective_count": 41,
        "collective_payload_bytes_per_rank": 336,
    }


def test_tn_rank_evidence_is_auditable_and_single_rank_is_not_sharded():
    single = {
        "world_size": 1,
        "local_world_size": 1,
        "node_count": 1,
        "scalability_claim_allowed": False,
        **_audit_evidence_fields([_rank(0)], world_size=1),
    }
    distributed = {
        "world_size": 2,
        "local_world_size": 2,
        "node_count": 1,
        "scalability_claim_allowed": False,
        **_audit_evidence_fields([_rank(0), _rank(1)], world_size=2),
    }

    assert single["distribution_semantics"] == "single_device_fast_path"
    assert audit_distributed_scalability(single).valid
    assert distributed["distribution_semantics"] == "sharded_across_ranks"
    assert audit_distributed_scalability(distributed).valid
