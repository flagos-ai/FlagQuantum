"""Fail-closed ISSUE-092 variable-bond capacity artifact validation."""

from __future__ import annotations

from typing import Any, Mapping

ISSUE091_TRUNCATION_BUDGET = 0.1


class MPSCapacityCertificationError(ValueError):
    pass


def require_general_mps_capacity(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != "flagquantum.issue092.general_mps_capacity.v1":
        raise MPSCapacityCertificationError("unexpected ISSUE-092 schema")
    if int(payload.get("batch_size", 0)) != 1:
        raise MPSCapacityCertificationError("capacity workload must use batch size one")
    if not payload.get("single_gpu_capacity_failure"):
        raise MPSCapacityCertificationError("measured single-GPU failure is required")
    if not payload.get("sharded_completion") or int(payload.get("world_size", 0)) != 8:
        raise MPSCapacityCertificationError(
            "measured eight-rank completion is required"
        )
    if payload.get("distribution_semantics") != "sharded_across_ranks":
        raise MPSCapacityCertificationError("capacity state is not rank sharded")
    if payload.get("full_mps_materialization") is not False:
        raise MPSCapacityCertificationError("full MPS materialization is forbidden")
    boundaries = payload.get("boundary_evidence", ())
    if len(boundaries) != 7 or {tuple(item["ranks"]) for item in boundaries} != {
        (rank, rank + 1) for rank in range(7)
    }:
        raise MPSCapacityCertificationError("all seven rank boundaries are required")
    if not all(item["forward"] and item["reverse"] for item in boundaries):
        raise MPSCapacityCertificationError(
            "boundary forward/reverse evidence is incomplete"
        )
    for rank, item in enumerate(boundaries):
        if int(item.get("bond", -1)) <= 0:
            raise MPSCapacityCertificationError("boundary bond identity is missing")
        update = item.get("bond_update", {})
        if tuple(update.get("owner_ranks", ())) != (rank, rank + 1):
            raise MPSCapacityCertificationError(
                "boundary ownership evidence is invalid"
            )
        if (
            update.get("forward_transport") != "batched_isend_irecv"
            or update.get("reverse_transport") != "batched_isend_irecv"
        ):
            raise MPSCapacityCertificationError(
                "boundary transport evidence is invalid"
            )
        if int(update.get("original_rank", 0)) <= int(update.get("kept_rank", 0)):
            raise MPSCapacityCertificationError("boundary did not record truncation")
    if not payload.get("bond_dimension_changed"):
        raise MPSCapacityCertificationError("capacity workload did not change a bond")
    if (
        float(payload.get("truncation_error_budget", float("inf")))
        != ISSUE091_TRUNCATION_BUDGET
    ):
        raise MPSCapacityCertificationError("ISSUE-091 truncation budget is not pinned")
    if (
        float(payload.get("discarded_weight", float("inf")))
        > ISSUE091_TRUNCATION_BUDGET
    ):
        raise MPSCapacityCertificationError("truncation error budget exceeded")
    records = payload.get("rank_records", ())
    if len(records) != 8 or {int(record.get("rank", -1)) for record in records} != set(
        range(8)
    ):
        raise MPSCapacityCertificationError("rank evidence is incomplete")
    if not all(
        record.get("status") == "passed" and record.get("useful_work")
        for record in records
    ):
        raise MPSCapacityCertificationError("every rank must complete useful work")
    if not all(
        int(record.get("peak_memory_bytes", 0)) > 0
        and int(record.get("boundary_bytes", 0)) > 0
        and record.get("cleanup_verified") is True
        for record in records
    ):
        raise MPSCapacityCertificationError(
            "rank memory, communication, or cleanup evidence is incomplete"
        )
    if int(payload.get("single_gpu_peak_memory_bytes", 0)) <= 40_000_000_000:
        raise MPSCapacityCertificationError(
            "single-GPU capacity failure lacks measured memory evidence"
        )
    sources = payload.get("source_artifacts", ())
    required_kinds = {"single_gpu_failure", "workload", "raw_log", "gpu_samples"}
    if {item.get("kind") for item in sources} != required_kinds or any(
        not item.get("path") or len(str(item.get("sha256", ""))) != 64
        for item in sources
    ):
        raise MPSCapacityCertificationError("capacity source provenance is incomplete")
    if not payload.get("topology_fingerprint") or not payload.get("command"):
        raise MPSCapacityCertificationError(
            "capacity workload provenance is incomplete"
        )
    if payload.get("scalability_claim_allowed") or payload.get("release_gate_allowed"):
        raise MPSCapacityCertificationError("ISSUE-092 cannot promote release claims")


__all__ = (
    "ISSUE091_TRUNCATION_BUDGET",
    "MPSCapacityCertificationError",
    "require_general_mps_capacity",
)
