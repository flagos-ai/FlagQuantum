"""Fail-closed ISSUE-092 variable-bond capacity artifact validation."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

ISSUE091_TRUNCATION_BUDGET = 0.1


def finalize_capacity_source_integrity(
    payload: Mapping[str, Any], *, base_dir: Path
) -> dict[str, Any]:
    """Seal capacity provenance after logs and telemetry have been closed."""
    finalized = deepcopy(dict(payload))
    sources = finalized.get("source_artifacts", ())
    for source in sources:
        path = Path(str(source.get("path", "")))
        resolved = path if path.is_absolute() else base_dir / path
        if not resolved.is_file():
            raise MPSCapacityCertificationError(
                f"capacity source artifact is missing: {resolved}"
            )
        source["sha256"] = hashlib.sha256(resolved.read_bytes()).hexdigest()
    finalized["source_integrity_finalized"] = True
    return finalized


def require_capacity_source_integrity(
    payload: Mapping[str, Any], *, base_dir: Path
) -> None:
    """Verify finalized provenance against immutable bytes on disk."""
    if payload.get("source_integrity_finalized") is not True:
        raise MPSCapacityCertificationError("capacity sources are not finalized")
    expected = finalize_capacity_source_integrity(payload, base_dir=base_dir)
    for recorded, actual in zip(
        payload.get("source_artifacts", ()),
        expected.get("source_artifacts", ()),
        strict=True,
    ):
        if recorded.get("sha256") != actual.get("sha256"):
            raise MPSCapacityCertificationError(
                f"capacity source integrity mismatch: {recorded.get('path')}"
            )


class MPSCapacityCertificationError(ValueError):
    pass


def require_general_mps_capacity(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != "flagquantum.issue092.general_mps_capacity.v1":
        raise MPSCapacityCertificationError("unexpected ISSUE-092 schema")
    if int(payload.get("batch_size", 0)) != 1:
        raise MPSCapacityCertificationError("capacity workload must use batch size one")
    if not payload.get("single_gpu_capacity_failure"):
        raise MPSCapacityCertificationError("measured single-GPU failure is required")
    world_size = int(payload.get("world_size", 0))
    if not payload.get("sharded_completion") or world_size not in {8, 16}:
        raise MPSCapacityCertificationError(
            "measured eight- or sixteen-rank completion is required"
        )
    if payload.get("distribution_semantics") != "sharded_across_ranks":
        raise MPSCapacityCertificationError("capacity state is not rank sharded")
    if payload.get("full_mps_materialization") is not False:
        raise MPSCapacityCertificationError("full MPS materialization is forbidden")
    boundaries = payload.get("boundary_evidence", ())
    if len(boundaries) != world_size - 1 or {
        tuple(item["ranks"]) for item in boundaries
    } != {
        (rank, rank + 1) for rank in range(world_size - 1)
    }:
        raise MPSCapacityCertificationError(
            "every adjacent rank boundary is required"
        )
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
        if int(update.get("original_rank", 0)) < int(update.get("kept_rank", 0)):
            raise MPSCapacityCertificationError("boundary increased its bond rank")
    gradient_policy = payload.get("gradient_policy", "approximate")
    exact_capacity = gradient_policy == "exact"
    if not exact_capacity and not payload.get("bond_dimension_changed"):
        raise MPSCapacityCertificationError("approximate capacity workload did not change a bond")
    expected_budget = 0.0 if exact_capacity else ISSUE091_TRUNCATION_BUDGET
    if float(payload.get("truncation_error_budget", float("inf"))) != expected_budget:
        raise MPSCapacityCertificationError("capacity truncation budget is not pinned")
    if (
        float(payload.get("discarded_weight", float("inf")))
        > expected_budget
    ):
        raise MPSCapacityCertificationError("truncation error budget exceeded")
    records = payload.get("rank_records", ())
    if len(records) != world_size or {
        int(record.get("rank", -1)) for record in records
    } != set(range(world_size)):
        raise MPSCapacityCertificationError("rank evidence is incomplete")
    if not all(
        record.get("status") == "passed" and record.get("useful_work")
        for record in records
    ):
        raise MPSCapacityCertificationError("every rank must complete useful work")
    missing_memory = [
        int(record["rank"])
        for record in records
        if int(record.get("peak_memory_bytes", 0)) <= 0
    ]
    if missing_memory:
        raise MPSCapacityCertificationError(
            f"rank memory evidence is incomplete: ranks={missing_memory}"
        )
    missing_communication = [
        int(record["rank"])
        for record in records
        if int(record.get("boundary_bytes", 0)) <= 0
    ]
    if missing_communication:
        raise MPSCapacityCertificationError(
            "rank communication evidence is incomplete: "
            f"ranks={missing_communication}"
        )
    failed_cleanup = [
        {
            "rank": int(record["rank"]),
            "allocated": int(record.get("cleanup_allocated_bytes", -1)),
            "limit": int(record.get("cleanup_limit_bytes", -1)),
        }
        for record in records
        if record.get("cleanup_verified") is not True
    ]
    if failed_cleanup:
        raise MPSCapacityCertificationError(
            f"rank cleanup evidence is incomplete: {failed_cleanup}"
        )
    peak = int(payload.get("single_gpu_peak_memory_bytes", 0))
    device_total = int(payload.get("single_gpu_device_total_memory_bytes", 0))
    minimum_peak = device_total // 2 if device_total > 0 else 40_000_000_000
    if peak <= minimum_peak:
        raise MPSCapacityCertificationError(
            "single-GPU capacity failure lacks measured memory evidence"
        )
    sources = payload.get("source_artifacts", ())
    required_kinds = {"single_gpu_failure", "workload", "raw_log", "gpu_samples"}
    source_kinds = [item.get("kind") for item in sources]
    if not required_kinds.issubset(source_kinds) or len(source_kinds) != len(
        set(source_kinds)
    ) or any(
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
