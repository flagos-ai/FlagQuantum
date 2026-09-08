"""Fail-closed calibration evidence for distributed TN memory plans."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from math import prod
from typing import Any, Mapping, Sequence

from .joint_planning import DistributedTNJointPlan

TN_MEMORY_EVIDENCE_VERSION = "flagquantum.distributed_tn_memory_evidence.v1"


@dataclass(frozen=True)
class DistributedTNRankMemoryMeasurement:
    """Measured memory high-water marks from one distributed rank."""

    rank: int
    cuda_peak_allocated_bytes: int
    cuda_peak_reserved_bytes: int
    peak_cached_forward_bytes: int
    rematerialization_peak_transient_bytes: int


@dataclass(frozen=True)
class DistributedTNMemoryEvidence:
    """Auditable comparison between a joint plan and rank measurements."""

    version: str
    identity: str
    joint_plan_identity: str
    world_size: int
    rank_measurements: tuple[DistributedTNRankMemoryMeasurement, ...]
    predicted_working_set_local_bytes: int
    max_cuda_peak_allocated_bytes: int
    max_cuda_peak_reserved_bytes: int
    max_peak_cached_forward_bytes: int
    max_rematerialization_peak_transient_bytes: int
    allocated_to_predicted_ratio: float
    reserved_to_predicted_ratio: float
    max_underprediction_ratio: float
    prediction_calibrated: bool
    memory_budget_satisfied: bool
    passed: bool
    blockers: tuple[str, ...]

    def summary(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "distribution_semantics": "sharded_across_ranks",
            "claim_evidence_type": "development_memory_calibration",
            "scalability_claim_allowed": False,
        }


def build_distributed_tn_memory_evidence(
    plan: DistributedTNJointPlan,
    rank_measurements: Sequence[DistributedTNRankMemoryMeasurement | Mapping[str, int]],
    *,
    max_underprediction_ratio: float = 1.25,
) -> DistributedTNMemoryEvidence:
    """Compare measured rank peaks with a versioned joint memory plan."""

    if max_underprediction_ratio < 1.0:
        raise ValueError("TN memory underprediction ratio must be at least one")
    measurements = tuple(
        (
            item
            if isinstance(item, DistributedTNRankMemoryMeasurement)
            else DistributedTNRankMemoryMeasurement(
                rank=int(item["rank"]),
                cuda_peak_allocated_bytes=int(item["cuda_peak_allocated_bytes"]),
                cuda_peak_reserved_bytes=int(item["cuda_peak_reserved_bytes"]),
                peak_cached_forward_bytes=int(item["peak_cached_forward_bytes"]),
                rematerialization_peak_transient_bytes=int(
                    item["rematerialization_peak_transient_bytes"]
                ),
            )
        )
        for item in rank_measurements
    )
    world_size = prod(plan.mesh_shape)
    if len(measurements) != world_size:
        raise ValueError("TN memory evidence must contain every rank")
    ranks = tuple(measurement.rank for measurement in measurements)
    if tuple(sorted(ranks)) != tuple(range(world_size)):
        raise ValueError("TN memory evidence ranks must be unique and contiguous")
    if any(
        value < 0
        for measurement in measurements
        for value in (
            measurement.cuda_peak_allocated_bytes,
            measurement.cuda_peak_reserved_bytes,
            measurement.peak_cached_forward_bytes,
            measurement.rematerialization_peak_transient_bytes,
        )
    ):
        raise ValueError("TN memory measurements must be non-negative")
    if any(measurement.cuda_peak_allocated_bytes == 0 for measurement in measurements):
        raise ValueError("TN allocated memory measurements must be positive")
    if any(
        measurement.cuda_peak_reserved_bytes < measurement.cuda_peak_allocated_bytes
        for measurement in measurements
    ):
        raise ValueError("TN reserved memory cannot be below allocated memory")

    ordered = tuple(sorted(measurements, key=lambda item: item.rank))
    max_allocated = max(item.cuda_peak_allocated_bytes for item in ordered)
    max_reserved = max(item.cuda_peak_reserved_bytes for item in ordered)
    max_forward_cache = max(item.peak_cached_forward_bytes for item in ordered)
    max_rematerialization = max(
        item.rematerialization_peak_transient_bytes for item in ordered
    )
    predicted = plan.predicted_working_set_local_bytes
    allocated_ratio = max_allocated / max(1, predicted)
    reserved_ratio = max_reserved / max(1, predicted)
    prediction_calibrated = reserved_ratio <= max_underprediction_ratio
    budget_satisfied = (
        plan.memory_budget_local_bytes is not None
        and max_reserved <= plan.memory_budget_local_bytes
    )
    blockers = []
    if plan.memory_budget_local_bytes is None:
        blockers.append("joint_plan_memory_budget_missing")
    if not prediction_calibrated:
        blockers.append("measured_allocator_reservation_exceeds_prediction_tolerance")
    if not budget_satisfied:
        blockers.append("measured_allocator_reservation_exceeds_memory_budget")
    payload = {
        "version": TN_MEMORY_EVIDENCE_VERSION,
        "joint_plan_identity": plan.identity,
        "world_size": world_size,
        "rank_measurements": tuple(asdict(item) for item in ordered),
        "predicted_working_set_local_bytes": predicted,
        "max_cuda_peak_allocated_bytes": max_allocated,
        "max_cuda_peak_reserved_bytes": max_reserved,
        "max_peak_cached_forward_bytes": max_forward_cache,
        "max_rematerialization_peak_transient_bytes": max_rematerialization,
        "allocated_to_predicted_ratio": allocated_ratio,
        "reserved_to_predicted_ratio": reserved_ratio,
        "max_underprediction_ratio": max_underprediction_ratio,
        "prediction_calibrated": prediction_calibrated,
        "memory_budget_satisfied": budget_satisfied,
        "passed": not blockers,
        "blockers": tuple(blockers),
    }
    return DistributedTNMemoryEvidence(
        **payload,
        identity=hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    )


def require_distributed_tn_memory_evidence(
    evidence: DistributedTNMemoryEvidence,
) -> None:
    """Reject uncalibrated or over-budget distributed TN execution evidence."""

    if not evidence.passed:
        raise RuntimeError(
            "distributed TN memory evidence failed: " + ", ".join(evidence.blockers)
        )


__all__ = (
    "DistributedTNMemoryEvidence",
    "DistributedTNRankMemoryMeasurement",
    "TN_MEMORY_EVIDENCE_VERSION",
    "build_distributed_tn_memory_evidence",
    "require_distributed_tn_memory_evidence",
)
