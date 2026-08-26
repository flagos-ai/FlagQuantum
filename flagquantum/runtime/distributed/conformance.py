"""Fail-closed evidence contracts for FlagOS distributed conformance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .identity import DistributedIdentity

REQUIRED_FLAGOS_COLLECTIVES = (
    "all_gather_into_tensor",
    "all_reduce",
    "broadcast",
    "isend_irecv",
    "reduce_scatter_tensor",
)
REQUIRED_FLAGOS_DTYPES = ("complex64", "complex128")
STATEVECTOR_REQUIRED_FLAGOS_COLLECTIVES = (
    "all_gather_into_tensor",
    "isend_irecv",
)


@dataclass(frozen=True)
class FlagOSCollectiveConformanceCheck:
    """One rank-symmetric tensor collective check."""

    primitive: str
    dtype: str
    passed: bool
    max_abs_error: float
    elapsed_seconds: float
    payload_bytes: int
    device_type: str
    error: str | None = None

    def __post_init__(self) -> None:
        if self.primitive not in REQUIRED_FLAGOS_COLLECTIVES:
            raise ValueError(f"unsupported FlagOS collective {self.primitive!r}")
        if self.dtype not in REQUIRED_FLAGOS_DTYPES:
            raise ValueError(f"unsupported FlagOS conformance dtype {self.dtype!r}")
        if self.max_abs_error < 0 or self.elapsed_seconds < 0 or self.payload_bytes < 0:
            raise ValueError("collective measurements must be non-negative")
        if self.passed and self.error is not None:
            raise ValueError("a passing collective check cannot retain an error")

    def to_dict(self) -> dict[str, Any]:
        return {
            "primitive": self.primitive,
            "dtype": self.dtype,
            "passed": self.passed,
            "max_abs_error": self.max_abs_error,
            "elapsed_seconds": self.elapsed_seconds,
            "payload_bytes": self.payload_bytes,
            "device_type": self.device_type,
            "error": self.error,
        }


@dataclass(frozen=True)
class FlagOSStatevectorConformanceCheck:
    """Rank-local correctness for one truly sharded statevector dtype."""

    dtype: str
    passed: bool
    max_abs_error: float
    tolerance: float
    local_amplitudes: int
    total_amplitudes: int
    distributed_gate_count: int
    communication_count: int
    communication_bytes: int
    device_type: str
    distribution_semantics: str = "sharded_across_ranks"
    full_state_materialization: bool = False

    def __post_init__(self) -> None:
        if self.dtype not in REQUIRED_FLAGOS_DTYPES:
            raise ValueError(f"unsupported statevector dtype {self.dtype!r}")
        if self.max_abs_error < 0 or self.tolerance <= 0:
            raise ValueError("statevector accuracy measurements are invalid")
        if self.local_amplitudes <= 0 or self.total_amplitudes <= 0:
            raise ValueError("statevector amplitude counts must be positive")
        if self.local_amplitudes >= self.total_amplitudes:
            raise ValueError("distributed statevector check must retain a strict shard")
        if self.distributed_gate_count <= 0 or self.communication_count <= 0:
            raise ValueError("statevector check must execute a communicating gate")
        if self.communication_bytes <= 0:
            raise ValueError("statevector communication bytes must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "passed": self.passed,
            "max_abs_error": self.max_abs_error,
            "tolerance": self.tolerance,
            "local_amplitudes": self.local_amplitudes,
            "total_amplitudes": self.total_amplitudes,
            "distributed_gate_count": self.distributed_gate_count,
            "communication_count": self.communication_count,
            "communication_bytes": self.communication_bytes,
            "device_type": self.device_type,
            "distribution_semantics": self.distribution_semantics,
            "full_state_materialization": self.full_state_materialization,
        }


@dataclass(frozen=True)
class FlagOSDistributedConformanceReport:
    """Single-node FlagOS ProcessGroup evidence without an inferred FlagCX route."""

    identity: DistributedIdentity
    collective_checks: tuple[FlagOSCollectiveConformanceCheck, ...]
    statevector_checks: tuple[FlagOSStatevectorConformanceCheck, ...]
    rank_placement: tuple[Mapping[str, Any], ...]
    environment: Mapping[str, Any]
    world_size: int
    local_world_size: int
    node_count: int
    schema: str = "flagquantum_flagos_distributed_conformance_v1"
    blockers: tuple[str, ...] = field(
        default=(
            "flagcx_provider_identity_unavailable",
            "host_staging_unverified",
            "single_node_conformance_not_scalability_evidence",
        )
    )

    def __post_init__(self) -> None:
        if self.world_size < 2 or self.local_world_size != self.world_size:
            raise ValueError("F1 conformance requires one node with at least two ranks")
        if self.node_count != 1:
            raise ValueError("F1 conformance is single-node only")
        if self.identity.outer_backend != "flagos":
            raise ValueError("F1 conformance requires outer backend='flagos'")
        if self.identity.world_size != self.world_size:
            raise ValueError("distributed identity and report world sizes must match")
        if not self.identity.logical_device.startswith("flagos:"):
            raise ValueError("F1 conformance requires a logical flagos device")
        if len(self.rank_placement) != self.world_size:
            raise ValueError("rank placement must contain every distributed rank")

    @property
    def mechanical_conformance_accepted(self) -> bool:
        required = {
            (primitive, dtype)
            for primitive in REQUIRED_FLAGOS_COLLECTIVES
            for dtype in REQUIRED_FLAGOS_DTYPES
        }
        observed = {(item.primitive, item.dtype) for item in self.collective_checks}
        placement_ranks = {int(item["rank"]) for item in self.rank_placement}
        placement_devices = {int(item["device_index"]) for item in self.rank_placement}
        return bool(
            len(self.collective_checks) == len(required)
            and observed == required
            and all(item.passed for item in self.collective_checks)
            and all(item.device_type == "flagos" for item in self.collective_checks)
            and len(self.statevector_checks) == len(REQUIRED_FLAGOS_DTYPES)
            and {item.dtype for item in self.statevector_checks}
            == set(REQUIRED_FLAGOS_DTYPES)
            and all(item.passed for item in self.statevector_checks)
            and all(
                item.device_type == "flagos"
                and item.distribution_semantics == "sharded_across_ranks"
                and not item.full_state_materialization
                for item in self.statevector_checks
            )
            and placement_ranks == set(range(self.world_size))
            and len(placement_devices) == self.world_size
            and self.identity.process_group_initialized
        )

    @property
    def statevector_workload_conformance_accepted(self) -> bool:
        """Whether the measured statevector route works, not the whole provider."""

        required = {
            (primitive, dtype)
            for primitive in STATEVECTOR_REQUIRED_FLAGOS_COLLECTIVES
            for dtype in REQUIRED_FLAGOS_DTYPES
        }
        passing = {
            (item.primitive, item.dtype)
            for item in self.collective_checks
            if item.passed and item.device_type == "flagos"
        }
        return bool(
            required <= passing
            and len(self.statevector_checks) == len(REQUIRED_FLAGOS_DTYPES)
            and {item.dtype for item in self.statevector_checks}
            == set(REQUIRED_FLAGOS_DTYPES)
            and all(
                item.passed
                and item.device_type == "flagos"
                and item.distribution_semantics == "sharded_across_ranks"
                and not item.full_state_materialization
                for item in self.statevector_checks
            )
            and self.identity.process_group_initialized
        )

    @property
    def measurement_blockers(self) -> tuple[str, ...]:
        blockers = []
        if any(not item.passed for item in self.collective_checks):
            blockers.append("collective_conformance_failed")
        if any(
            item.primitive == "reduce_scatter_tensor" and not item.passed
            for item in self.collective_checks
        ):
            blockers.append("complex_reduce_scatter_unavailable")
        if any(not item.passed for item in self.statevector_checks):
            blockers.append("statevector_accuracy_or_execution_failed")
        return tuple(blockers)

    def require_accepted(self) -> None:
        if not self.mechanical_conformance_accepted:
            raise RuntimeError("FlagOS distributed mechanical conformance failed")

    def to_dict(self) -> dict[str, Any]:
        accepted = self.mechanical_conformance_accepted
        blockers = tuple(
            dict.fromkeys(
                (*self.identity.blockers, *self.blockers, *self.measurement_blockers)
            )
        )
        return {
            "schema": self.schema,
            "status": "passed" if accepted else "failed",
            "validation_scope": "flagos_process_group_single_node_candidate",
            "outer_backend": self.identity.outer_backend,
            "world_size": self.world_size,
            "local_world_size": self.local_world_size,
            "node_count": self.node_count,
            "rank_placement": [dict(item) for item in self.rank_placement],
            "collective_checks": [item.to_dict() for item in self.collective_checks],
            "statevector_checks": [item.to_dict() for item in self.statevector_checks],
            "distributed_identity": self.identity.to_dict(),
            "environment": dict(self.environment),
            "mechanical_conformance_accepted": accepted,
            "statevector_workload_conformance_accepted": (
                self.statevector_workload_conformance_accepted
            ),
            "flagcx_route_verified": self.identity.flagcx_route_verified,
            "host_staging_observed": self.identity.host_staging_observed,
            "communication_claim_allowed": self.identity.communication_claim_allowed,
            "scalability_claim_allowed": False,
            "release_gate_allowed": False,
            "blockers": blockers,
        }


__all__ = (
    "FlagOSCollectiveConformanceCheck",
    "FlagOSDistributedConformanceReport",
    "FlagOSStatevectorConformanceCheck",
    "REQUIRED_FLAGOS_COLLECTIVES",
    "REQUIRED_FLAGOS_DTYPES",
    "STATEVECTOR_REQUIRED_FLAGOS_COLLECTIVES",
)
