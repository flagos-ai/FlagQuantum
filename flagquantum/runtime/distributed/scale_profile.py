"""Fail-closed contracts for FlagOS statevector scale-ladder evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

FLAGOS_SCALE_WORLD_SIZES = (2, 4, 8)
FLAGOS_SCALE_DTYPES = ("complex64", "complex128")
FLAGOS_SCALE_CASES = (
    "cross_shard_reference",
    "persistent_layout_reference",
    "capacity_invariant",
)


@dataclass(frozen=True)
class FlagOSStatevectorScaleCase:
    """One dtype/circuit result from a genuinely sharded FlagOS execution."""

    name: str
    dtype: str
    passed: bool
    n_wires: int
    local_amplitudes: int
    total_amplitudes: int
    local_state_bytes: int
    local_memory_bytes_by_rank: tuple[int, ...]
    memory_measurement: str
    communication_count: int
    communication_bytes: int
    peak_scratch_bytes: int
    distributed_gate_count: int
    elapsed_seconds: float
    max_abs_error: float | None
    norm_error: float
    determinism_error: float
    tolerance: float
    persistent_wire_layout: bool
    device_type: str
    distribution_semantics: str = "sharded_across_ranks"
    reference_scope: str = "rank_local_invariant_only"
    full_state_materialization: bool = False
    error: str | None = None

    def __post_init__(self) -> None:
        if self.name not in FLAGOS_SCALE_CASES:
            raise ValueError(f"unsupported FlagOS scale case {self.name!r}")
        if self.dtype not in FLAGOS_SCALE_DTYPES:
            raise ValueError(f"unsupported FlagOS scale dtype {self.dtype!r}")
        if self.n_wires < 2:
            raise ValueError("scale cases require at least two wires")
        if not 0 < self.local_amplitudes < self.total_amplitudes:
            raise ValueError("scale cases must retain a strict rank-local shard")
        measurements = (
            self.local_state_bytes,
            self.communication_count,
            self.communication_bytes,
            self.peak_scratch_bytes,
            self.distributed_gate_count,
            self.elapsed_seconds,
            self.norm_error,
            self.determinism_error,
            self.tolerance,
        )
        if any(value < 0 for value in measurements) or self.tolerance == 0:
            raise ValueError("scale measurements must be non-negative")
        if not self.local_memory_bytes_by_rank or any(
            value <= 0 for value in self.local_memory_bytes_by_rank
        ):
            raise ValueError("every rank must report positive local memory ownership")
        if self.memory_measurement not in {
            "provider_peak_allocator_bytes",
            "runtime_accounted_state_scratch_workspace",
        }:
            raise ValueError("unsupported local memory measurement method")
        if self.max_abs_error is not None and self.max_abs_error < 0:
            raise ValueError("reference error must be non-negative")
        if self.passed and self.error is not None:
            raise ValueError("a passing scale case cannot retain an error")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "passed": self.passed,
            "n_wires": self.n_wires,
            "local_amplitudes": self.local_amplitudes,
            "total_amplitudes": self.total_amplitudes,
            "local_state_bytes": self.local_state_bytes,
            "local_memory_bytes_by_rank": list(self.local_memory_bytes_by_rank),
            "memory_measurement": self.memory_measurement,
            "communication_count": self.communication_count,
            "communication_bytes": self.communication_bytes,
            "peak_scratch_bytes": self.peak_scratch_bytes,
            "distributed_gate_count": self.distributed_gate_count,
            "elapsed_seconds": self.elapsed_seconds,
            "max_abs_error": self.max_abs_error,
            "norm_error": self.norm_error,
            "determinism_error": self.determinism_error,
            "tolerance": self.tolerance,
            "persistent_wire_layout": self.persistent_wire_layout,
            "device_type": self.device_type,
            "distribution_semantics": self.distribution_semantics,
            "reference_scope": self.reference_scope,
            "full_state_materialization": self.full_state_materialization,
            "error": self.error,
        }


@dataclass(frozen=True)
class FlagOSStatevectorScaleRun:
    """All statevector cases measured at one single-node world size."""

    world_size: int
    local_world_size: int
    node_count: int
    cases: tuple[FlagOSStatevectorScaleCase, ...]
    rank_placement: tuple[Mapping[str, Any], ...]
    environment: Mapping[str, Any]
    outer_backend: str = "flagos"
    logical_device_type: str = "flagos"
    flagcx_route_verified: bool = False
    host_staging_observed: bool | None = None

    def __post_init__(self) -> None:
        if self.world_size not in FLAGOS_SCALE_WORLD_SIZES:
            raise ValueError("scale run world size must be 2, 4, or 8")
        if self.local_world_size != self.world_size or self.node_count != 1:
            raise ValueError("F2 scale runs require one complete node")
        if len(self.rank_placement) != self.world_size:
            raise ValueError("rank placement must include every rank")
        if self.outer_backend != "flagos" or self.logical_device_type != "flagos":
            raise ValueError("F2 scale runs require the public FlagOS boundary")
        if self.flagcx_route_verified:
            raise ValueError("F2 cannot infer a provider-owned FlagCX route")

    @property
    def accepted(self) -> bool:
        expected = {
            (name, dtype)
            for name in FLAGOS_SCALE_CASES
            for dtype in FLAGOS_SCALE_DTYPES
        }
        observed = {(case.name, case.dtype) for case in self.cases}
        ranks = {int(item["rank"]) for item in self.rank_placement}
        devices = {int(item["device_index"]) for item in self.rank_placement}
        return bool(
            len(self.cases) == len(expected)
            and observed == expected
            and all(
                case.passed
                and case.device_type == "flagos"
                and case.distribution_semantics == "sharded_across_ranks"
                and not case.full_state_materialization
                and case.communication_count > 0
                and case.communication_bytes > 0
                and (
                    case.distributed_gate_count > 0
                    or (
                        case.name == "persistent_layout_reference"
                        and case.persistent_wire_layout
                    )
                )
                for case in self.cases
            )
            and ranks == set(range(self.world_size))
            and len(devices) == self.world_size
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_size": self.world_size,
            "local_world_size": self.local_world_size,
            "node_count": self.node_count,
            "outer_backend": self.outer_backend,
            "logical_device_type": self.logical_device_type,
            "rank_placement": [dict(item) for item in self.rank_placement],
            "cases": [case.to_dict() for case in self.cases],
            "environment": dict(self.environment),
            "scale_run_accepted": self.accepted,
            "flagcx_route_verified": self.flagcx_route_verified,
            "host_staging_observed": self.host_staging_observed,
        }


@dataclass(frozen=True)
class FlagOSStatevectorScaleProfile:
    """A 2/4/8-card development profile with deliberately narrow claims."""

    runs: tuple[FlagOSStatevectorScaleRun, ...]
    environment: Mapping[str, Any]
    schema: str = "flagquantum_flagos_statevector_scale_profile_v1"
    blockers: tuple[str, ...] = field(
        default=(
            "flagcx_provider_identity_unavailable",
            "host_staging_unverified",
            "single_node_forward_only_development_evidence",
            "single_device_capacity_failure_not_measured",
            "sharded_backward_and_optimizer_not_measured",
        )
    )

    @property
    def accepted(self) -> bool:
        return bool(
            len(self.runs) == len(FLAGOS_SCALE_WORLD_SIZES)
            and {run.world_size for run in self.runs} == set(FLAGOS_SCALE_WORLD_SIZES)
            and all(run.accepted for run in self.runs)
        )

    @property
    def measurement_blockers(self) -> tuple[str, ...]:
        blockers = []
        observed = {run.world_size for run in self.runs}
        if observed != set(FLAGOS_SCALE_WORLD_SIZES):
            blockers.append("scale_ladder_incomplete")
        if any(not run.accepted for run in self.runs):
            blockers.append("statevector_scale_run_failed")
        return tuple(blockers)

    def require_accepted(self) -> None:
        if not self.accepted:
            raise RuntimeError("FlagOS statevector scale ladder failed")

    def to_dict(self) -> dict[str, Any]:
        blockers = tuple(dict.fromkeys((*self.blockers, *self.measurement_blockers)))
        return {
            "schema": self.schema,
            "status": "passed" if self.accepted else "failed",
            "validation_scope": "flagos_statevector_single_node_2_4_8_card_ladder",
            "distribution_semantics": "sharded_across_ranks",
            "claim_evidence_type": "development_hardware_profile",
            "world_sizes": [run.world_size for run in self.runs],
            "runs": [run.to_dict() for run in self.runs],
            "environment": dict(self.environment),
            "scale_ladder_accepted": self.accepted,
            "statevector_forward_scale_profile_accepted": self.accepted,
            "flagcx_route_verified": False,
            "host_staging_observed": None,
            "communication_claim_allowed": False,
            "scalability_claim_allowed": False,
            "production_support_claim_allowed": False,
            "release_gate_allowed": False,
            "blockers": blockers,
        }


def scale_case_from_dict(payload: Mapping[str, Any]) -> FlagOSStatevectorScaleCase:
    """Parse a worker payload without accepting unknown constructor behavior."""

    fields = FlagOSStatevectorScaleCase.__dataclass_fields__
    values = {name: payload[name] for name in fields if name in payload}
    if "local_memory_bytes_by_rank" in values:
        values["local_memory_bytes_by_rank"] = tuple(
            values["local_memory_bytes_by_rank"]
        )
    return FlagOSStatevectorScaleCase(**values)


def scale_run_from_dict(payload: Mapping[str, Any]) -> FlagOSStatevectorScaleRun:
    """Parse one worker run for fail-closed outer-profile aggregation."""

    return FlagOSStatevectorScaleRun(
        world_size=int(payload["world_size"]),
        local_world_size=int(payload["local_world_size"]),
        node_count=int(payload["node_count"]),
        cases=tuple(scale_case_from_dict(item) for item in payload["cases"]),
        rank_placement=tuple(payload["rank_placement"]),
        environment=dict(payload["environment"]),
        outer_backend=str(payload.get("outer_backend", "flagos")),
        logical_device_type=str(payload.get("logical_device_type", "flagos")),
        flagcx_route_verified=bool(payload.get("flagcx_route_verified", False)),
        host_staging_observed=payload.get("host_staging_observed"),
    )


def build_scale_profile(
    runs: Sequence[Mapping[str, Any] | FlagOSStatevectorScaleRun],
    *,
    environment: Mapping[str, Any],
) -> FlagOSStatevectorScaleProfile:
    """Build a profile from independently launched world-size reports."""

    parsed = tuple(
        run if isinstance(run, FlagOSStatevectorScaleRun) else scale_run_from_dict(run)
        for run in runs
    )
    return FlagOSStatevectorScaleProfile(
        runs=tuple(sorted(parsed, key=lambda item: item.world_size)),
        environment=environment,
    )


__all__ = (
    "FLAGOS_SCALE_CASES",
    "FLAGOS_SCALE_DTYPES",
    "FLAGOS_SCALE_WORLD_SIZES",
    "FlagOSStatevectorScaleCase",
    "FlagOSStatevectorScaleProfile",
    "FlagOSStatevectorScaleRun",
    "build_scale_profile",
    "scale_case_from_dict",
    "scale_run_from_dict",
)
