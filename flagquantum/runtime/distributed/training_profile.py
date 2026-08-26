"""Fail-closed contracts for the FlagOS sharded-training scale ladder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

FLAGOS_TRAINING_WORLD_SIZES = (2, 4, 8)
FLAGOS_TRAINING_DTYPES = ("complex64", "complex128")
FLAGOS_TRAINING_CASES = ("gradient_reference", "sgd_trajectory", "adam_trajectory")


@dataclass(frozen=True)
class FlagOSTrainingCase:
    """One measured backward or optimizer trajectory on FlagOS shards."""

    name: str
    dtype: str
    passed: bool
    n_wires: int
    steps: int
    local_amplitudes: int
    total_amplitudes: int
    value_max_abs_error: float
    gradient_max_abs_error: float
    parameter_max_abs_error: float
    rank_consistency_error: float
    tolerance: float
    communication_count: int
    communication_bytes: int
    peak_memory_bytes: int
    device_type: str
    distribution_semantics: str = "sharded_across_ranks"
    gradient_distribution: str = "replicated_after_all_reduce"
    optimizer_update_semantics: str = "owner_step_then_broadcast"
    full_state_materialization: bool = False
    backward_uses_full_state_replay: bool = False
    reference_scope: str = "bounded_cpu_complex128"
    error: str | None = None

    def __post_init__(self) -> None:
        if self.name not in FLAGOS_TRAINING_CASES:
            raise ValueError(f"unsupported FlagOS training case {self.name!r}")
        if self.dtype not in FLAGOS_TRAINING_DTYPES:
            raise ValueError(f"unsupported FlagOS training dtype {self.dtype!r}")
        if self.n_wires < 2 or self.steps < 1:
            raise ValueError("training cases require at least two wires and one step")
        if not 0 < self.local_amplitudes < self.total_amplitudes:
            raise ValueError("training cases must retain a strict rank-local shard")
        measurements = (
            self.value_max_abs_error,
            self.gradient_max_abs_error,
            self.parameter_max_abs_error,
            self.rank_consistency_error,
            self.tolerance,
            self.communication_count,
            self.communication_bytes,
            self.peak_memory_bytes,
        )
        if any(value < 0 for value in measurements) or self.tolerance == 0:
            raise ValueError("training measurements must be non-negative")
        if self.passed and self.error is not None:
            raise ValueError("a passing training case cannot retain an error")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class FlagOSTrainingRun:
    """All F3 training cases measured for one single-node world size."""

    world_size: int
    local_world_size: int
    node_count: int
    cases: tuple[FlagOSTrainingCase, ...]
    rank_placement: tuple[Mapping[str, Any], ...]
    environment: Mapping[str, Any]
    outer_backend: str = "flagos"
    logical_device_type: str = "flagos"
    flagcx_route_verified: bool = False
    host_staging_observed: bool | None = None

    def __post_init__(self) -> None:
        if self.world_size not in FLAGOS_TRAINING_WORLD_SIZES:
            raise ValueError("training run world size must be 2, 4, or 8")
        if self.local_world_size != self.world_size or self.node_count != 1:
            raise ValueError("F3 training runs require one complete node")
        if len(self.rank_placement) != self.world_size:
            raise ValueError("rank placement must include every rank")
        if self.outer_backend != "flagos" or self.logical_device_type != "flagos":
            raise ValueError("F3 training runs require the public FlagOS boundary")
        if self.flagcx_route_verified:
            raise ValueError("F3 cannot infer a provider-owned FlagCX route")

    @property
    def accepted(self) -> bool:
        expected = {
            (name, dtype)
            for name in FLAGOS_TRAINING_CASES
            for dtype in FLAGOS_TRAINING_DTYPES
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
                and case.gradient_distribution == "replicated_after_all_reduce"
                and case.optimizer_update_semantics == "owner_step_then_broadcast"
                and not case.full_state_materialization
                and not case.backward_uses_full_state_replay
                and case.communication_count > 0
                and case.communication_bytes > 0
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
            "training_run_accepted": self.accepted,
            "flagcx_route_verified": self.flagcx_route_verified,
            "host_staging_observed": self.host_staging_observed,
        }


@dataclass(frozen=True)
class FlagOSTrainingProfile:
    """A narrow 2/4/8-card F3 training-development profile."""

    runs: tuple[FlagOSTrainingRun, ...]
    environment: Mapping[str, Any]
    schema: str = "flagquantum_flagos_statevector_training_profile_v1"
    blockers: tuple[str, ...] = field(
        default=(
            "inner_communication_route_unattributed",
            "host_staging_unverified",
            "single_node_bounded_training_development_evidence",
            "single_device_capacity_failure_not_measured",
            "training_performance_and_convergence_not_measured",
        )
    )

    @property
    def accepted(self) -> bool:
        return bool(
            len(self.runs) == len(FLAGOS_TRAINING_WORLD_SIZES)
            and {run.world_size for run in self.runs}
            == set(FLAGOS_TRAINING_WORLD_SIZES)
            and all(run.accepted for run in self.runs)
        )

    @property
    def measurement_blockers(self) -> tuple[str, ...]:
        blockers = []
        if {run.world_size for run in self.runs} != set(FLAGOS_TRAINING_WORLD_SIZES):
            blockers.append("training_scale_ladder_incomplete")
        if any(not run.accepted for run in self.runs):
            blockers.append("statevector_training_run_failed")
        return tuple(blockers)

    def require_accepted(self) -> None:
        if not self.accepted:
            raise RuntimeError("FlagOS statevector training ladder failed")

    def to_dict(self) -> dict[str, Any]:
        blockers = tuple(dict.fromkeys((*self.blockers, *self.measurement_blockers)))
        return {
            "schema": self.schema,
            "status": "passed" if self.accepted else "failed",
            "validation_scope": "flagos_statevector_single_node_2_4_8_card_training",
            "distribution_semantics": "sharded_across_ranks",
            "gradient_distribution": "replicated_after_all_reduce",
            "optimizer_update_semantics": "owner_step_then_broadcast",
            "claim_evidence_type": "development_hardware_profile",
            "world_sizes": [run.world_size for run in self.runs],
            "runs": [run.to_dict() for run in self.runs],
            "environment": dict(self.environment),
            "training_ladder_accepted": self.accepted,
            "sharded_backward_profile_accepted": self.accepted,
            "sharded_optimizer_profile_accepted": self.accepted,
            "flagcx_route_verified": False,
            "host_staging_observed": None,
            "communication_claim_allowed": False,
            "scalability_claim_allowed": False,
            "production_support_claim_allowed": False,
            "release_gate_allowed": False,
            "blockers": blockers,
        }


def training_case_from_dict(payload: Mapping[str, Any]) -> FlagOSTrainingCase:
    fields = FlagOSTrainingCase.__dataclass_fields__
    return FlagOSTrainingCase(
        **{name: payload[name] for name in fields if name in payload}
    )


def training_run_from_dict(payload: Mapping[str, Any]) -> FlagOSTrainingRun:
    return FlagOSTrainingRun(
        world_size=int(payload["world_size"]),
        local_world_size=int(payload["local_world_size"]),
        node_count=int(payload["node_count"]),
        cases=tuple(training_case_from_dict(item) for item in payload["cases"]),
        rank_placement=tuple(payload["rank_placement"]),
        environment=dict(payload["environment"]),
        outer_backend=str(payload.get("outer_backend", "flagos")),
        logical_device_type=str(payload.get("logical_device_type", "flagos")),
        flagcx_route_verified=bool(payload.get("flagcx_route_verified", False)),
        host_staging_observed=payload.get("host_staging_observed"),
    )


def build_training_profile(
    runs: Sequence[Mapping[str, Any] | FlagOSTrainingRun],
    *,
    environment: Mapping[str, Any],
) -> FlagOSTrainingProfile:
    parsed = tuple(
        run if isinstance(run, FlagOSTrainingRun) else training_run_from_dict(run)
        for run in runs
    )
    return FlagOSTrainingProfile(
        runs=tuple(sorted(parsed, key=lambda item: item.world_size)),
        environment=environment,
    )


__all__ = (
    "FLAGOS_TRAINING_CASES",
    "FLAGOS_TRAINING_DTYPES",
    "FLAGOS_TRAINING_WORLD_SIZES",
    "FlagOSTrainingCase",
    "FlagOSTrainingProfile",
    "FlagOSTrainingRun",
    "build_training_profile",
    "training_case_from_dict",
    "training_run_from_dict",
)
