"""Fail-closed MPS production acceptance and measured planner integration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from ....core.ir import ensure_circuit_ir
from .errors import NonlocalMPSCompilationError


class MPSProductionAcceptanceError(RuntimeError):
    """A workload or evidence bundle cannot be promoted to production MPS."""


def select_mps_crossover_decision(
    surface: Mapping[str, Any],
    *,
    family: str,
    sites: int,
    max_bond: int,
    depth: int,
    boundary_rate: str,
    truncation_policy: Mapping[str, Any],
    topology: str,
) -> Mapping[str, Any]:
    """Select an exact measured crossover point, failing closed off-surface."""
    if not surface.get("performance_gate", {}).get("passed") or not surface.get(
        "scalability_claim_allowed", False
    ):
        return {
            "decision": "local",
            "world_size": 1,
            "reason": "scaling_performance_gate_not_certified",
            "measured": False,
        }
    points = surface.get("planner_crossover_surface", ())
    matches = [
        point
        for point in points
        if point.get("family") == family
        and point.get("sites") == sites
        and point.get("max_bond") == max_bond
        and point.get("depth") == depth
        and point.get("boundary_rate") == boundary_rate
        and point.get("truncation_policy") == dict(truncation_policy)
        and point.get("topology") == topology
    ]
    if not matches:
        return {
            "decision": "local",
            "world_size": 1,
            "reason": "no_exact_measured_crossover_point",
            "measured": False,
        }
    distributed = [
        point
        for point in matches
        if point.get("decision") == "distributed"
        and float(point.get("speedup_ci_low", 0.0)) > 1.0
    ]
    if distributed:
        return max(distributed, key=lambda point: float(point["speedup_ci_low"]))
    return {
        "decision": "local",
        "world_size": 1,
        "reason": "no_matching_speedup_confidence_interval_excludes_one",
        "measured": True,
    }


@dataclass(frozen=True)
class MPSProductionSupport:
    gate_arity: tuple[int, ...] = (1, 2)
    two_site_topology: str = "adjacent_only"
    backends: tuple[str, ...] = ("gloo", "nccl")
    precisions: tuple[str, ...] = ("complex64", "complex128")
    truncation_policies: tuple[str, ...] = ("exact", "max_bond_cutoff")
    optimizers: tuple[str, ...] = ("sgd", "adam")
    world_sizes: tuple[int, ...] = (2, 4, 8)
    full_mps_materialization_allowed: bool = False
    jax_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MPSAcceptanceGates:
    correctness_passed: bool
    performance_passed: bool
    capacity_passed: bool
    correctness_artifact: str | None = None
    performance_artifact: str | None = None
    capacity_artifact: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MPSAcceptanceGates":
        return cls(
            correctness_passed=bool(payload.get("correctness_passed", False)),
            performance_passed=bool(payload.get("performance_passed", False)),
            capacity_passed=bool(payload.get("capacity_passed", False)),
            correctness_artifact=payload.get("correctness_artifact"),
            performance_artifact=payload.get("performance_artifact"),
            capacity_artifact=payload.get("capacity_artifact"),
        )

    @property
    def production_passed(self) -> bool:
        return (
            self.correctness_passed
            and self.performance_passed
            and self.capacity_passed
            and all(
                (
                    self.correctness_artifact,
                    self.performance_artifact,
                    self.capacity_artifact,
                )
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "production_passed": self.production_passed}


@dataclass(frozen=True)
class MPSCrossoverMeasurement:
    world_size: int
    workload_min_bytes: int
    workload_max_bytes: int
    speedup_mean: float
    speedup_ci_low: float
    speedup_ci_high: float
    sample_count: int
    measured: bool = True

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MPSCrossoverMeasurement":
        return cls(**{key: payload[key] for key in cls.__dataclass_fields__})

    def __post_init__(self) -> None:
        if self.world_size not in {2, 4, 8}:
            raise ValueError("MPS crossover world_size must be 2, 4, or 8")
        if (
            self.workload_min_bytes < 0
            or self.workload_max_bytes < self.workload_min_bytes
        ):
            raise ValueError("invalid MPS crossover workload interval")
        if self.sample_count < 2:
            raise ValueError("MPS crossover requires at least two samples")
        if not self.measured:
            raise ValueError(
                "estimated crossover records cannot drive production planning"
            )
        if not self.speedup_ci_low <= self.speedup_mean <= self.speedup_ci_high:
            raise ValueError("speedup mean must lie inside its confidence interval")


@dataclass(frozen=True)
class MPSProductionPlan:
    execution_class: str
    world_size: int
    rationale: str
    gates: MPSAcceptanceGates
    support: MPSProductionSupport
    matched_measurement: MPSCrossoverMeasurement | None
    single_gpu_capacity_bytes: int
    estimated_workload_bytes: int
    distribution_semantics: str
    production_promotion_allowed: bool

    def summary(self) -> dict[str, Any]:
        return {
            "planner": "measured_mps_production_planner_v1",
            "execution_class": self.execution_class,
            "world_size": self.world_size,
            "rationale": self.rationale,
            "acceptance_gates": self.gates.to_dict(),
            "support": self.support.to_dict(),
            "matched_measurement": (
                None
                if self.matched_measurement is None
                else asdict(self.matched_measurement)
            ),
            "single_gpu_capacity_bytes": self.single_gpu_capacity_bytes,
            "estimated_workload_bytes": self.estimated_workload_bytes,
            "distribution_semantics": self.distribution_semantics,
            "production_promotion_allowed": self.production_promotion_allowed,
            "gpu_availability_used_as_distribution_reason": False,
        }


def validate_production_mps_workload(circuit_or_ir: Any, *, world_size: int) -> Any:
    """Validate the complete workload before any executor tensor allocation."""

    ir = ensure_circuit_ir(circuit_or_ir)
    if world_size not in {1, 2, 4, 8}:
        raise MPSProductionAcceptanceError(
            "production MPS world_size must be 1, 2, 4, or 8"
        )
    if world_size > ir.n_wires:
        raise MPSProductionAcceptanceError("MPS world_size cannot exceed wire count")
    for index, instruction in enumerate(ir.instructions):
        arity = len(instruction.wires)
        if arity not in {1, 2}:
            raise MPSProductionAcceptanceError(
                f"instruction {index}:{instruction.name} has unsupported arity {arity}"
            )
        if arity == 2 and abs(instruction.wires[0] - instruction.wires[1]) != 1:
            raise NonlocalMPSCompilationError(
                f"instruction {index}:{instruction.name} requires MPS routing"
            )
    return ir


def plan_production_mps(
    circuit_or_ir: Any,
    *,
    estimated_workload_bytes: int,
    single_gpu_capacity_bytes: int,
    gates: MPSAcceptanceGates | Mapping[str, Any],
    crossover: tuple[MPSCrossoverMeasurement | Mapping[str, Any], ...] = (),
    available_gpu_count: int = 0,
) -> MPSProductionPlan:
    """Choose local/speed/capacity execution from measured evidence only."""

    del (
        available_gpu_count
    )  # availability is checked by launch, never used as rationale
    if estimated_workload_bytes <= 0 or single_gpu_capacity_bytes <= 0:
        raise ValueError("MPS workload and single-GPU capacity bytes must be positive")
    resolved_gates = (
        gates
        if isinstance(gates, MPSAcceptanceGates)
        else MPSAcceptanceGates.from_mapping(gates)
    )
    measurements = tuple(
        item
        if isinstance(item, MPSCrossoverMeasurement)
        else MPSCrossoverMeasurement.from_mapping(item)
        for item in crossover
    )
    if not resolved_gates.production_passed:
        validate_production_mps_workload(circuit_or_ir, world_size=1)
        return MPSProductionPlan(
            execution_class="single_gpu_fast_path",
            world_size=1,
            rationale="production MPS gates are incomplete; distribution is fail-closed",
            gates=resolved_gates,
            support=MPSProductionSupport(),
            matched_measurement=None,
            single_gpu_capacity_bytes=single_gpu_capacity_bytes,
            estimated_workload_bytes=estimated_workload_bytes,
            distribution_semantics="single_device_fast_path",
            production_promotion_allowed=False,
        )
    candidates = tuple(
        item
        for item in measurements
        if item.workload_min_bytes
        <= estimated_workload_bytes
        <= item.workload_max_bytes
    )
    if estimated_workload_bytes > single_gpu_capacity_bytes:
        capacity = tuple(item for item in candidates if item.world_size > 1)
        if not capacity:
            raise MPSProductionAcceptanceError(
                "capacity workload has no matching measured distributed artifact"
            )
        selected = min(capacity, key=lambda item: item.world_size)
        execution_class = "capacity_oriented_distribution"
        rationale = "measured workload exceeds the single-GPU capacity baseline"
    else:
        speed = tuple(item for item in candidates if item.speedup_ci_low > 1.0)
        if speed:
            selected = max(
                speed, key=lambda item: (item.speedup_ci_low, -item.world_size)
            )
            execution_class = "speed_oriented_distribution"
            rationale = "measured speedup confidence interval excludes no improvement"
        else:
            validate_production_mps_workload(circuit_or_ir, world_size=1)
            return MPSProductionPlan(
                execution_class="single_gpu_fast_path",
                world_size=1,
                rationale="no matched measurement proves distributed speedup",
                gates=resolved_gates,
                support=MPSProductionSupport(),
                matched_measurement=None,
                single_gpu_capacity_bytes=single_gpu_capacity_bytes,
                estimated_workload_bytes=estimated_workload_bytes,
                distribution_semantics="single_device_fast_path",
                production_promotion_allowed=True,
            )
    validate_production_mps_workload(circuit_or_ir, world_size=selected.world_size)
    return MPSProductionPlan(
        execution_class=execution_class,
        world_size=selected.world_size,
        rationale=rationale,
        gates=resolved_gates,
        support=MPSProductionSupport(),
        matched_measurement=selected,
        single_gpu_capacity_bytes=single_gpu_capacity_bytes,
        estimated_workload_bytes=estimated_workload_bytes,
        distribution_semantics="sharded_across_ranks",
        production_promotion_allowed=True,
    )


def build_mps_release_artifact(
    *,
    gates: MPSAcceptanceGates,
    runtime_records: tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    """Build a release payload only from independently passing measured records."""

    if not gates.production_passed:
        raise MPSProductionAcceptanceError(
            "MPS release artifact requires correctness, performance, and capacity gates"
        )
    required_kinds = {"correctness", "performance", "capacity"}
    kinds = {str(record.get("gate")) for record in runtime_records}
    if not required_kinds <= kinds:
        raise MPSProductionAcceptanceError(
            "MPS release records must include correctness, performance, and capacity"
        )
    for record in runtime_records:
        if record.get("evidence_source") != "measured_runtime":
            raise MPSProductionAcceptanceError(
                "MPS release records must come from measured_runtime"
            )
        if not record.get("artifact_path") or not record.get("artifact_sha256"):
            raise MPSProductionAcceptanceError(
                "MPS release records require artifact path and sha256"
            )
        if (
            record.get("gate") == "performance"
            and float(record.get("speedup_ci_low", 0.0)) <= 1.0
        ):
            raise MPSProductionAcceptanceError(
                "MPS performance confidence interval must exclude no improvement"
            )
        if record.get("gate") == "capacity" and not (
            record.get("single_gpu_capacity_failure")
            and record.get("sharded_completion")
            and record.get("full_mps_materialization") is False
        ):
            raise MPSProductionAcceptanceError(
                "MPS capacity record must prove single-GPU failure and sharded completion"
            )
    return {
        "schema_version": "flagquantum.mps_production_release.v1",
        "artifact_classification": "measured_production_release",
        "acceptance_gates": gates.to_dict(),
        "support": MPSProductionSupport().to_dict(),
        "runtime_records": tuple(dict(record) for record in runtime_records),
        "production_distributed_mps": True,
        "distribution_semantics": "sharded_across_ranks",
        "scalability_claim_allowed": True,
    }


__all__ = (
    "MPSAcceptanceGates",
    "MPSCrossoverMeasurement",
    "MPSProductionAcceptanceError",
    "MPSProductionPlan",
    "MPSProductionSupport",
    "plan_production_mps",
    "select_mps_crossover_decision",
    "build_mps_release_artifact",
    "validate_production_mps_workload",
)
