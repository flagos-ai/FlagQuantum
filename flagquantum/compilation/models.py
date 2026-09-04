"""Typed analysis and execution-plan models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Mapping, cast

from ..core.contracts import RuntimePlanContract
from ..core.ir import Instruction

if TYPE_CHECKING:
    from .performance_calibration import CalibratedPlanCost

StateRepresentation = Literal["density_matrix", "statevector", "mps", "tensor_network"]
EvolutionSemantics = Literal["exact_channel", "quantum_trajectory"]


@dataclass(frozen=True)
class TrajectoryPlan:
    """Sampling controls shared by all quantum-trajectory backends."""

    count: int
    seed: int | None = None
    min_count: int = 1
    target_standard_error: float | None = None

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("trajectory count must be positive")
        if self.min_count <= 0 or self.min_count > self.count:
            raise ValueError("minimum trajectory count must be in [1, count]")
        if self.target_standard_error is not None and self.target_standard_error <= 0:
            raise ValueError("target standard error must be positive")


@dataclass(frozen=True)
class ParallelPlan:
    """Backend-neutral parallel ownership requested by a noisy workload."""

    world_size: int = 1

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")


@dataclass(frozen=True)
class NoiseErrorBudget:
    """Approximation controls attributable to noisy evolution."""

    sampling_error_enabled: bool
    truncation_cutoff: float = 0.0

    def __post_init__(self) -> None:
        if self.truncation_cutoff < 0:
            raise ValueError("truncation_cutoff must be non-negative")


@dataclass(frozen=True)
class MemoryPlan:
    """Memory estimate and optional capacity limit for noisy execution."""

    estimated_bytes: int
    limit_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.estimated_bytes < 0:
            raise ValueError("estimated_bytes must be non-negative")
        if self.limit_bytes is not None and self.limit_bytes <= 0:
            raise ValueError("limit_bytes must be positive when provided")

    @property
    def fits(self) -> bool:
        return self.limit_bytes is None or self.estimated_bytes <= self.limit_bytes


@dataclass(frozen=True)
class NoisyExecutionPlan:
    """Representation-independent contract consumed by noise executors."""

    representation: StateRepresentation
    evolution: EvolutionSemantics
    trajectory: TrajectoryPlan | None
    parallel: ParallelPlan
    error_budget: NoiseErrorBudget
    memory: MemoryPlan
    noise_model_identity: str | None = None

    def __post_init__(self) -> None:
        if self.evolution == "exact_channel" and self.trajectory is not None:
            raise ValueError("exact channel evolution cannot have a trajectory plan")
        if self.evolution == "quantum_trajectory" and self.trajectory is None:
            raise ValueError("quantum trajectory evolution requires a trajectory plan")

    def summary(self) -> dict[str, object]:
        return {
            "representation": self.representation,
            "evolution": self.evolution,
            "trajectory_count": (
                None if self.trajectory is None else self.trajectory.count
            ),
            "trajectory_seed": (
                None if self.trajectory is None else self.trajectory.seed
            ),
            "min_trajectory_count": (
                None if self.trajectory is None else self.trajectory.min_count
            ),
            "target_standard_error": (
                None
                if self.trajectory is None
                else self.trajectory.target_standard_error
            ),
            "world_size": self.parallel.world_size,
            "sampling_error_enabled": self.error_budget.sampling_error_enabled,
            "truncation_cutoff": self.error_budget.truncation_cutoff,
            "estimated_memory_bytes": self.memory.estimated_bytes,
            "memory_limit_bytes": self.memory.limit_bytes,
            "memory_fits": self.memory.fits,
            "noise_model_identity": self.noise_model_identity,
        }


@dataclass(frozen=True)
class LayerPlan:
    index: int
    instructions: tuple[Instruction, ...]
    wires: tuple[int, ...]


@dataclass(frozen=True)
class CircuitAnalysis:
    n_wires: int
    n_instructions: int
    depth: int
    gate_counts: dict[str, int]
    wire_usage: tuple[int, ...]
    max_gate_width: int
    two_qubit_gates: int
    multi_qubit_gates: int
    channel_count: int = 0
    has_noise: bool = False


@dataclass(frozen=True)
class ExecutionPlan:
    analysis: CircuitAnalysis
    layers: tuple[LayerPlan, ...]
    state_bytes: int
    recommended_mode: str
    world_size: int
    shardable_wires: tuple[int, ...]
    state_mode: str = "statevector"
    user_tier: str = "single_device"
    usability_contract: str = "single_api_fast_path"
    runtime_config: Mapping[str, Any] | None = None
    routing_plan: Mapping[str, Any] | None = None
    noisy_execution_plan: NoisyExecutionPlan | None = None
    _contract_payload_json: str | None = None

    @property
    def identity(self) -> str:
        return str(self.to_dict()["identity"])

    def _contract_section(self, name: str) -> Mapping[str, object]:
        return cast(Mapping[str, object], self.to_dict()[name])

    @property
    def schema_version(self) -> str:
        return str(self.to_dict()["version"])

    @property
    def program_fingerprint(self) -> str:
        return str(self._contract_section("fingerprints")["program"])

    @property
    def options_fingerprint(self) -> str:
        return str(self._contract_section("fingerprints")["options"])

    @property
    def environment_fingerprint(self) -> str:
        return str(self._contract_section("fingerprints")["environment"])

    @property
    def compiler_fingerprint(self) -> str:
        return str(self._contract_section("fingerprints")["compiler"])

    @property
    def mode(self) -> str:
        return str(self._contract_section("decision")["mode"])

    @property
    def backend(self) -> str:
        return str(self._contract_section("decision")["backend"])

    @property
    def device(self) -> str:
        return str(self._contract_section("decision")["device"])

    @property
    def target(self) -> str:
        return str(self._contract_section("decision")["target"])

    @property
    def batch_size(self) -> int:
        return cast(int, self._contract_section("decision")["batch_size"])

    @property
    def precision(self) -> str:
        return str(self._contract_section("decision")["precision"])

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    def to_dict(self) -> dict[str, object]:
        from .execution_plan_contract import plan_to_dict

        return plan_to_dict(self)

    def to_json(self, *, indent: int | None = None) -> str:
        from .execution_plan_contract import plan_to_json

        return plan_to_json(self, indent=indent)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExecutionPlan":
        from .execution_plan_contract import plan_from_dict

        return plan_from_dict(payload)

    @classmethod
    def from_json(cls, text: str) -> "ExecutionPlan":
        from .execution_plan_contract import plan_from_json

        return plan_from_json(text)

    def to_contract(self) -> RuntimePlanContract:
        from .execution_plan_contract import execution_plan_contract

        return execution_plan_contract(self)

    def calibrated_cost(self, artifact: Mapping[str, Any]) -> CalibratedPlanCost:
        from .performance_calibration import calibrate_plan_cost

        return calibrate_plan_cost(self, artifact)

    def summary(self) -> dict[str, Any]:
        is_distributed = self.world_size > 1
        distribution_semantics = (
            "sharded_across_ranks"
            if self.recommended_mode == "distributed_statevector" and is_distributed
            else (
                "single_device_fast_path"
                if not is_distributed
                else "requires_runtime_summary"
            )
        )
        sharding_available = distribution_semantics == "sharded_across_ranks"
        blockers: tuple[str, ...] = ()
        if is_distributed and not sharding_available:
            blockers = ("runtime_summary_required_for_scalability_claim",)
        summary = {
            "state_mode": self.state_mode,
            "recommended_mode": self.recommended_mode,
            "claim_evidence_type": "plan_preflight",
            "user_tier": self.user_tier,
            "usability_contract": self.usability_contract,
            "distribution_semantics": distribution_semantics,
            "scalability_claim_allowed": False,
            "release_gate_allowed": False,
            "sharding_plan_available": sharding_available,
            "scalability_blockers": blockers,
            "world_size": self.world_size,
            "state_bytes": self.state_bytes,
            "depth": self.analysis.depth,
            "n_wires": self.analysis.n_wires,
            "n_instructions": self.analysis.n_instructions,
            "two_qubit_gates": self.analysis.two_qubit_gates,
            "multi_qubit_gates": self.analysis.multi_qubit_gates,
            "shardable_wires": self.shardable_wires,
            "runtime_config": dict(self.runtime_config or {}),
            "routing_plan": dict(self.routing_plan or {}),
            "noisy_execution_plan": (
                None
                if self.noisy_execution_plan is None
                else self.noisy_execution_plan.summary()
            ),
        }
        if self._contract_payload_json is not None:
            summary.update(
                {
                    "identity": self.identity,
                    "schema_version": self.schema_version,
                    "program_fingerprint": self.program_fingerprint,
                    "options_fingerprint": self.options_fingerprint,
                    "environment_fingerprint": self.environment_fingerprint,
                    "compiler_fingerprint": self.compiler_fingerprint,
                    "mode": self.mode,
                    "backend": self.backend,
                    "device": self.device,
                    "target": self.target,
                    "batch_size": self.batch_size,
                    "precision": self.precision,
                    "is_distributed": self.is_distributed,
                }
            )
        return summary


__all__ = [
    "CircuitAnalysis",
    "EvolutionSemantics",
    "ExecutionPlan",
    "LayerPlan",
    "MemoryPlan",
    "NoiseErrorBudget",
    "NoisyExecutionPlan",
    "ParallelPlan",
    "StateRepresentation",
    "TrajectoryPlan",
]
