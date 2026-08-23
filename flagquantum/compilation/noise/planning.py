"""Structured planning contracts for noisy execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..models import ExecutionPlan

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


def build_noisy_execution_plan(
    execution_plan: ExecutionPlan,
    *,
    representation: StateRepresentation,
    evolution: EvolutionSemantics,
    trajectories: int | None = None,
    seed: int | None = None,
    min_trajectories: int = 1,
    target_standard_error: float | None = None,
    cutoff: float = 0.0,
    memory_limit_bytes: int | None = None,
    estimated_memory_bytes: int | None = None,
    noise_model_identity: str | None = None,
) -> NoisyExecutionPlan:
    """Project a conventional execution plan into the noisy runtime contract."""

    trajectory = (
        TrajectoryPlan(
            count=trajectories,
            seed=seed,
            min_count=min_trajectories,
            target_standard_error=target_standard_error,
        )
        if evolution == "quantum_trajectory" and trajectories is not None
        else None
    )
    return NoisyExecutionPlan(
        representation=representation,
        evolution=evolution,
        trajectory=trajectory,
        parallel=ParallelPlan(world_size=execution_plan.world_size),
        error_budget=NoiseErrorBudget(
            sampling_error_enabled=evolution == "quantum_trajectory",
            truncation_cutoff=cutoff,
        ),
        memory=MemoryPlan(
            estimated_bytes=(
                execution_plan.state_bytes
                if estimated_memory_bytes is None
                else int(estimated_memory_bytes)
            ),
            limit_bytes=memory_limit_bytes,
        ),
        noise_model_identity=noise_model_identity,
    )


__all__ = (
    "EvolutionSemantics",
    "MemoryPlan",
    "NoiseErrorBudget",
    "NoisyExecutionPlan",
    "ParallelPlan",
    "StateRepresentation",
    "TrajectoryPlan",
    "build_noisy_execution_plan",
)
