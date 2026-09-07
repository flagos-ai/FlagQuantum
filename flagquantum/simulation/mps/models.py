"""Configuration, compiled schedules, and results for MPS numerics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ...core.ir import Instruction

_MPS_INSTRUCTION_SCHEDULE_CACHE: dict[tuple[Any, ...], tuple[tuple[int, ...], ...]] = {}


@dataclass(frozen=True)
class MPSConfig:
    max_bond: int | None = None
    cutoff: float = 0.0
    dense_observable_wires: int = 12
    svd_driver: str | None = "gesvd"


if TYPE_CHECKING:
    from ...runtime.trajectories import TrajectoryFailure, TrajectoryStatistics
    from .state import MPSState


def _mps_instruction_schedule(
    instructions: tuple[Instruction, ...], fuse_single_qubit: bool
) -> tuple[tuple[int, ...], ...]:
    key = (
        bool(fuse_single_qubit),
        tuple(
            (
                instruction.name,
                instruction.wires,
                tuple(sorted(instruction.params)),
                instruction.matrix is not None,
                bool(instruction.metadata.get("is_channel")),
            )
            for instruction in instructions
        ),
    )
    cached = _MPS_INSTRUCTION_SCHEDULE_CACHE.get(key)
    if cached is not None:
        return cached

    groups: list[tuple[int, ...]] = []
    index = 0
    while index < len(instructions):
        instruction = instructions[index]
        group = [index]
        if fuse_single_qubit and _is_fusible_one_qubit_instruction(instruction):
            wire = int(instruction.wires[0])
            next_index = index + 1
            while (
                next_index < len(instructions)
                and _is_fusible_one_qubit_instruction(instructions[next_index])
                and int(instructions[next_index].wires[0]) == wire
            ):
                group.append(next_index)
                next_index += 1
        elif (
            len(instruction.wires) == 2
            and abs(instruction.wires[0] - instruction.wires[1]) == 1
            and instruction.wires[0] < instruction.wires[1]
        ):
            occupied = set(instruction.wires)
            next_index = index + 1
            while next_index < len(instructions):
                candidate = instructions[next_index]
                if (
                    len(candidate.wires) != 2
                    or abs(candidate.wires[0] - candidate.wires[1]) != 1
                    or candidate.wires[0] > candidate.wires[1]
                    or occupied.intersection(candidate.wires)
                ):
                    break
                group.append(next_index)
                occupied.update(candidate.wires)
                next_index += 1
        groups.append(tuple(group))
        index = group[-1] + 1
    schedule = tuple(groups)
    _MPS_INSTRUCTION_SCHEDULE_CACHE[key] = schedule
    return schedule


def _is_fusible_one_qubit_instruction(instruction: Instruction) -> bool:
    return (
        len(instruction.wires) == 1
        and not instruction.metadata.get("is_channel")
        and instruction.name not in {"measure", "reset"}
    )


@dataclass(frozen=True)
class CompiledMPSOperation:
    instruction_indices: tuple[int, ...]
    kind: str
    wires: tuple[int, ...]


@dataclass(frozen=True)
class CompiledMPSProgram:
    """Shape-specialized immutable MPS execution schedule."""

    signature: tuple[Any, ...]
    operations: tuple[CompiledMPSOperation, ...]

    @classmethod
    def compile(
        cls,
        instructions: tuple[Instruction, ...],
        *,
        signature: tuple[Any, ...],
        fuse_single_qubit: bool,
    ) -> "CompiledMPSProgram":
        schedule = _mps_instruction_schedule(instructions, fuse_single_qubit)
        operations = []
        for group in schedule:
            instruction = instructions[group[0]]
            wires = tuple(instruction.wires)
            if len(group) > 1:
                kind = (
                    "adjacent_two_bucket"
                    if len(instruction.wires) == 2
                    else "fused_one"
                )
            elif len(wires) == 1:
                kind = "one"
            elif len(wires) == 2 and abs(wires[0] - wires[1]) == 1:
                kind = "adjacent_two"
            elif len(wires) == 2:
                kind = "remote_two"
            else:
                kind = "dense_fallback"
            operations.append(CompiledMPSOperation(group, kind, wires))
        return cls(signature=signature, operations=tuple(operations))


@dataclass(frozen=True)
class MPSTruncationRecord:
    """Per-bond truncation metadata emitted by MPS SVD updates."""

    bond: int
    kept_rank: int
    original_rank: int
    discarded_weight: float
    max_bond: int | None
    cutoff: float
    source: str = "two_site"

    def summary(self) -> dict[str, Any]:
        return {
            "bond": self.bond,
            "kept_rank": self.kept_rank,
            "original_rank": self.original_rank,
            "discarded_weight": self.discarded_weight,
            "max_bond": self.max_bond,
            "cutoff": self.cutoff,
            "source": self.source,
        }


@dataclass(frozen=True)
class MPSAdaptiveBondPlan:
    """Suggested bond growth plan derived from observed truncation records."""

    current_max_bond: int
    suggested_max_bond: int
    global_error_budget: float | None
    observed_error: float
    budget_satisfied: bool
    hot_bonds: tuple[int, ...]
    per_bond_suggestions: tuple[tuple[int, int], ...]

    def summary(self) -> dict[str, Any]:
        return {
            "current_max_bond": self.current_max_bond,
            "suggested_max_bond": self.suggested_max_bond,
            "global_error_budget": self.global_error_budget,
            "observed_error": self.observed_error,
            "budget_satisfied": self.budget_satisfied,
            "hot_bonds": self.hot_bonds,
            "per_bond_suggestions": self.per_bond_suggestions,
        }


@dataclass(frozen=True)
class MPSLocalRefinementPlan:
    """Local windows suggested for future adaptive MPS refinement."""

    windows: tuple[tuple[int, int], ...]
    hot_bonds: tuple[int, ...]
    suggested_max_bond: int
    observed_error: float

    def summary(self) -> dict[str, Any]:
        return {
            "windows": self.windows,
            "hot_bonds": self.hot_bonds,
            "suggested_max_bond": self.suggested_max_bond,
            "observed_error": self.observed_error,
        }


@dataclass(frozen=True)
class MPSAdaptiveRunResult:
    """Two-stage adaptive MPS execution result."""

    state: MPSState
    initial_state: MPSState
    initial_plan: MPSAdaptiveBondPlan
    final_plan: MPSAdaptiveBondPlan
    refinement_plan: MPSLocalRefinementPlan
    rerun: bool

    def summary(self) -> dict[str, Any]:
        state_summary = self.state.summary()
        state_summary.update(
            {
                "state_mode": "adaptive_mps",
                "rerun": self.rerun,
                "initial_plan": self.initial_plan.summary(),
                "final_plan": self.final_plan.summary(),
                "refinement_plan": self.refinement_plan.summary(),
            }
        )
        return state_summary

    def to_statevector(self) -> torch.Tensor:
        return self.state.to_statevector()

    statevector = to_statevector


@dataclass(frozen=True)
class MPSBondProfile:
    """Observable profile for an MPS state and its truncation budget."""

    n_wires: int
    batch_size: int
    bond_dims: tuple[int, ...]
    max_bond: int
    mean_bond: float
    parameter_count: int
    state_norm_min: float
    state_norm_max: float
    orthogonality_center: int | None
    left_canonical_residual: float
    right_canonical_residual: float
    mixed_canonical_residual: float
    truncation_steps: int
    truncation_error: float
    max_truncation_error: float
    truncation_by_bond: tuple[tuple[int, float], ...]
    truncation_records: tuple[MPSTruncationRecord, ...]
    error_budget_satisfied: bool | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "n_wires": self.n_wires,
            "batch_size": self.batch_size,
            "bond_dims": self.bond_dims,
            "max_bond": self.max_bond,
            "mean_bond": self.mean_bond,
            "parameter_count": self.parameter_count,
            "state_norm_min": self.state_norm_min,
            "state_norm_max": self.state_norm_max,
            "orthogonality_center": self.orthogonality_center,
            "left_canonical_residual": self.left_canonical_residual,
            "right_canonical_residual": self.right_canonical_residual,
            "mixed_canonical_residual": self.mixed_canonical_residual,
            "truncation_steps": self.truncation_steps,
            "truncation_error": self.truncation_error,
            "max_truncation_error": self.max_truncation_error,
            "truncation_by_bond": self.truncation_by_bond,
            "truncation_records": tuple(
                record.summary() for record in self.truncation_records
            ),
            "error_budget_satisfied": self.error_budget_satisfied,
        }


@dataclass(frozen=True)
class MPSMonteCarloResult:
    trajectories: tuple[MPSState, ...]
    expectation_z_mean: torch.Tensor
    expectation_z_variance: torch.Tensor
    n_trajectories: int
    trajectory_ids: tuple[int, ...] = ()
    trajectory_seeds: tuple[int, ...] = ()
    statistics: TrajectoryStatistics | None = None
    requested_trajectories: int | None = None
    retained_trajectory_ids: tuple[int, ...] = ()
    failures: tuple[TrajectoryFailure, ...] = ()
    rank: int = 0
    world_size: int = 1
    target_standard_error: float | None = None
    min_trajectories: int = 1
    converged: bool = False
    stopped_early: bool = False
    noise_model_identity: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "state_mode": "mps_trajectory",
            "n_trajectories": self.n_trajectories,
            "requested_trajectories": (
                self.n_trajectories
                if self.requested_trajectories is None
                else self.requested_trajectories
            ),
            "trajectory_ids": self.trajectory_ids,
            "trajectory_seeds": self.trajectory_seeds,
            "retained_trajectory_ids": self.retained_trajectory_ids,
            "retained_trajectory_count": len(self.trajectories),
            "failures": tuple(item.summary() for item in self.failures),
            "rank": self.rank,
            "world_size": self.world_size,
            "target_standard_error": self.target_standard_error,
            "min_trajectories": self.min_trajectories,
            "converged": self.converged,
            "stopped_early": self.stopped_early,
            "noise_model_identity": self.noise_model_identity,
            "expectation_z_mean": self.expectation_z_mean,
            "expectation_z_variance": self.expectation_z_variance,
            "expectation_z_standard_error": (
                None if self.statistics is None else self.statistics.standard_error
            ),
        }
