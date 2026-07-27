"""Native matrix-product-state execution for FlagQuantum."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ..core.ir import Instruction

_DENSE_Z_SUM_WEIGHT_CACHE: dict[
    tuple[int, tuple[int, ...], str, torch.dtype], torch.Tensor
] = {}
_MPS_INSTRUCTION_SCHEDULE_CACHE: dict[tuple[Any, ...], tuple[tuple[int, ...], ...]] = {}


@dataclass(frozen=True)
class MPSConfig:
    max_bond: int | None = None
    cutoff: float = 0.0
    dense_observable_wires: int = 12
    svd_driver: str | None = "gesvd"


if TYPE_CHECKING:
    from .mps_state import MPSState


def _mps_instruction_schedule(
    instructions: tuple[Instruction, ...], fuse_single_qubit: bool
) -> tuple[tuple[int, ...], ...]:
    from .mps_execution import _mps_instruction_schedule as schedule

    return schedule(instructions, fuse_single_qubit)


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

    def summary(self) -> dict[str, Any]:
        return {
            "state_mode": "mps_trajectory",
            "n_trajectories": self.n_trajectories,
            "expectation_z_mean": self.expectation_z_mean,
            "expectation_z_variance": self.expectation_z_variance,
        }
