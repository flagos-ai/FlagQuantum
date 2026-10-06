"""Compiled local-statevector program step types."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

from ...core.ir import Instruction


@dataclass(frozen=True)
class _StatevectorGateStep:
    instruction: Instruction
    layout: tuple[tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class _StatevectorRXRZLoopStep:
    wire: int
    pairs: tuple[tuple[Instruction, Instruction], ...]


@dataclass(frozen=True)
class _StatevectorFusedGateStep:
    instructions: tuple[Instruction, ...]
    wires: tuple[int, ...]
    layout: tuple[tuple[int, ...], tuple[int, ...]]
    dependency_reordered: bool = False
    diagonal: bool = False


@dataclass(frozen=True)
class _StatevectorControlledPhaseDecompositionStep:
    """Exact diagonal replacement for the five-gate QFT phase sequence."""

    control: int
    target: int
    half_angle: float


@dataclass(frozen=True)
class _StatevectorControlledPhaseGraphStep:
    """One phase application for consecutive static controlled-phase edges."""

    edges: tuple[tuple[int, int, float], ...]


@dataclass(frozen=True)
class _StatevectorCrossWireDiagonalStep:
    regions: tuple[_StatevectorGateStep | _StatevectorFusedGateStep, ...]


@dataclass(frozen=True)
class _StatevectorCZGraphStep:
    """One exact phase application for a consecutive graph of CZ gates."""

    edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _StatevectorCliffordMatchingStep:
    """One native pass over a disjoint matching of exact CX and CZ gates."""

    controls: tuple[int, ...]
    targets: tuple[int, ...]
    cz_edges: tuple[tuple[int, int], ...]


_StatevectorDenseRegion: TypeAlias = _StatevectorGateStep | _StatevectorFusedGateStep


@dataclass(frozen=True)
class _StatevectorDisjointDenseStep:
    regions: tuple[_StatevectorDenseRegion, ...]
    native_preferred: bool = False
    native_parameterized: bool = False
    native_clifford: bool = False
    fused_cx_controls: tuple[int, ...] = ()
    fused_cx_targets: tuple[int, ...] = ()


@dataclass(frozen=True)
class _StatevectorCXSequenceStep:
    controls: tuple[int, ...]
    targets: tuple[int, ...]


_StatevectorPreCXStep: TypeAlias = (
    _StatevectorGateStep
    | _StatevectorRXRZLoopStep
    | _StatevectorFusedGateStep
    | _StatevectorControlledPhaseDecompositionStep
    | _StatevectorControlledPhaseGraphStep
    | _StatevectorCrossWireDiagonalStep
    | _StatevectorCZGraphStep
    | _StatevectorCliffordMatchingStep
    | _StatevectorDisjointDenseStep
)
_StatevectorProgramStep: TypeAlias = _StatevectorPreCXStep | _StatevectorCXSequenceStep


def _preallocated_batch_assembly_beneficial(
    program: Sequence[_StatevectorProgramStep],
) -> bool:
    """Avoid overlapping the final output with measured large mixed workspaces."""

    matrix_steps = tuple(
        region
        for step in program
        for region in (
            step.regions
            if isinstance(
                step,
                (_StatevectorCrossWireDiagonalStep, _StatevectorDisjointDenseStep),
            )
            else (step,)
        )
    )
    has_rotation_sequence = any(
        isinstance(step, _StatevectorFusedGateStep)
        and len(step.instructions) >= 2
        and all(item.name in {"rx", "ry", "rz"} for item in step.instructions)
        for step in matrix_steps
    )
    has_clifford_matching = any(
        isinstance(step, _StatevectorCliffordMatchingStep) for step in program
    )
    return not (has_rotation_sequence and has_clifford_matching)


def _static_product_state_initialization_enabled() -> bool:
    """Whether a complete static Clifford layer may initialize ``|0>``."""

    return os.getenv(
        "FQ_CPU_NATIVE_STATIC_PRODUCT_STATE_INITIALIZATION", "1"
    ).strip().lower() not in {"0", "false", "off", "no"}


def _native_zero_state_prefix_length(
    program: Sequence[_StatevectorProgramStep], width: int
) -> int:
    """Return the native prefix that prepares every qubit from ``|0>`` once."""

    occupied: set[int] = set()
    expected = set(range(width))
    for index, step in enumerate(program):
        if not isinstance(step, _StatevectorDisjointDenseStep) or not (
            step.native_parameterized
            or (step.native_clifford and _static_product_state_initialization_enabled())
        ):
            return 0
        step_wires = tuple(
            int(wire)
            for region in step.regions
            for wire in (
                region.instruction.wires
                if isinstance(region, _StatevectorGateStep)
                else region.wires
            )
        )
        if (
            len(step_wires) != len(step.regions)
            or len(set(step_wires)) != len(step_wires)
            or occupied.intersection(step_wires)
        ):
            return 0
        occupied.update(step_wires)
        if occupied == expected:
            return index + 1
    return 0


def _direct_batch_assembly_beneficial(
    program: Sequence[_StatevectorProgramStep], width: int
) -> bool:
    """Whether every window can begin in its owned final-result slice."""

    return bool(_native_zero_state_prefix_length(program, width)) and all(
        isinstance(step, _StatevectorCliffordMatchingStep)
        or (
            isinstance(step, _StatevectorDisjointDenseStep)
            and (step.native_parameterized or step.native_clifford)
        )
        for step in program
    )
