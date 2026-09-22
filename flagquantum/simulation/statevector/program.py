"""Compiled local-statevector program step types."""

from __future__ import annotations

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
class _StatevectorCrossWireDiagonalStep:
    regions: tuple[_StatevectorGateStep | _StatevectorFusedGateStep, ...]


_StatevectorDenseRegion: TypeAlias = _StatevectorGateStep | _StatevectorFusedGateStep


@dataclass(frozen=True)
class _StatevectorDisjointDenseStep:
    regions: tuple[_StatevectorDenseRegion, ...]


@dataclass(frozen=True)
class _StatevectorCXSequenceStep:
    controls: tuple[int, ...]
    targets: tuple[int, ...]


_StatevectorPreCXStep: TypeAlias = (
    _StatevectorGateStep
    | _StatevectorRXRZLoopStep
    | _StatevectorFusedGateStep
    | _StatevectorCrossWireDiagonalStep
    | _StatevectorDisjointDenseStep
)
_StatevectorProgramStep: TypeAlias = _StatevectorPreCXStep | _StatevectorCXSequenceStep
