"""Compile exact portable controlled-phase decompositions for CPU execution."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from numbers import Real
from typing import TypeAlias

from ...core.operator_schema import canonical_opcode
from .program import (
    _StatevectorControlledPhaseDecompositionStep,
    _StatevectorGateStep,
    _StatevectorRXRZLoopStep,
)

_ControlledPhaseStep: TypeAlias = (
    _StatevectorGateStep
    | _StatevectorRXRZLoopStep
    | _StatevectorControlledPhaseDecompositionStep
)


def _constant_rz_angle(step: _StatevectorGateStep) -> float | None:
    """Return a scalar constant RZ angle, declining every ambiguous case."""

    instruction = step.instruction
    if (
        instruction.matrix is not None
        or canonical_opcode(instruction.name) != "rz"
        or len(instruction.wires) != 1
    ):
        return None
    angle = instruction.params.get("theta")
    if isinstance(angle, bool) or not isinstance(angle, Real):
        return None
    real_angle = float(angle)
    return real_angle if isfinite(real_angle) else None


def _fuse_controlled_phase_decompositions(
    program: Sequence[_StatevectorGateStep | _StatevectorRXRZLoopStep],
) -> list[_ControlledPhaseStep]:
    """Fuse the exact ``RZ-RZ-CX-RZ-CX`` controlled-phase decomposition.

    Only literal scalar angles with exact equality are accepted. Parameterized
    and approximately matching sequences deliberately remain on the general
    path so this optimization cannot infer an algebraic relation that the IR
    does not prove.
    """

    optimized: list[_ControlledPhaseStep] = []
    index = 0
    while index < len(program):
        window = program[index : index + 5]
        gate_window = tuple(
            step for step in window if isinstance(step, _StatevectorGateStep)
        )
        if len(gate_window) == 5:
            first, second, first_cx, inverse, second_cx = gate_window
            first_angle = _constant_rz_angle(first)
            second_angle = _constant_rz_angle(second)
            inverse_angle = _constant_rz_angle(inverse)
            first_cx_instruction = first_cx.instruction
            second_cx_instruction = second_cx.instruction
            control = first.instruction.wires[0]
            target = second.instruction.wires[0]
            if (
                control != target
                and first_angle is not None
                and second_angle == first_angle
                and inverse_angle == -first_angle
                and inverse.instruction.wires == (target,)
                and first_cx_instruction.matrix is None
                and second_cx_instruction.matrix is None
                and canonical_opcode(first_cx_instruction.name) == "cx"
                and canonical_opcode(second_cx_instruction.name) == "cx"
                and first_cx_instruction.wires == (control, target)
                and second_cx_instruction.wires == (control, target)
            ):
                optimized.append(
                    _StatevectorControlledPhaseDecompositionStep(
                        control=control,
                        target=target,
                        half_angle=first_angle,
                    )
                )
                index += 5
                continue
        optimized.append(program[index])
        index += 1
    return optimized
