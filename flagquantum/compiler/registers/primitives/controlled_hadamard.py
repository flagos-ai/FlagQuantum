# Copyright 2018 The Cirq Developers
# SPDX-License-Identifier: Apache-2.0
# Adapted from Cirq 1.7 HPowGate, EigenGate phase extraction, GlobalPhaseGate
# control dispatch, DiagonalGate and CCZPowGate as called by Qualtran v0.7.0.

"""Retain the source controlled-H front end and its exact scalar phases."""

from cmath import phase
from math import pi

from ....core.ir import Instruction
from ...optimization.controlled_unitary_synthesis import _phase, controlled_unitary_ir
from ..bits.bitwise import _validate_wires
from .global_phase import scalar_phase_ir


def _controlled_matrix(
    matrix: list[list[complex]],
    controls: tuple[int, ...],
    values: tuple[int, ...],
    target: int,
) -> tuple[Instruction, ...]:
    flips = tuple(
        Instruction("x", (q,)) for q, cv in zip(controls, values, strict=True) if not cv
    )
    return (*flips, *controlled_unitary_ir(matrix, controls, target), *flips)


def _diagonal_phase(
    controls: tuple[int, ...], values: tuple[int, ...], angle: float
) -> tuple[Instruction, ...]:
    # The source's zero-last-control global scalar falls back to a DiagonalGate.
    # Walsh/Gray traversal emits no whole-block matrix and borrows no workspace.
    n = len(controls)
    hot = sum(cv << (n - i - 1) for i, cv in enumerate(values))
    coefficients = [angle if i == hot else 0.0 for i in range(1 << n)]
    step = 1
    while step < len(coefficients):
        for offset in range(0, len(coefficients), 2 * step):
            for j in range(offset, offset + step):
                x, y = coefficients[j], coefficients[j + step]
                coefficients[j], coefficients[j + step] = x + y, x - y
        step *= 2
    coefficients = [v / (1 << n) for v in coefficients]
    coefficient = 1j ** (2 * coefficients[0] / pi)
    gates = list(scalar_phase_ir(controls[0], phase(coefficient)))
    gray = 0
    for i in range(1, (1 << n) + 1):
        next_gray = (i ^ (i >> 1)) if i < (1 << n) else 0
        flip = (gray ^ next_gray).bit_length() - 1
        if gray:
            largest = n - gray.bit_length()
            gates.append(
                Instruction(
                    "rz", (controls[largest],), {"theta": -2 * coefficients[gray]}
                )
            )
            other = n - flip - 1
            if other != largest:
                left, right = sorted((largest, other))
                gates.append(Instruction("cx", (controls[right], controls[left])))
        gray = next_gray
    return tuple(gates)


def _ccz_phase(controls: tuple[int, ...], exponent: float) -> tuple[Instruction, ...]:
    a, b, c = controls
    # Follow the source LineQubit adjacency choice before the fixed CX sweep.
    if abs(b - a) != 1:
        b, c = c, b
    elif abs(b - c) != 1:
        a, b = b, a
    p = exponent / 4
    sweep = (Instruction("cx", (a, b)), Instruction("cx", (b, c)))
    return (
        _phase(a, p),
        _phase(b, p),
        _phase(c, p),
        *sweep,
        _phase(b, -p),
        _phase(c, p),
        *sweep,
        _phase(c, -p),
        *sweep,
        _phase(c, -p),
        *sweep,
    )


def _control_phase(
    controls: tuple[int, ...], values: tuple[int, ...]
) -> tuple[Instruction, ...]:
    coefficient = 1j**-0.5
    angle = phase(coefficient)
    if not values[-1]:
        return _diagonal_phase(controls, values, angle)
    exponent = angle / pi
    if all(values):
        if len(controls) == 2:
            return (Instruction("cphase", controls, {"theta": exponent * pi}),)
        if len(controls) == 3:
            return _ccz_phase(controls, exponent)
    matrix = [[1 + 0j, 0j], [0j, coefficient]]
    return _controlled_matrix(matrix, controls[:-1], values[:-1], controls[-1])


def _controlled_y(
    controls: tuple[int, ...], values: tuple[int, ...], target: int, exponent: float
) -> tuple[Instruction, ...]:
    factor = 1j ** (2 * exponent)
    a, b = (1 + factor) / 2, (1 - factor) / 2
    if values == (1, 1):
        return (
            Instruction("sdg", (target,)),
            Instruction("h", (target,)),
            *_ccz_phase((*controls, target), exponent),
            Instruction("h", (target,)),
            Instruction("s", (target,)),
        )
    return _controlled_matrix([[a, -1j * b], [1j * b, a]], controls, values, target)


def controlled_hadamard_ir(
    controls: tuple[int, ...],
    target: int,
    *,
    control_values: tuple[int, ...] | None = None,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Emit the actual source H control dispatch at its local matrix leaf level.

    Wires are distinct nonnegative native indices. Controls are ordered and
    arbitrary zero/one polarities are supported. No work or result slots are
    used. A single positive control retains canonical CH; the other paths keep
    source half-Y, shifted-X, extracted control phase and recursion order.
    """
    _validate_wires((*controls, target))
    if type(adjoint) is not bool:
        raise ValueError("Controlled Hadamard direction must be Boolean")
    values = (1,) * len(controls) if control_values is None else control_values
    if len(values) != len(controls) or any(
        type(cv) is not int or cv not in (0, 1) for cv in values
    ):
        raise ValueError("Controlled Hadamard requires one binary value per control")
    if not controls:
        return (Instruction("h", (target,)),)
    if values == (1,):
        return (Instruction("ch", (*controls, target)),)
    y_part = _controlled_y(controls, values, target, 0.25 if adjoint else 0.5)
    x_part: tuple[Instruction, ...]
    if all(values) and len(controls) <= 2:
        x_part = (Instruction(("cx", "ccx")[len(controls) - 1], (*controls, target)),)
    else:
        x_part = _controlled_matrix(
            [[0j, 1 + 0j], [1 + 0j, 0j]], controls, values, target
        )
    if adjoint:
        return (*y_part, *x_part, *_controlled_y(controls, values, target, -0.25))
    return (*y_part, *x_part, *_control_phase(controls, values))
