# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 givens_bloq.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/quad_fermion/givens_bloq.py

"""Lower the phase-gradient Givens Clifford ladder and its phase rotations."""

from ....core._composition import inverted_instruction
from ....core.ir import Instruction
from ..bits.bitwise import _validate_wires
from .rz_phase_gradient import rz_via_phase_gradient_ir


def gradient_givens_ir(
    i: int,
    j: int,
    real_angle: tuple[int, ...],
    gradient: tuple[int, ...],
    carries: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    complex_angle: tuple[int, ...] | None = None,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Use big-endian fractional angle words and a negative-exponent gradient.

    All logical ports are through. Each Rz returns all carries to zero.
    The classical slots are reused by the next Rz in the same operation.
    """
    extra_angle = () if complex_angle is None else complex_angle
    _validate_wires((i, j, *real_angle, *extra_angle, *gradient, *carries))
    width = len(gradient)
    if (
        not width
        or len(real_angle) != width
        or (complex_angle is not None and len(complex_angle) != width)
        or len(carries) != width - 1
        or len(classical_bits) != width - 1
        or len(set(classical_bits)) != len(classical_bits)
    ):
        raise ValueError(
            "Givens requires matching angle and gradient widths and n-1 work slots"
        )
    if type(adjoint) is not bool:
        raise ValueError("Givens direction must be boolean")

    x = Instruction("x", (i,))
    s = Instruction("s", (j,))
    sdg_i = Instruction("sdg", (i,))
    s_i = Instruction("s", (i,))
    h = Instruction("h", (j,))
    cx = Instruction("cx", (j, i))
    reverse_cx = Instruction("cx", (i, j))
    before: tuple[Instruction, ...] = (x, s, cx, h, sdg_i, cx)
    after: tuple[Instruction, ...] = (cx, x, reverse_cx, h, cx, s_i)
    targets = (i, j)
    if adjoint:
        before, after = (
            tuple(inverted_instruction(gate) for gate in reversed(after)),
            tuple(inverted_instruction(gate) for gate in reversed(before)),
        )
        targets = (j, i)

    rotations = tuple(
        gate
        for target in targets
        for gate in rz_via_phase_gradient_ir(
            target, real_angle, gradient, carries, classical_bits, adjoint=adjoint
        )
    )
    real = (*before, *rotations, *after)
    if complex_angle is None:
        return real
    complex_phase = rz_via_phase_gradient_ir(
        j, complex_angle, gradient, carries, classical_bits, adjoint=adjoint
    )
    return (*complex_phase, *real) if adjoint else (*real, *complex_phase)
