# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 GeneralizedQSP.decompose_from_registers.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/qsp/generalized_qsp.py

"""Emit the finite GQSP signal sequence from already computed phases."""

from collections.abc import Callable, Sequence

from ....core.ir import Instruction
from ..bits.bitwise import _validate_wires
from .su2_rotation import su2_rotation_ir

BoundUnitary = Callable[[int | None, bool], tuple[Instruction, ...]]


def generalized_qsp_ir(
    signal: int,
    operands: tuple[int, ...],
    theta: Sequence[float],
    phi: Sequence[float],
    lambd: float,
    *,
    unitary: BoundUnitary,
    negative_power: int = 0,
) -> tuple[Instruction, ...]:
    """Interleave signal rotations and a caller-bound coherent unitary.

    The callback receives the signal control value (None means uncontrolled)
    and the adjoint flag. It owns U's operand ordering and any workspace in
    operands, restoring that workspace after each call. Signal is disjoint
    from operands. Phases come from the numerical QSP recursion; this emitter
    neither computes nor certifies complementary polynomials.
    """
    _validate_wires((signal, *operands))
    if len(theta) < 2 or len(theta) != len(phi):
        raise ValueError("GQSP requires equal phase lengths of at least two")
    if type(negative_power) is not int or negative_power < 0:
        raise ValueError("GQSP negative_power must be a nonnegative integer")
    rotations = tuple(
        su2_rotation_ir(signal, t, p, lambd if i == 0 else 0.0)
        for i, (t, p) in enumerate(zip(theta, phi, strict=True))
    )
    allowed = set((signal, *operands))

    def apply(value: int | None, adjoint: bool) -> tuple[Instruction, ...]:
        gates = unitary(value, adjoint)
        if any(wire not in allowed for gate in gates for wire in gate.wires):
            raise ValueError("GQSP unitary used a wire outside its binding")
        return gates

    remaining = negative_power
    gates = list(rotations[0])
    for rotation in rotations[1:]:
        if remaining > 0:
            gates.extend(apply(1, True))
            remaining -= 1
        else:
            gates.extend(apply(0, False))
        gates.extend(rotation)
    for _ in range(remaining):
        gates.extend(apply(None, True))
    return tuple(gates)
