# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _HalfLinearDepthGreaterThan.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/comparison.py
# Changes: explicit logical outputs/sign work, shared register emitters and native IR.

"""Retain a difference and comparison flag for zero-T measured uncomputation."""

from ....core.ir import Instruction
from ..arithmetic.out_of_place_addition import out_of_place_add_ir
from ..bits.bitwise import bitwise_not_ir, xor_ir
from ..bits.sign_extension import sign_extend_ir


def half_greater_ir(
    left: tuple[int, ...],
    right: tuple[int, ...],
    difference: tuple[int, ...],
    target: int,
    signs: tuple[int, ...],
    *,
    signed: bool = False,
    adjoint: bool = False,
    classical_bits: tuple[int, ...] = (),
) -> tuple[Instruction, ...]:
    """Compute (b-a mod 2**(n+1), a>b), or clear the correct computed pair.

    Equal positive-width operands and outputs are big-endian. Difference has
    n+1 qubits; target and difference are initially zero in forward mode. Two
    zero sign work qubits are always restored. Operands are preserved. Measured
    adjoint requires the correct logical pair and n distinct classical slots;
    it restores both outputs to zero without more quantum scratch.
    """
    n = len(left)
    if type(signed) is not bool or type(adjoint) is not bool:
        raise ValueError("Half comparison flags must be boolean")
    if n < 1 or len(right) != n or len(difference) != n + 1 or len(signs) != 2:
        raise ValueError(
            "Half comparison requires equal positive inputs, n+1 difference and two sign wires"
        )
    wires = (*left, *right, *difference, target, *signs)
    if any(type(w) is not int or w < 0 for w in wires) or len(set(wires)) != len(wires):
        raise ValueError("Half comparison requires distinct nonnegative wires")
    c = n if adjoint else 0
    if (
        len(classical_bits) != c
        or len(set(classical_bits)) != c
        or any(type(bit) is not int or bit < 0 for bit in classical_bits)
    ):
        raise ValueError(
            f"Half comparison requires {c} distinct nonnegative classical slots"
        )
    a, b = (signs[0], *left), (signs[1], *right)
    extension = (
        (*sign_extend_ir(left, signs[:1]), *sign_extend_ir(right, signs[1:]))
        if signed
        else ()
    )
    complement_b = bitwise_not_ir(b)
    complement_c = bitwise_not_ir(difference)
    copy = xor_ir(difference[:1], (target,))
    addition = out_of_place_add_ir(
        a,
        b,
        difference,
        include_msb=False,
        adjoint=adjoint,
        classical_bits=classical_bits,
    )
    middle = (
        (*copy, *complement_c, *addition)
        if adjoint
        else (*addition, *complement_c, *copy)
    )
    return (*extension, *complement_b, *middle, *complement_b, *extension)
