# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _DirtyOutOfPlaceMontgomeryModMulImpl.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_multiplication.py

"""Windowed Montgomery source composition with explicit result and work aliases."""

from ....core.ir import Instruction
from ..arithmetic.controlled_addition import (
    controlled_add_constant_adjoint_ir,
    controlled_add_constant_ir,
)
from ..bits.bitwise import _validate_wires
from ..comparison.less_than import less_than_constant_ir
from .single_window_mod_mul import single_window_mod_mul_ir, single_window_qroam_table


def windowed_mod_mul_ir(
    x: tuple[int, ...],
    y: tuple[int, ...],
    target: tuple[int, ...],
    indices: tuple[int, ...],
    reduced: int,
    modulus: int,
    padding: tuple[int, ...],
    bank: tuple[tuple[int, ...], ...],
    work: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    adjoint: bool = False,
) -> tuple[tuple[Instruction, ...], tuple[int, ...]]:
    """Emit the source circuit and return the physical order of its output word.

    x, y, target and each retained lookup index are big-endian. The index
    chunks are ordered from the least-significant multiplier window upward.
    The forward path starts with target, indices, reduced, padding, bank and
    work clean. Result bits occupy target[w:] followed by padding. The first
    w qubits of target become clean work. Both directions accept the original
    layout; the adjoint reads the result and retained outputs in that layout.

    Width w=len(padding) divides n=len(x); odd p>1 requires 2*p-1 < 2**n.
    This is the literal source +p final stage. On a set reduction flag it
    can disagree with the mathematical residue. The inverse restores the
    actual forward image, including the index and flag initialized to zero.
    """
    n, w = len(x), len(padding)
    single_window_qroam_table(n, w, modulus)
    if modulus <= 1 or 2 * modulus - 1 >= 1 << n:
        raise ValueError("Windowed multiplication requires 1 < p and 2*p-1 < 2**n")
    if len(y) != n or len(target) != n or len(indices) != n:
        raise ValueError("Windowed multiplication requires four equal-width words")
    bank_wires = tuple(wire for word in bank for wire in word)
    _validate_wires((*x, *y, *target, *indices, reduced, *padding, *bank_wires, *work))
    if (
        type(adjoint) is not bool
        or len(classical_bits) != 2 * n
        or len(set(classical_bits)) != len(classical_bits)
        or any(type(bit) is not int or bit < 0 for bit in classical_bits)
    ):
        raise ValueError("Windowed multiplication requires 2n distinct classical slots")

    original = (*padding, *target)
    rotated = (*target, *padding)
    result = rotated[w:]
    accumulator = rotated if adjoint else original
    multiplier = x[::-1]
    windows: list[Instruction] = []
    starts = range(0, n, w)
    for i in reversed(starts) if adjoint else starts:
        chunk, accumulator = single_window_mod_mul_ir(
            multiplier[i : i + w],
            y,
            accumulator,
            indices[i : i + w],
            modulus,
            bank,
            work,
            classical_bits,
            adjoint=adjoint,
        )
        windows.extend(chunk)

    pool = (*work, *bank_wires)
    comparison = less_than_constant_ir(
        result, modulus, reduced, pool[: n + 1], classical_bits[:n]
    )
    add = controlled_add_constant_adjoint_ir if adjoint else controlled_add_constant_ir
    reduction = add(
        reduced, result, modulus, pool[:n], pool[n : 2 * n - 1], classical_bits[: n - 1]
    )
    flip = Instruction("x", (reduced,))
    if adjoint:
        return (*reduction, *comparison, flip, *windows), accumulator[w:]
    return (flip, *windows, *comparison, *reduction), accumulator[w:]
