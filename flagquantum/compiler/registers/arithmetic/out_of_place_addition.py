# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 OutOfPlaceAdder.decompose_from_registers.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/addition.py
# Changes: native IR and shared And; logical output and measurement slots explicit.

"""Out-of-place unsigned addition and its measured source adjoint."""

from ....core.ir import Instruction
from ..primitives.and_compute import and_compute_ir, and_uncompute_ir


def out_of_place_add_ir(
    left: tuple[int, ...],
    right: tuple[int, ...],
    output: tuple[int, ...],
    *,
    include_msb: bool = True,
    adjoint: bool = False,
    classical_bits: tuple[int, ...] = (),
) -> tuple[Instruction, ...]:
    """Compute a clean sum output, or clear a correct sum with measured adjoint.

    Big-endian inputs have width n. Output has n+1 bits with MSB, otherwise n.
    No separate quantum work is used. The output is logical data after compute,
    not clean scratch; adjoint requires its correct sum and clears it entirely.
    """
    if type(include_msb) is not bool or type(adjoint) is not bool:
        raise ValueError("Adder flags must be boolean")
    n = len(left)
    if n < 1 or len(right) != n or len(output) != n + int(include_msb):
        raise ValueError(
            "Out-of-place addition requires equal positive inputs and matching output width"
        )
    wires = (*left, *right, *output)
    if any(type(w) is not int or w < 0 for w in wires) or len(set(wires)) != len(wires):
        raise ValueError(
            "Out-of-place addition requires distinct nonnegative quantum wires"
        )
    count = len(output) - 1
    measured = count if adjoint else 0
    if (
        len(classical_bits) != measured
        or len(set(classical_bits)) != measured
        or any(type(b) is not int or b < 0 for b in classical_bits)
    ):
        raise ValueError(
            f"Adder requires {measured} distinct nonnegative classical slots"
        )
    a, b, c = left[::-1], right[::-1], output[::-1]
    blocks = []
    for i in range(count):
        if adjoint:
            blocks.append(
                (
                    Instruction("cx", (b[i], c[i])),
                    Instruction("cx", (a[i], c[i + 1])),
                    Instruction("cx", (a[i], b[i])),
                    *and_uncompute_ir(b[i], c[i], c[i + 1], classical_bits[i]),
                    Instruction("cx", (a[i], c[i])),
                    Instruction("cx", (a[i], b[i])),
                )
            )
        else:
            blocks.append(
                (
                    Instruction("cx", (a[i], b[i])),
                    Instruction("cx", (a[i], c[i])),
                    *and_compute_ir(b[i], c[i], c[i + 1]),
                    Instruction("cx", (a[i], b[i])),
                    Instruction("cx", (a[i], c[i + 1])),
                    Instruction("cx", (b[i], c[i])),
                )
            )
    top = (
        (Instruction("cx", (a[-1], c[-1])), Instruction("cx", (b[-1], c[-1])))
        if not include_msb
        else ()
    )
    if adjoint:
        return (*reversed(top), *(g for block in reversed(blocks) for g in block))
    return (*(g for block in blocks for g in block), *top)
