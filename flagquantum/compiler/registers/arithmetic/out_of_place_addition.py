# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 OutOfPlaceAdder.decompose_from_registers.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/addition.py
# Changes: native IR and shared And; logical output and measurement slots explicit.

"""Out-of-place unsigned addition and its measured source adjoint."""

from ....core.ir import Instruction
from ..primitives.and_compute import and_compute_ir, and_uncompute_ir
from .out_of_place_addition_structure import out_of_place_addition_components


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
    slots = dict(zip(output[::-1][1:], classical_bits, strict=False))
    gates: list[Instruction] = []
    for kind, operands in out_of_place_addition_components(
        left, right, output, include_msb=include_msb, adjoint=adjoint
    ):
        if kind == "cx":
            gates.append(Instruction("cx", operands))
        elif kind == "and":
            gates.extend(and_compute_ir(operands[0], operands[1], operands[2]))
        else:
            gates.extend(
                and_uncompute_ir(
                    operands[0], operands[1], operands[2], slots[operands[2]]
                )
            )
    return tuple(gates)
