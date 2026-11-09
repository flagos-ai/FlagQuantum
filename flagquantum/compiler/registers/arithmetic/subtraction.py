# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Subtract, SubtractFrom and MultiTargetCNOT.
# Sources: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/subtraction.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/multi_target_cnot.py
# Changes: native IR, explicit clean workspace and concrete bit-pattern semantics.

"""Two subtraction directions using the shared measured register adder."""

from ....core.ir import Instruction
from ..bits.sign_extension import sign_extend_ir
from .addition import add_ir


def _validate_subtraction(
    left: tuple[int, ...],
    right: tuple[int, ...],
    prefix: tuple[int, ...],
    carries: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> None:
    if not 1 <= len(left) <= len(right) or len(prefix) != len(right) - len(left):
        raise ValueError(
            "Subtraction requires nonempty m<=n inputs and n-m prefix wires"
        )
    wires = (*left, *right, *prefix, *carries)
    if any(type(w) is not int or w < 0 for w in wires) or len(set(wires)) != len(wires):
        raise ValueError("Subtraction requires distinct nonnegative quantum wires")
    if any(type(bit) is not int or bit < 0 for bit in classical_bits):
        raise ValueError("Subtraction requires nonnegative integer classical slots")


def subtract_ir(
    left: tuple[int, ...],
    right: tuple[int, ...],
    prefix: tuple[int, ...],
    carries: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    signed: bool = False,
) -> tuple[Instruction, ...]:
    """Preserve a and replace b with a-b modulo the right register width.

    Big-endian a may be narrower than b. Supply n-m clean prefix qubits for
    zero/sign extension and n-1 clean carry qubits with distinct classical slots.
    All quantum work returns to zero. signed controls a's extension only; the
    output bit pattern can be interpreted in either signed or unsigned form.
    """
    _validate_subtraction(left, right, prefix, carries, classical_bits)
    if type(signed) is not bool:
        raise ValueError("Subtraction signed flag must be a boolean")
    extended = (*prefix, *left)
    addition = add_ir(extended, right, carries, classical_bits)
    extension = sign_extend_ir(left, prefix) if signed and prefix else ()
    flip_left = tuple(Instruction("x", (wire,)) for wire in extended)
    flip_right = tuple(Instruction("x", (wire,)) for wire in right)
    return (*extension, *flip_left, *addition, *flip_right, *flip_left, *extension)


def subtract_from_ir(
    left: tuple[int, ...],
    right: tuple[int, ...],
    carries: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """Preserve a and replace b with b-a on equal-width big-endian registers.

    Carry ownership and measured cleanup follow add_ir. This is the upstream
    SubtractFrom chain, not a remapped or optimized version of Subtract.
    """
    _validate_subtraction(left, right, (), carries, classical_bits)
    addition = add_ir(left, right, carries, classical_bits)
    complement = tuple(Instruction("x", (wire,)) for wire in right)
    return (*complement, *addition, *complement)
