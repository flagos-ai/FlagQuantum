# Adapted from Qualtran v0.7.0, qualtran/bloqs/arithmetic/controlled_add_or_subtract.py.
# Copyright 2023 Google LLC (addition.py and and_bloq.py)
# Copyright 2024 Google LLC (controlled_add_or_subtract.py)
# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/quantumlib/Qualtran/blob/v0.7.0/qualtran/bloqs/arithmetic/controlled_add_or_subtract.py
# Sources: https://github.com/quantumlib/Qualtran/blob/v0.7.0/qualtran/bloqs/arithmetic/addition.py
#          https://github.com/quantumlib/Qualtran/blob/v0.7.0/qualtran/bloqs/mcmt/and_bloq.py
# Changes: emit native IR for equal-width unsigned addition and bind scratch explicitly.

"""Register-level arithmetic decompositions."""

from ...core.ir import Instruction
from .._arithmetic_kernels import ripple_add_modular


def controlled_add_or_subtract_ir(
    control: int, angle: tuple[int, ...], gradient: tuple[int, ...], carry: int
) -> tuple[Instruction, ...]:
    """Add on zero and subtract on one by conjugating with controlled NOT."""
    complement = tuple(Instruction("cx", (control, wire)) for wire in gradient)
    return (*complement, *ripple_add_modular(angle, gradient, carry), *complement)


def add_ir(
    left: tuple[int, ...],
    right: tuple[int, ...],
    carries: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """Lower Qualtran Add for equal-width unsigned big-endian registers."""
    width = len(right)
    if not width or len(left) != width or len(carries) != width - 1:
        raise ValueError(
            "Add requires equal-width inputs and one clean carry per lower bit"
        )
    if len(classical_bits) != len(carries) or len(set(classical_bits)) != len(carries):
        raise ValueError("Add requires one distinct classical bit per carry")

    a, b = left[::-1], right[::-1]
    instructions: list[Instruction] = []

    def cx(control: int, target: int) -> None:
        instructions.append(Instruction("cx", (control, target)))

    def compute(depth: int) -> None:
        x, y, target = a[depth], b[depth], carries[depth]
        instructions.extend(
            (
                Instruction("h", (target,)),
                Instruction("t", (target,)),
                Instruction("cx", (x, target)),
                Instruction("cx", (y, target)),
                Instruction("cx", (target, x)),
                Instruction("cx", (target, y)),
                Instruction("tdg", (x,)),
                Instruction("tdg", (y,)),
                Instruction("t", (target,)),
                Instruction("cx", (target, x)),
                Instruction("cx", (target, y)),
                Instruction("h", (target,)),
                Instruction("s", (target,)),
            )
        )

    def uncompute(depth: int) -> None:
        target, bit = carries[depth], classical_bits[depth]
        instructions.extend(
            (
                Instruction("h", (target,)),
                Instruction(
                    "measure",
                    (target,),
                    metadata={"is_dynamic": True, "classical_bit": bit},
                ),
                Instruction(
                    "cz", (a[depth], b[depth]), metadata={"conditions": ((bit, 1),)}
                ),
                Instruction("reset", (target,), metadata={"is_dynamic": True}),
            )
        )

    if width == 1:
        cx(a[0], b[0])
        return tuple(instructions)

    compute(0)
    for depth in range(1, width - 1):
        previous = carries[depth - 1]
        cx(previous, a[depth])
        cx(previous, b[depth])
        compute(depth)
        cx(previous, carries[depth])
    cx(carries[-1], b[-1])
    cx(a[-1], b[-1])
    for depth in range(width - 2, 0, -1):
        previous = carries[depth - 1]
        cx(previous, carries[depth])
        uncompute(depth)
        cx(previous, a[depth])
        cx(a[depth], b[depth])
    uncompute(0)
    cx(a[0], b[0])
    return tuple(instructions)


def measurement_add_or_subtract_ir(
    control: int,
    left: tuple[int, ...],
    right: tuple[int, ...],
    carries: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """Lower Qualtran ControlledAddOrSubtract with add on zero."""
    complement = tuple(Instruction("cx", (control, wire)) for wire in right)
    return (*complement, *add_ir(left, right, carries, classical_bits), *complement)
