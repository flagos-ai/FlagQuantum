# Adapted from Qualtran v0.7.0, qualtran/bloqs/arithmetic/controlled_add_or_subtract.py.
# Copyright 2023 Google LLC (addition.py and and_bloq.py)
# Copyright 2024 Google LLC (controlled_add_or_subtract.py)
# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/quantumlib/Qualtran/blob/v0.7.0/qualtran/bloqs/arithmetic/controlled_add_or_subtract.py
# Sources: https://github.com/quantumlib/Qualtran/blob/v0.7.0/qualtran/bloqs/arithmetic/addition.py
#          https://github.com/quantumlib/Qualtran/blob/v0.7.0/qualtran/bloqs/mcmt/and_bloq.py
# Changes: emit native IR for unsigned addition and bind scratch explicitly.

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
    """Lower Qualtran Add for unsigned big-endian registers."""
    width = len(right)
    if not 1 <= len(left) <= width or len(carries) != width - 1:
        raise ValueError(
            "Add requires a nonempty input no wider than the output and one clean carry per lower bit"
        )
    if len(classical_bits) != len(carries) or len(set(classical_bits)) != len(carries):
        raise ValueError("Add requires one distinct classical bit per carry")

    a, b = left[::-1], right[::-1]
    instructions: list[Instruction] = []

    def cx(control: int, target: int) -> None:
        instructions.append(Instruction("cx", (control, target)))

    def compute(x: int, y: int, target: int) -> None:
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

    def uncompute(x: int, y: int, target: int, bit: int) -> None:
        instructions.extend(
            (
                Instruction("h", (target,)),
                Instruction(
                    "measure",
                    (target,),
                    metadata={"is_dynamic": True, "classical_bit": bit},
                ),
                Instruction("cz", (x, y), metadata={"conditions": ((bit, 1),)}),
                Instruction("reset", (target,), metadata={"is_dynamic": True}),
            )
        )

    if width == 1:
        cx(a[0], b[0])
        return tuple(instructions)

    compute(a[0], b[0], carries[0])
    for depth in range(1, width - 1):
        previous = carries[depth - 1]
        control = previous
        if depth < len(a):
            control = a[depth]
            cx(previous, control)
        cx(previous, b[depth])
        compute(control, b[depth], carries[depth])
        cx(previous, carries[depth])
    cx(carries[-1], b[-1])
    if len(a) == width:
        cx(a[-1], b[-1])
    for depth in range(width - 2, 0, -1):
        previous = carries[depth - 1]
        cx(previous, carries[depth])
        control = a[depth] if depth < len(a) else previous
        uncompute(control, b[depth], carries[depth], classical_bits[depth])
        if depth < len(a):
            cx(previous, control)
            cx(control, b[depth])
    uncompute(a[0], b[0], carries[0], classical_bits[0])
    cx(a[0], b[0])
    return tuple(instructions)


def measurement_add_or_subtract_ir(
    control: int,
    left: tuple[int, ...],
    right: tuple[int, ...],
    carries: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    add_when_ctrl_is_on: bool = False,
) -> tuple[Instruction, ...]:
    """Add or subtract according to the control bit, using measured carries."""
    complement = tuple(Instruction("cx", (control, wire)) for wire in right)
    flip = (Instruction("x", (control,)),) if add_when_ctrl_is_on else ()
    return (
        *flip,
        *complement,
        *add_ir(left, right, carries, classical_bits),
        *complement,
        *flip,
    )
