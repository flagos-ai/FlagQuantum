# Adapted from Qualtran v0.7.0, qualtran/bloqs/arithmetic/controlled_add_or_subtract.py.
# Copyright 2023 Google LLC (addition.py and and_bloq.py)
# Copyright 2024 Google LLC (controlled_add_or_subtract.py)
# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/quantumlib/Qualtran/blob/v0.7.0/qualtran/bloqs/arithmetic/controlled_add_or_subtract.py
# Sources: https://github.com/quantumlib/Qualtran/blob/v0.7.0/qualtran/bloqs/arithmetic/addition.py
#          https://github.com/quantumlib/Qualtran/blob/v0.7.0/qualtran/bloqs/mcmt/and_bloq.py
# AddK load/add/unload: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/addition.py
# Negate: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/negate.py
# Changes: emit native IR for register arithmetic and bind scratch explicitly.

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


def add_constant_ir(
    target: tuple[int, ...],
    value: int,
    constant: tuple[int, ...],
    carries: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """Port unsigned AddK: load a constant, add it, then erase the load.

    Big-endian target and clean constant registers have equal positive width n.
    Supply n-1 clean carry wires and distinct nonnegative classical slots.
    The target is incremented modulo 2**n; all quantum work returns to zero.
    Negative constants follow upstream unsigned normalization. Positive constants
    must fit n bits. No constant-specific gate simplification is performed.
    """
    width = len(target)
    if width < 1 or len(constant) != width:
        raise ValueError(
            "Constant addition requires equal nonempty target and load widths"
        )
    wires = (*target, *constant, *carries)
    if any(type(w) is not int or w < 0 for w in wires) or len(set(wires)) != len(wires):
        raise ValueError(
            "Constant addition requires distinct nonnegative quantum wires"
        )
    if type(value) is not int or value >= 1 << width:
        raise ValueError("Additive constant must be an integer smaller than 2**width")
    if any(type(bit) is not int or bit < 0 for bit in classical_bits):
        raise ValueError(
            "Constant addition requires nonnegative integer classical slots"
        )
    normalized = value % (1 << width)
    load = tuple(
        Instruction("x", (wire,))
        for i, wire in enumerate(constant)
        if normalized >> (width - i - 1) & 1
    )
    return (*load, *add_ir(constant, target, carries, classical_bits), *load)


def negate_ir(
    target: tuple[int, ...],
    constant: tuple[int, ...],
    carries: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """Two's-complement negation using the pinned BitwiseNot -> AddK(1) chain.

    Interpret the same n-bit permutation as signed negation or unsigned modular
    negation. Work ownership and cleanup follow add_constant_ir. Preserve the
    source network, including its constant loading and measured carries.
    """
    addition = add_constant_ir(target, 1, constant, carries, classical_bits)
    return (*(Instruction("x", (wire,)) for wire in target), *addition)
