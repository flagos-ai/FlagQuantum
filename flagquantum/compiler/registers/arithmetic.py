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
from .and_compute import and_compute_ir, and_uncompute_ir
from .bitwise import bitwise_not_ir, controlled_xor_constant_ir, xor_constant_ir


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
    return _addition_ir(left, right, carries, classical_bits, adjoint=False)


def add_adjoint_ir(
    left: tuple[int, ...],
    right: tuple[int, ...],
    carries: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """Port the exact Add adjoint, b-=a, with measured borrow-prefix cleanup.

    Big-endian unsigned registers have 1 <= len(a) <= len(b). All n-1 caller
    carry wires start and end in zero, with one distinct classical slot each.
    Preserve the source component inverse, not a different subtraction network.
    """
    wires = (*left, *right, *carries)
    if any(type(w) is not int or w < 0 for w in wires) or len(set(wires)) != len(wires):
        raise ValueError("Add adjoint requires distinct nonnegative quantum wires")
    return _addition_ir(left, right, carries, classical_bits, adjoint=True)


def _addition_ir(
    left: tuple[int, ...],
    right: tuple[int, ...],
    carries: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    adjoint: bool,
) -> tuple[Instruction, ...]:
    """One source component sequence shared by Add and its abstract inverse."""
    width = len(right)
    if not 1 <= len(left) <= width or len(carries) != width - 1:
        raise ValueError(
            "Add requires a nonempty input no wider than the output and one clean carry per lower bit"
        )
    if len(classical_bits) != len(carries) or len(set(classical_bits)) != len(carries):
        raise ValueError("Add requires one distinct classical bit per carry")

    a, b = left[::-1], right[::-1]
    blocks: list[tuple[Instruction, ...]] = []

    def cx(control: int, target: int) -> None:
        blocks.append((Instruction("cx", (control, target)),))

    def logical_and(x: int, y: int, i: int, *, erase: bool = False) -> None:
        # Invert abstract components before expanding their measured gates.
        if erase ^ adjoint:
            block = and_uncompute_ir(x, y, carries[i], classical_bits[i])
        else:
            block = and_compute_ir(x, y, carries[i])
        blocks.append(block)

    if width == 1:
        cx(a[0], b[0])
        return tuple(g for block in blocks for g in block)

    logical_and(a[0], b[0], 0)
    for depth in range(1, width - 1):
        previous = carries[depth - 1]
        control = previous
        if depth < len(a):
            control = a[depth]
            cx(previous, control)
        cx(previous, b[depth])
        logical_and(control, b[depth], depth)
        cx(previous, carries[depth])
    cx(carries[-1], b[-1])
    if len(a) == width:
        cx(a[-1], b[-1])
    for depth in range(width - 2, 0, -1):
        previous = carries[depth - 1]
        cx(previous, carries[depth])
        control = a[depth] if depth < len(a) else previous
        logical_and(control, b[depth], depth, erase=True)
        if depth < len(a):
            cx(previous, control)
            cx(control, b[depth])
    logical_and(a[0], b[0], 0, erase=True)
    cx(a[0], b[0])
    ordered = reversed(blocks) if adjoint else iter(blocks)
    return tuple(g for block in ordered for g in block)


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
    return _constant_addition_ir(
        target, value, constant, carries, classical_bits, control=None
    )


def _constant_addition_ir(
    target: tuple[int, ...],
    value: int,
    constant: tuple[int, ...],
    carries: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    control: int | None,
) -> tuple[Instruction, ...]:
    """One load/add/unload implementation for plain and controlled AddK."""
    width = len(target)
    if width < 1 or len(constant) != width:
        raise ValueError(
            "Constant addition requires equal nonempty target and load widths"
        )
    wires = (*target, *constant, *carries, *((control,) if control is not None else ()))
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
    load = (
        xor_constant_ir(constant, normalized)
        if control is None
        else controlled_xor_constant_ir(control, constant, normalized)
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
    return (*bitwise_not_ir(target), *addition)
