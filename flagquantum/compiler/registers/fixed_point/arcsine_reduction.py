# SPDX-License-Identifier: Apache-2.0
# Range reduction from Appendix D of https://arxiv.org/pdf/1805.12445#page=10

"""Exact unsigned argument reduction for the arcsine polynomial."""

from ....core.ir import Instruction
from ..arithmetic.addition import add_adjoint_ir, add_ir
from ..arithmetic.integer_square import integer_square_ir, integer_square_workspace_size
from ..bits.bitwise import _validate_wires


def arcsine_reduction_requirements(fraction_bits: int) -> tuple[int, int]:
    """Return clean workspace width and reusable classical slot count."""
    square_work = integer_square_workspace_size(fraction_bits)
    return (
        1 + max(2 * fraction_bits + 1, 2 * fraction_bits + square_work),
        fraction_bits,
    )


def arcsine_reduction_ir(
    source: tuple[int, ...],
    result: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """XOR x² below one half, or (1-x)/2 otherwise, into a pure fraction.

    All words are big endian. The input has one integer bit and f fractional
    bits, and must encode x in [0, 1]. The result has 2f fractional bits.
    Workspace starts and finishes clean and the source is preserved.
    """
    fraction_bits = len(source) - 1
    work, slots = arcsine_reduction_requirements(fraction_bits)
    if len(result) != 2 * fraction_bits:
        raise ValueError("Arcsine reduction requires twice the input fractional width")
    if len(workspace) != work or len(classical_bits) != slots:
        raise ValueError(
            f"Arcsine reduction requires {work} work wires and {slots} slots"
        )
    if (
        any(type(bit) is not int or bit < 0 for bit in classical_bits)
        or len(set(classical_bits)) != slots
    ):
        raise ValueError("Arcsine reduction requires distinct nonnegative slots")
    _validate_wires((*source, *result, *workspace))

    flag, *scratch_list = workspace
    scratch = tuple(scratch_list)
    high_bits = tuple(Instruction("x", (wire,)) for wire in source[:2])
    low_compare = (*high_bits, Instruction("ccx", (*source[:2], flag)), *high_bits)
    flip_flag = Instruction("x", (flag,))

    half_difference = scratch[: len(source)]
    carries = scratch[len(source) : len(source) + fraction_bits]
    half = Instruction("x", (half_difference[0],))
    borrow = add_adjoint_ir(source, half_difference, carries, classical_bits)
    restore = add_ir(source, half_difference, carries, classical_bits)
    copy_difference = tuple(
        Instruction("ccx", (flag, wire, target))
        for wire, target in zip(
            half_difference, result[: len(half_difference)], strict=True
        )
    )

    square = scratch[: len(result)]
    square_work = scratch[
        len(result) : len(result) + integer_square_workspace_size(fraction_bits)
    ]
    compute_square = integer_square_ir(
        source[1:], square, square_work, measurement_bit=classical_bits[0]
    )
    copy_square = tuple(
        Instruction("ccx", (flag, wire, target))
        for wire, target in zip(square, result, strict=True)
    )
    return (
        *low_compare,
        half,
        *borrow,
        flip_flag,
        *copy_difference,
        flip_flag,
        *restore,
        half,
        *compute_square,
        *copy_square,
        *compute_square,
        *low_compare,
    )
