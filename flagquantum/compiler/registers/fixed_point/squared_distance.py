"""Reversible squared distance between two unsigned three-axis points."""

from ....core.ir import Instruction
from ..arithmetic.controlled_addition import (
    controlled_add_constant_adjoint_ir,
    controlled_add_constant_ir,
)
from ..arithmetic.integer_square import sum_of_squares_ir, sum_of_squares_requirements
from ..arithmetic.out_of_place_addition import out_of_place_add_ir
from ..bits.bitwise import _validate_wires

PointWires = tuple[tuple[int, ...], ...]


def squared_distance_requirements(width: int) -> tuple[int, int, int]:
    """Return result width, work width and reusable classical slots."""
    result, square_work = sum_of_squares_requirements(width, 3)
    difference_work = 3 * (width + 1)
    return result, difference_work + max(square_work, 2 * width - 1), width


def squared_distance_ir(
    left: PointWires,
    right: PointWires,
    result: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """XOR the integer squared distance into result and restore all workspace.

    Each point has three big-endian unsigned coordinates. The temporary
    difference magnitudes fit in the original coordinate width even when
    an axis difference is negative. The operation also erases a matching
    distance result.
    """
    width = len(left[0]) if len(left) == 3 else 0
    if (
        not width
        or len(right) != 3
        or any(len(axis) != width for axis in (*left, *right))
    ):
        raise ValueError(
            "Squared distance requires equal positive widths on three axes"
        )
    result_width, work_width, slots = squared_distance_requirements(width)
    if len(result) != result_width or len(workspace) != work_width:
        raise ValueError("Squared distance result or workspace width does not match")
    if (
        len(classical_bits) != slots
        or len(set(classical_bits)) != slots
        or any(type(bit) is not int or bit < 0 for bit in classical_bits)
    ):
        raise ValueError("Squared distance requires distinct nonnegative slots")
    _validate_wires(
        (*[wire for axis in (*left, *right) for wire in axis], *result, *workspace)
    )
    words = tuple(
        workspace[start : start + width + 1]
        for start in range(0, 3 * (width + 1), width + 1)
    )
    scratch = workspace[3 * (width + 1) :]
    constant, carries = scratch[:width], scratch[width : 2 * width - 1]
    forward: list[Instruction] = []
    reverse: list[tuple[Instruction, ...]] = []
    for a, b, word in zip(left, right, words, strict=True):
        high, low = word[0], word[1:]
        flip_b = tuple(Instruction("x", (wire,)) for wire in b)
        flip_low = (
            Instruction("x", (high,)),
            *(Instruction("cx", (high, wire)) for wire in low),
            Instruction("x", (high,)),
        )
        # a + ~b = a - b + 2**width - 1. With carry, add one to
        # the low word. Without carry, complement it.
        addition = out_of_place_add_ir(a, b, word)
        increment = controlled_add_constant_ir(
            high, low, 1, constant, carries, classical_bits[:-1]
        )
        forward.extend((*flip_b, *addition, *flip_b, *flip_low, *increment))
        reverse.append(
            (
                *controlled_add_constant_adjoint_ir(
                    high, low, 1, constant, carries, classical_bits[:-1]
                ),
                *flip_low,
                *flip_b,
                *out_of_place_add_ir(
                    a, b, word, adjoint=True, classical_bits=classical_bits
                ),
                *flip_b,
            )
        )
    _, square_work = sum_of_squares_requirements(width, 3)
    square = sum_of_squares_ir(
        tuple(word[1:] for word in words),
        result,
        scratch[:square_work],
        measurement_bit=classical_bits[0],
    )
    return (*forward, *square, *(gate for stage in reversed(reverse) for gate in stage))
