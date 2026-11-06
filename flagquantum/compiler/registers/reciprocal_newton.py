# SPDX-License-Identifier: Apache-2.0
# Qualtran v0.7.0 InvertRealNumber models Newton reciprocal iteration.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/multiplication.py
# Exact-step-then-truncate construction: https://arxiv.org/pdf/1511.08253#page=30

"""One widened Newton update for an unsigned mixed-format reciprocal."""

from ...core.ir import Instruction
from .bitwise import _validate_wires, xor_ir
from .integer_product import integer_product_requirements
from .integer_square import integer_square_ir, integer_square_workspace_size
from .product_accumulation import product_accumulation_ir


def reciprocal_newton_requirements(input_width: int, precision: int) -> tuple[int, int]:
    """Return the clean quantum work and classical slot counts."""
    _, product_work = integer_product_requirements(input_width, 2 * precision)
    numerator_width = input_width + 2 * precision
    shared = max(integer_square_workspace_size(precision), product_work)
    carries = numerator_width - 1
    return 2 * precision + numerator_width + shared + carries, carries


def reciprocal_newton_step_ir(
    source: tuple[int, ...],
    estimate: tuple[int, ...],
    result: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    source_fraction_bits: int,
) -> tuple[Instruction, ...]:
    """XOR floor((2*x - a*x*x) * 2**b) into a b-bit fractional result.

    All words are big endian. The numerical contract assumes a >= 1 and
    0 <= x <= 1/a. Estimate and result have the same fractional width b.
    The numerator is exact until the result slice is copied. Source,
    estimate and clean workspace are restored. Reapplying erases the result.
    """
    if type(source_fraction_bits) is not int or not 0 <= source_fraction_bits < len(
        source
    ):
        raise ValueError("Reciprocal update requires a nonempty source integer part")
    precision = len(estimate)
    if len(result) != precision:
        raise ValueError(
            "Reciprocal update requires matching estimate and result widths"
        )
    required, slots = reciprocal_newton_requirements(len(source), precision)
    if len(workspace) != required or len(classical_bits) != slots:
        raise ValueError(
            f"Reciprocal update requires {required} work wires and {slots} slots"
        )
    if (
        any(type(bit) is not int or bit < 0 for bit in classical_bits)
        or len(set(classical_bits)) != slots
    ):
        raise ValueError(
            "Reciprocal update requires distinct nonnegative classical slots"
        )
    _validate_wires((*source, *estimate, *result, *workspace))

    square_width = 2 * precision
    numerator_width = len(source) + square_width
    square = workspace[:square_width]
    numerator = workspace[square_width : square_width + numerator_width]
    shared = workspace[square_width + numerator_width : -slots]
    carries = workspace[-slots:]
    _, product_work = integer_product_requirements(len(source), square_width)
    square_work = integer_square_workspace_size(precision)
    compute_square = integer_square_ir(
        estimate,
        square,
        shared[:square_work],
        measurement_bit=classical_bits[0],
    )
    fraction = source_fraction_bits + precision
    double = xor_ir(estimate, numerator[-fraction - precision - 1 : -fraction - 1])
    arguments = (
        source,
        square,
        numerator,
        shared[:product_work],
        carries,
        classical_bits,
    )
    subtract = product_accumulation_ir(*arguments, adjoint=True)
    copy = xor_ir(numerator[-fraction - precision : -fraction], result)
    restore = product_accumulation_ir(*arguments)
    return (
        *compute_square,
        *double,
        *subtract,
        *copy,
        *restore,
        *double,
        *compute_square,
    )
