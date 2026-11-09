# SPDX-License-Identifier: Apache-2.0

"""Polynomial-size ArcTan through reciprocal reduction and Horner evaluation."""

from fractions import Fraction

from ....core.ir import Instruction
from ..arithmetic.addition import add_constant_adjoint_ir, negate_ir
from ..arithmetic.integer_product import _product_terms, integer_product_requirements
from ..arithmetic.integer_square import integer_square_ir, integer_square_workspace_size
from ..bits.bit_compression import with_weighted_columns_ir
from ..bits.bitwise import _validate_wires, bitwise_not_ir
from ..control.multi_control import multi_control_x_ir
from .fixed_point_polynomial import (
    fixed_point_polynomial_ir,
    fixed_point_polynomial_requirements,
)
from .reciprocal import reciprocal_ir, reciprocal_requirements


def _normalized_arctan_coefficients(precision: int, degree: int) -> tuple[int, ...]:
    """Encode the alternating series for 2*atan(r)/(pi*r)."""
    terms = (precision + 5) // 4 + 2

    def atan_inverse(denominator: int) -> Fraction:
        return sum(
            (
                Fraction((-1) ** j, (2 * j + 1) * denominator ** (2 * j + 1))
                for j in range(terms)
            ),
            Fraction(0),
        )

    pi_estimate = 16 * atan_inverse(5) - 4 * atan_inverse(239)
    return tuple(
        round(Fraction((-1) ** j * (2 << precision), 2 * j + 1) / pi_estimate)
        for j in range(degree, -1, -1)
    )


def arctan_arithmetic_requirements(
    source_width: int, precision: int, *, degree: int
) -> tuple[int, int]:
    """Return live word storage, shared scratch and reusable slots."""
    reciprocal, reciprocal_slots = reciprocal_requirements(
        source_width, precision, source_fraction_bits=0
    )
    polynomial, polynomial_slots = fixed_point_polynomial_requirements(
        precision + 1, degree
    )
    if source_width == 1:
        return 0, 0
    square = integer_square_workspace_size(precision)
    _, product = integer_product_requirements(precision, precision + 1)
    numerator = 2 * precision + 1
    return (
        1
        + 3 * precision
        + (precision + 1)
        + numerator
        + max(reciprocal, square, polynomial, product + 2 * numerator - 1),
        max(reciprocal_slots, polynomial_slots, numerator - 1, source_width - 1),
    )


def arctan_arithmetic_ir(
    source: tuple[int, ...],
    sign: int,
    target: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    precision: int,
    degree: int,
) -> tuple[Instruction, ...]:
    """XOR an arithmetic approximation to the locked ArcTan map.

    Source is an unsigned integer, target is a pure fraction no wider
    than the working precision, and sign receives x!=0. Inputs 0 and 1
    use exact masks. For larger inputs, use reciprocal range reduction,
    a normalized Taylor polynomial and exact modular product negation.
    All words are big endian and initially clean work is restored.
    """
    n, b = len(source), precision
    work, slots = arctan_arithmetic_requirements(n, b, degree=degree)
    if not 1 <= len(target) <= b:
        raise ValueError("ArcTan target must fit the working precision")
    if len(workspace) != work or len(classical_bits) != slots:
        raise ValueError(f"ArcTan requires {work} work qubits and {slots} slots")
    if (
        any(type(bit) is not int or bit < 0 for bit in classical_bits)
        or len(set(classical_bits)) != slots
    ):
        raise ValueError("ArcTan requires distinct nonnegative slots")
    _validate_wires((*source, sign, *target, *workspace))
    if n == 1:
        return (
            Instruction("cx", (source[0], sign)),
            Instruction("cx", (source[0], target[0])),
        )

    flag = workspace[0]
    reciprocal = workspace[1 : b + 1]
    square = workspace[b + 1 : 3 * b + 1]
    polynomial = workspace[3 * b + 1 : 4 * b + 2]
    numerator = workspace[4 * b + 2 : 6 * b + 3]
    scratch = workspace[6 * b + 3 :]
    inverse_work, inverse_slots = reciprocal_requirements(n, b, source_fraction_bits=0)
    inv = reciprocal_ir(
        source,
        reciprocal,
        scratch[:inverse_work],
        classical_bits[:inverse_slots],
        source_fraction_bits=0,
        precision=b,
    )
    sqr = integer_square_ir(
        reciprocal,
        square,
        scratch[: integer_square_workspace_size(b)],
        measurement_bit=classical_bits[0],
    )
    poly_work, poly_slots = fixed_point_polynomial_requirements(b + 1, degree)
    poly = fixed_point_polynomial_ir(
        square,
        polynomial,
        scratch[:poly_work],
        classical_bits[:poly_slots],
        argument_fraction_bits=2 * b,
        coefficients=_normalized_arctan_coefficients(b, degree),
    )
    _, product_work = integer_product_requirements(b, b + 1)
    const = scratch[product_work : product_work + len(numerator)]
    carry = scratch[
        product_work + len(numerator) : product_work + 2 * len(numerator) - 1
    ]
    negate_bits = classical_bits[: len(numerator) - 1]
    negate = negate_ir(numerator, const, carry, negate_bits)
    restore = (
        *add_constant_adjoint_ir(numerator, 1, const, carry, negate_bits),
        *bitwise_not_ir(numerator),
    )

    def copy_normalized(word: tuple[int | None, ...]) -> tuple[Instruction, ...]:
        load = tuple(
            Instruction("cx", (bit, qubit))
            for bit, qubit in zip(word, numerator, strict=True)
            if bit is not None
        )
        copy = tuple(
            Instruction("ccx", (flag, bit, qubit))
            for bit, qubit in zip(numerator[1 : 1 + len(target)], target, strict=True)
        )
        return (*load, *negate, *copy, *restore, *load)

    multiply = with_weighted_columns_ir(
        len(numerator),
        scratch[:product_work],
        classical_bits[0],
        _product_terms(reciprocal, polynomial),
        copy_normalized,
    )
    high = (
        *multi_control_x_ir(
            source[:-1], (0,) * (n - 1), flag, scratch[: n - 2], classical_bits[: n - 2]
        ),
        Instruction("x", (flag,)),
    )
    sign_mask = (
        *multi_control_x_ir(
            source, (0,) * n, sign, scratch[: n - 1], classical_bits[: n - 1]
        ),
        Instruction("x", (sign,)),
    )
    one = multi_control_x_ir(
        source,
        (0,) * (n - 1) + (1,),
        target[0],
        scratch[: n - 1],
        classical_bits[: n - 1],
    )
    return (
        *inv,
        *sqr,
        *poly,
        *high,
        *multiply,
        *high,
        *poly,
        *sqr,
        *inv,
        *sign_mask,
        *one,
    )
