# SPDX-License-Identifier: Apache-2.0
# Nonnegative range reduction and reconstruction from Appendix D:
# https://arxiv.org/pdf/1805.12445#page=10

"""Nonnegative fixed-point arcsine using a caller-supplied polynomial."""

from ...core.ir import Instruction
from .arcsine_reduction import arcsine_reduction_ir, arcsine_reduction_requirements
from .arithmetic import add_adjoint_ir, add_ir
from .bit_compression import with_weighted_columns_ir
from .bitwise import _validate_wires, xor_constant_ir
from .fixed_point_polynomial import (
    fixed_point_polynomial_ir,
    fixed_point_polynomial_requirements,
)
from .fixed_point_sqrt import fixed_point_sqrt_ir, fixed_point_sqrt_requirements
from .integer_product import _product_terms, integer_product_requirements


def _arcsine_layout(
    f: int, p: int, output_width: int, degree: int, precision: int, iterations: int
) -> tuple[int, tuple[int, int], tuple[int, int], tuple[int, int], int]:
    reduction = arcsine_reduction_requirements(f)
    if type(p) is not int or p < 0:
        raise ValueError("Arcsine polynomial precision must be nonnegative")
    if type(output_width) is not int or output_width < 1:
        raise ValueError("Arcsine requires a result integer bit")
    polynomial = fixed_point_polynomial_requirements(p + 2, degree)
    root = fixed_point_sqrt_requirements(
        2 * f, source_fraction_bits=2 * f, precision=precision, iterations=iterations
    )
    factor_width = max(f, precision)
    _, product = integer_product_requirements(factor_width, p + 2)
    return factor_width, reduction, polynomial, root, product


def fixed_point_arcsine_requirements(
    source_fraction_bits: int,
    polynomial_fraction_bits: int,
    output_width: int,
    degree: int,
    *,
    precision: int,
    iterations: int,
) -> tuple[int, int]:
    """Return clean workspace and reusable classical slots."""
    f, p = source_fraction_bits, polynomial_fraction_bits
    (
        q,
        (reduce_work, reduce_slots),
        (poly_work, poly_slots),
        (root_work, root_slots),
        product,
    ) = _arcsine_layout(f, p, output_width, degree, precision, iterations)
    carries = output_width - 1
    live = 1 + 2 * f + p + 2 + 2 * q + 2 * output_width
    return (
        live + max(reduce_work, poly_work, root_work, product + carries),
        max(reduce_slots, poly_slots, root_slots, carries),
    )


def fixed_point_arcsine_ir(
    source: tuple[int, ...],
    result: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    coefficients: tuple[int, ...],
    polynomial_fraction_bits: int,
    half_pi_word: int,
    precision: int,
    iterations: int,
    tuned: bool = False,
) -> tuple[Instruction, ...]:
    """XOR the Appendix D approximation into an unsigned big-endian result.

    Source and result have one integer bit each. The source must encode
    x in [0, 1]. Coefficients are signed p+2 bit words in descending order
    and the evaluated polynomial must be nonnegative on [0, 1/4].
    The caller supplies pi/2 in result ulps. For high inputs the scaled
    product must not exceed that word, and the low product must fit.
    Source and initially clean work are restored.
    """
    f, p, out = len(source) - 1, polynomial_fraction_bits, len(result)
    degree = len(coefficients) - 1
    (
        q,
        (reduce_work, reduce_slots),
        (poly_work, poly_slots),
        (root_work, root_slots),
        product,
    ) = _arcsine_layout(f, p, out, degree, precision, iterations)
    work, slots = fixed_point_arcsine_requirements(
        f, p, out, degree, precision=precision, iterations=iterations
    )
    if type(half_pi_word) is not int or not 0 <= half_pi_word < 1 << out:
        raise ValueError("Arcsine half-pi word must fit the unsigned result")
    if len(workspace) != work or len(classical_bits) != slots:
        raise ValueError(f"Arcsine requires {work} work wires and {slots} slots")
    if (
        any(type(bit) is not int or bit < 0 for bit in classical_bits)
        or len(set(classical_bits)) != slots
    ):
        raise ValueError("Arcsine requires distinct nonnegative slots")
    _validate_wires((*source, *result, *workspace))

    flag = workspace[0]
    reduced = workspace[1 : 1 + 2 * f]
    start = 1 + 2 * f
    poly = workspace[start : start + p + 2]
    start += p + 2
    root = workspace[start : start + q]
    factor = workspace[start + q : start + 2 * q]
    start += 2 * q
    term = workspace[start : start + out]
    angle = workspace[start + out : start + 2 * out]
    scratch = workspace[start + 2 * out :]
    carries = scratch[product : product + out - 1]
    carry_bits = classical_bits[: out - 1]

    flips = tuple(Instruction("x", (wire,)) for wire in source[:2])
    compare = (*flips, Instruction("ccx", (*source[:2], flag)), *flips)
    flip_flag = Instruction("x", (flag,))

    def select_xor(
        left: tuple[int | None, ...], right: tuple[int, ...], offset: int = 0
    ) -> tuple[Instruction, ...]:
        """Controlled XOR after dropping low source or padding low target bits."""
        drop = max(0, offset)
        target = right[: len(right) + min(0, offset)]
        return tuple(
            Instruction("ccx", (flag, wire, dest))
            for wire, dest in zip(
                reversed(left[: len(left) - drop]), reversed(target), strict=False
            )
            if wire is not None
        )

    reduce = arcsine_reduction_ir(
        source, reduced, scratch[:reduce_work], classical_bits[:reduce_slots]
    )
    polynomial = fixed_point_polynomial_ir(
        reduced,
        poly,
        scratch[:poly_work],
        classical_bits[:poly_slots],
        argument_fraction_bits=2 * f,
        coefficients=coefficients,
    )
    square_root = fixed_point_sqrt_ir(
        reduced,
        root,
        scratch[:root_work],
        classical_bits[:root_slots],
        source_fraction_bits=2 * f,
        precision=precision,
        iterations=iterations,
        tuned=tuned,
    )
    select_factor = (
        *select_xor(source[1:], factor, f - q),
        flip_flag,
        *select_xor(root, factor),
        flip_flag,
    )
    constant = xor_constant_ir(angle, half_pi_word)
    subtract = add_adjoint_ir(term, angle, carries, carry_bits)
    restore = add_ir(term, angle, carries, carry_bits)
    copy = (*select_xor(term, result), flip_flag, *select_xor(angle, result), flip_flag)
    scale = q + p - (out - 1)

    def reconstruct(word: tuple[int | None, ...]) -> tuple[Instruction, ...]:
        select_term = (
            *select_xor(word, term, scale),
            flip_flag,
            *select_xor(word, term, scale - 1),
            flip_flag,
        )
        return (
            *select_term,
            *constant,
            *subtract,
            *copy,
            *restore,
            *constant,
            *select_term,
        )

    multiply = with_weighted_columns_ir(
        q + p + 2,
        scratch[:product],
        classical_bits[0],
        _product_terms(factor, poly),
        reconstruct,
    )
    return (
        *compare,
        *reduce,
        *polynomial,
        *square_root,
        *select_factor,
        *multiply,
        *select_factor,
        *square_root,
        *polynomial,
        *reduce,
        *compare,
    )
