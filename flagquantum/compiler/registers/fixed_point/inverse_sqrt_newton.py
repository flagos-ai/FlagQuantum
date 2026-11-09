# SPDX-License-Identifier: Apache-2.0
# Inverse square root Newton update from Appendix C Figure 5:
# https://arxiv.org/pdf/1805.12445#page=9

"""One widened fixed-point inverse-square-root update."""

from ....core.ir import Instruction
from ..arithmetic.integer_product import _product_terms, integer_product_requirements
from ..arithmetic.integer_square import integer_square_ir, integer_square_workspace_size
from ..arithmetic.product_accumulation import product_accumulation_ir
from ..bits.bit_compression import with_weighted_columns_ir
from ..bits.bitwise import _validate_wires, xor_constant_ir, xor_ir


def _factor_width(n: int, m: int, f: int, b: int) -> int:
    return max(n + 2 * m, f + 2 * b + 2)


def inverse_sqrt_newton_requirements(
    source_width: int,
    estimate_width: int,
    *,
    source_fraction_bits: int,
    estimate_fraction_bits: int,
) -> tuple[int, int]:
    """Return clean workspace width and reusable classical slot count."""
    n, m = source_width, estimate_width
    _, inner_work = integer_product_requirements(n, 2 * m)
    f, b = source_fraction_bits, estimate_fraction_bits
    if any(
        type(fraction) is not int or not 0 <= fraction <= width
        for fraction, width in ((f, n), (b, m))
    ):
        raise ValueError("Inverse square root requires valid fractional widths")
    factor = _factor_width(n, m, f, b)
    _, final_work = integer_product_requirements(m, factor)
    carries = factor - 1
    shared = max(integer_square_workspace_size(m), inner_work, final_work)
    return 2 * m + factor + shared + carries, carries


def inverse_sqrt_newton_step_ir(
    source: tuple[int, ...],
    estimate: tuple[int, ...],
    result: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    source_fraction_bits: int,
    estimate_fraction_bits: int,
    correction: tuple[int, ...] = (),
) -> tuple[Instruction, ...]:
    """XOR the truncation of x*(3-a*x*x)/2 into a big-endian result.

    A correction word optionally adds delta to 3, with f + 2*b fraction
    bits for source fraction width f and estimate fraction width b.
    The unsigned result shares the estimate's fractional width. The valid
    domain has a*x*x <= 3 and a truncated update that fits the result.
    Source, estimate and clean caller-owned work are restored. Reapply
    the same operation to clear a previously computed update.
    """
    n, m = len(source), len(estimate)
    f, b = source_fraction_bits, estimate_fraction_bits
    work, slots = inverse_sqrt_newton_requirements(
        n, m, source_fraction_bits=f, estimate_fraction_bits=b
    )
    if len(result) < max(1, b):
        raise ValueError("Inverse square root result must fit the fractional width")
    if len(workspace) != work or len(classical_bits) != slots:
        raise ValueError(
            f"Inverse square root requires {work} work wires and {slots} slots"
        )
    if (
        any(type(bit) is not int or bit < 0 for bit in classical_bits)
        or len(set(classical_bits)) != slots
    ):
        raise ValueError("Inverse square root requires distinct nonnegative slots")
    if len(correction) > f + 2 * b:
        raise ValueError("Newton correction must be an unsigned fraction below one")
    _validate_wires((*source, *estimate, *result, *workspace, *correction))

    factor_width = _factor_width(n, m, f, b)
    square = workspace[: 2 * m]
    factor = workspace[2 * m : 2 * m + factor_width]
    scratch = workspace[2 * m + factor_width : -slots]
    carries = workspace[-slots:]
    square_work = integer_square_workspace_size(m)
    _, inner_work = integer_product_requirements(n, 2 * m)
    _, final_work = integer_product_requirements(m, factor_width)
    compute_square = integer_square_ir(
        estimate, square, scratch[:square_work], measurement_bit=classical_bits[0]
    )
    set_constant = xor_constant_ir(factor, 3 << (f + 2 * b))
    set_factor = (
        (*set_constant, *xor_ir(correction, factor[-len(correction) :]))
        if correction
        else set_constant
    )
    operands = (source, square, factor, scratch[:inner_work], carries, classical_bits)
    subtract = product_accumulation_ir(*operands, adjoint=True)
    add = product_accumulation_ir(*operands)

    drop = f + 2 * b + 1

    def copy_update(word: tuple[int | None, ...]) -> tuple[Instruction, ...]:
        return tuple(
            Instruction("cx", (wire, target))
            for wire, target in zip(
                reversed(word[:-drop]), reversed(result), strict=False
            )
            if wire is not None
        )

    multiply = with_weighted_columns_ir(
        m + factor_width,
        scratch[:final_work],
        classical_bits[0],
        _product_terms(estimate, factor),
        copy_update,
    )
    return (
        *compute_square,
        *set_factor,
        *subtract,
        *multiply,
        *add,
        *set_factor,
        *compute_square,
    )
