# SPDX-License-Identifier: Apache-2.0
# Qualtran v0.7.0 PlusEqualProduct supplies the arithmetic and resource contracts.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/multiplication.py

"""Accumulate a live unsigned product with shared reversible arithmetic."""

from ....core.ir import Instruction
from ..bits.bit_compression import with_weighted_columns_ir
from ..bits.bitwise import _validate_wires
from .addition import add_adjoint_ir, add_ir
from .integer_product import _product_terms, integer_product_requirements


def product_accumulation_ir(
    left: tuple[int, ...],
    right: tuple[int, ...],
    result: tuple[int, ...],
    product_work: tuple[int, ...],
    carries: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Add or subtract the product modulo the width of the big-endian result.

    The accumulator may be occupied. The input words are preserved, and
    product work and addition carries start and finish clean. The first
    classical slot is reused by product erasure after the add completes.
    """
    product_width = len(left) + len(right)
    _, work_width = integer_product_requirements(len(left), len(right))
    if len(result) < product_width:
        raise ValueError(
            "Product accumulation requires a result wide enough for the product"
        )
    if len(product_work) != work_width:
        raise ValueError(
            f"Product accumulation requires {work_width} clean product wires"
        )
    if len(carries) != len(result) - 1 or len(classical_bits) != len(carries):
        raise ValueError(
            "Product accumulation requires one carry and slot per lower result bit"
        )
    if any(type(bit) is not int or bit < 0 for bit in classical_bits):
        raise ValueError("Product accumulation requires nonnegative classical slots")
    if type(adjoint) is not bool:
        raise ValueError("Product accumulation adjoint must be boolean")
    _validate_wires((*left, *right, *result, *product_work, *carries))
    add = add_adjoint_ir if adjoint else add_ir

    def accumulate(word: tuple[int | None, ...]) -> tuple[Instruction, ...]:
        # Product columns are contiguous, with at most one leading zero.
        product = tuple(wire for wire in word if wire is not None)
        return add(product, result, carries, classical_bits)

    return with_weighted_columns_ir(
        product_width,
        product_work,
        classical_bits[0],
        _product_terms(left, right),
        accumulate,
    )
