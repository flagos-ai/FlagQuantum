# SPDX-License-Identifier: Apache-2.0
# Qualtran v0.7.0 Product supplies the register and cost contracts.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/multiplication.py
# Executable construction: https://arxiv.org/pdf/2105.12767#page=81

"""Unsigned products using measured reversible column compression."""

from ...core.ir import Instruction
from .bit_compression import weighted_columns_xor_ir


def integer_product_requirements(left_width: int, right_width: int) -> tuple[int, int]:
    """Return the pinned result width and the native product-and-carry work width."""
    if any(type(width) is not int or width < 1 for width in (left_width, right_width)):
        raise ValueError("Integer product requires positive input widths")
    largest = max(left_width, right_width)
    return 2 * largest, 2 * left_width * right_width - largest


def integer_product_ir(
    left: tuple[int, ...],
    right: tuple[int, ...],
    result: tuple[int, ...],
    workspace: tuple[int, ...],
    *,
    measurement_bit: int = 0,
) -> tuple[Instruction, ...]:
    """XOR the exact unsigned product into the result and erase all work.

    Words are big endian. From a zero result this implements Product.
    Applied again to the computed result, the same stream clears it.
    Inputs are preserved. The caller provides clean work and one reusable
    slot for immediate measured feedback.
    """
    result_width, work_width = integer_product_requirements(len(left), len(right))
    if len(result) != result_width or len(workspace) != work_width:
        raise ValueError(
            f"Integer product requires {result_width} result and {work_width} work wires"
        )
    return weighted_columns_xor_ir(
        (*left, *right),
        result,
        workspace,
        measurement_bit,
        (
            (i + j, a, b)
            for i, a in enumerate(reversed(left))
            for j, b in enumerate(reversed(right))
        ),
    )
