# SPDX-License-Identifier: Apache-2.0
# Qualtran v0.7.0 Square supplies the numerical and resource contracts.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/multiplication.py
# Executable construction: https://arxiv.org/pdf/2105.12767#page=76

"""Unsigned integer square through a reversible column-compression network."""

from collections import deque

from ...core.ir import Instruction
from .and_compute import and_compute_ir, and_uncompute_ir
from .bit_compression import compress_three_bits_ir
from .bitwise import _validate_wires


def integer_square_workspace_size(width: int) -> int:
    """Count the off-diagonal products and column carries."""
    if type(width) is not int or width < 1:
        raise ValueError("Integer square requires a positive input width")
    return width * (width - 1)


def integer_square_ir(
    source: tuple[int, ...],
    result: tuple[int, ...],
    workspace: tuple[int, ...],
    *,
    measurement_bit: int = 0,
) -> tuple[Instruction, ...]:
    """XOR an unsigned square into a double-width result, preserving input.

    Registers are big endian. Starting with zero result implements Square.
    Starting with the computed square implements its adjoint on the valid
    subspace. The caller supplies clean workspace and one reusable slot for
    immediate measured erasure. Workspace returns to zero in either direction.
    """
    width = len(source)
    required = integer_square_workspace_size(width)
    if len(result) != 2 * width or len(workspace) != required:
        raise ValueError(
            f"Integer square requires {2 * width} result and {required} work wires"
        )
    _validate_wires((*source, *result, *workspace))
    if type(measurement_bit) is not int or measurement_bit < 0:
        raise ValueError("Integer square requires a nonnegative measurement slot")

    columns: list[deque[int]] = [deque() for _ in result]
    available = iter(workspace)
    compute: list[Instruction] = []
    undo: list[tuple[Instruction, ...]] = []
    copy: list[Instruction] = []
    little_endian = source[::-1]
    for i, left in enumerate(little_endian):
        columns[2 * i].append(left)
        for j, right in enumerate(little_endian[:i]):
            product = next(available)
            columns[i + j + 1].append(product)
            compute.extend(and_compute_ir(left, right, product))
            undo.append(and_uncompute_ir(left, right, product, measurement_bit))

    for i, (column, output) in enumerate(zip(columns, reversed(result), strict=True)):
        while len(column) >= 2:
            a, b = column.popleft(), column.popleft()
            carry = next(available)
            columns[i + 1].append(carry)
            if column:
                total = column.popleft()
                compute.extend(compress_three_bits_ir(a, b, total, carry))
                undo.append(
                    compress_three_bits_ir(
                        a, b, total, carry, erase_bit=measurement_bit
                    )
                )
                column.append(total)
            else:
                parity = Instruction("cx", (a, b))
                compute.extend((*and_compute_ir(a, b, carry), parity))
                undo.append((parity, *and_uncompute_ir(a, b, carry, measurement_bit)))
                column.append(b)
        if column:
            copy.append(Instruction("cx", (column[0], output)))

    return (*compute, *copy, *(gate for block in reversed(undo) for gate in block))
