# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 HammingWeightCompute.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/hamming_weight.py

"""Shared reversible sum and carry compression."""

from collections import deque
from collections.abc import Callable, Iterable

from ....core.ir import Instruction
from ..primitives.and_compute import and_compute_ir, and_uncompute_ir
from .bitwise import _validate_wires


def compress_three_bits_ir(
    a: int, b: int, total: int, carry: int, *, erase_bit: int | None = None
) -> tuple[Instruction, ...]:
    """Replace total by a XOR b XOR total and write carry into a clean qubit."""
    if erase_bit is not None:
        return (
            Instruction("cx", (b, total)),
            Instruction("cx", (a, carry)),
            Instruction("cx", (a, b)),
            *and_uncompute_ir(b, total, carry, erase_bit),
            Instruction("cx", (a, total)),
            Instruction("cx", (a, b)),
        )
    return (
        Instruction("cx", (a, b)),
        Instruction("cx", (a, total)),
        *and_compute_ir(b, total, carry),
        Instruction("cx", (a, b)),
        Instruction("cx", (a, carry)),
        Instruction("cx", (b, total)),
    )


def weighted_columns_xor_ir(
    inputs: tuple[int, ...],
    result: tuple[int, ...],
    workspace: tuple[int, ...],
    measurement_bit: int,
    terms: Iterable[tuple[int, int, int | None]],
) -> tuple[Instruction, ...]:
    """Reduce bound bit terms, copy the parity word, then erase products and carries.

    The caller computes column capacity and the exact work width. A term
    provides a little-endian column and either one qubit or a pair to AND.
    """
    _validate_wires((*inputs, *result, *workspace))

    def copy(word: tuple[int | None, ...]) -> tuple[Instruction, ...]:
        return tuple(
            Instruction("cx", (wire, target))
            for wire, target in zip(reversed(word), reversed(result), strict=True)
            if wire is not None
        )

    return with_weighted_columns_ir(
        len(result), workspace, measurement_bit, terms, copy
    )


def with_weighted_columns_ir(
    result_width: int,
    workspace: tuple[int, ...],
    measurement_bit: int,
    terms: Iterable[tuple[int, int, int | None]],
    consume: Callable[[tuple[int | None, ...]], tuple[Instruction, ...]],
) -> tuple[Instruction, ...]:
    """Use the live big-endian parity word before reversing the column network.

    A missing column is a constant zero. The consumer must restore the
    parity qubits so the products and carries can be erased.
    """
    if type(measurement_bit) is not int or measurement_bit < 0:
        raise ValueError("Column compression requires a nonnegative measurement slot")

    columns: list[deque[int]] = [deque() for _ in range(result_width)]
    available = iter(workspace)
    compute: list[Instruction] = []
    undo: list[tuple[Instruction, ...]] = []
    for position, left, right in terms:
        if right is None:
            columns[position].append(left)
        else:
            product = next(available)
            columns[position].append(product)
            compute.extend(and_compute_ir(left, right, product))
            undo.append(and_uncompute_ir(left, right, product, measurement_bit))

    for i, column in enumerate(columns):
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

    word = tuple(column[0] if column else None for column in reversed(columns))
    return (
        *compute,
        *consume(word),
        *(gate for block in reversed(undo) for gate in block),
    )
