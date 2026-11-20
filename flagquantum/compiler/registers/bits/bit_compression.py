# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 HammingWeightCompute.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/hamming_weight.py

"""Shared reversible sum and carry compression."""

from collections import deque
from collections.abc import Callable, Iterable

from ....core.ir import Instruction
from ..primitives.and_compute import and_compute_ir, and_uncompute_ir
from .bit_compression_structure import CompressionComponent, compression_components
from .bitwise import _validate_wires


def compress_three_bits_ir(
    a: int, b: int, total: int, carry: int, *, erase_bit: int | None = None
) -> tuple[Instruction, ...]:
    """Replace total by a XOR b XOR total and write carry into a clean qubit."""
    components = compression_components(a, b, total, carry)
    if erase_bit is not None:
        components = tuple(reversed(components))
    return _expand_column_components(components, erase_bit=erase_bit)


def _expand_column_components(
    components: Iterable[CompressionComponent], *, erase_bit: int | None = None
) -> tuple[Instruction, ...]:
    """Expand the shared logical sequence through the existing measured And."""
    gates: list[Instruction] = []
    for kind, wires in components:
        if kind == "cx":
            gates.append(Instruction("cx", wires))
        elif erase_bit is None:
            gates.extend(and_compute_ir(wires[0], wires[1], wires[2]))
        else:
            gates.extend(and_uncompute_ir(wires[0], wires[1], wires[2], erase_bit))
    return tuple(gates)


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

    compute, word, undo = weighted_column_components(result_width, workspace, terms)
    return (
        *_expand_column_components(compute),
        *consume(word),
        *_expand_column_components(undo, erase_bit=measurement_bit),
    )


def weighted_column_components(
    result_width: int,
    workspace: tuple[int, ...],
    terms: Iterable[tuple[int, int, int | None]],
) -> tuple[
    tuple[CompressionComponent, ...],
    tuple[int | None, ...],
    tuple[CompressionComponent, ...],
]:
    """Return product/compression steps, live parity word and inverse steps.

    Addresses follow the caller's wire layout. The final word is MSB first.
    AND targets consume workspace in order and remain live through the consumer.
    Inverse AND steps erase their known targets with measurement feedback.
    """
    columns: list[deque[int]] = [deque() for _ in range(result_width)]
    available = iter(workspace)
    compute: list[CompressionComponent] = []
    undo: list[tuple[CompressionComponent, ...]] = []
    for position, left, right in terms:
        if right is None:
            columns[position].append(left)
        else:
            product = next(available)
            columns[position].append(product)
            step: CompressionComponent = ("and", (left, right, product))
            compute.append(step)
            undo.append((step,))

    for i, column in enumerate(columns):
        while len(column) >= 2:
            a, b = column.popleft(), column.popleft()
            carry = next(available)
            columns[i + 1].append(carry)
            if column:
                total = column.popleft()
                steps = compression_components(a, b, total, carry)
                compute.extend(steps)
                undo.append(tuple(reversed(steps)))
                column.append(total)
            else:
                parity: CompressionComponent = ("cx", (a, b))
                conjunction: CompressionComponent = ("and", (a, b, carry))
                compute.extend((conjunction, parity))
                undo.append((parity, conjunction))
                column.append(b)

    word = tuple(column[0] if column else None for column in reversed(columns))
    return (
        tuple(compute),
        word,
        tuple(step for block in reversed(undo) for step in block),
    )
