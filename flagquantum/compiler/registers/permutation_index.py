# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Rank register contract from Qualtran v0.7.0 SortInPlace:
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/lists/sort_in_place.py
# Executable construction from Lemma 4.12, Eq. 122:
# https://arxiv.org/pdf/2406.19378v1#page=33

"""Reversible permutation indexing over a distinct quantum table."""

from ...core.ir import Instruction
from .bitwise import _validate_wires, controlled_xor_constant_ir, xor_ir
from .equality import equality_constant_ir, equality_ir
from .multi_control import multi_control_x_ir


def permutation_index_requirements(count: int, width: int) -> tuple[int, int, int]:
    """Return rank width, clean work width and reusable classical slot count."""
    if type(count) is not int or type(width) is not int or count < 1 or width < 1:
        raise ValueError("Permutation indexing requires positive count and word width")
    rank_width = (count - 1).bit_length()
    if rank_width > width:
        raise ValueError("Permutation table exceeds the distinct word domain")
    if count == 1:
        return 0, 0, 0
    return rank_width, max(2, width), max(1, width - 1)


def _validate_layout(
    table: tuple[tuple[int, ...], ...],
    words: tuple[tuple[int, ...], ...],
    ranks: tuple[tuple[int, ...], ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> None:
    count = len(table)
    width = len(table[0]) if table else 0
    rank_width, work, slots = permutation_index_requirements(count, width)
    if (
        len(words) != count
        or len(ranks) != count
        or any(len(word) != width for word in (*table, *words))
        or any(len(rank) != rank_width for rank in ranks)
    ):
        raise ValueError(
            "Permutation indexing requires matching table, word and rank layouts"
        )
    if len(workspace) != work or len(classical_bits) != slots:
        raise ValueError(
            f"Permutation indexing requires {work} work qubits and {slots} slots"
        )
    if (
        any(type(bit) is not int or bit < 0 for bit in classical_bits)
        or len(set(classical_bits)) != slots
    ):
        raise ValueError("Permutation indexing requires distinct nonnegative slots")
    _validate_wires(
        (*(qubit for word in (*table, *words, *ranks) for qubit in word), *workspace)
    )


def permutation_rank_xor_ir(
    table: tuple[tuple[int, ...], ...],
    words: tuple[tuple[int, ...], ...],
    ranks: tuple[tuple[int, ...], ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """XOR zero-based ranks into big-endian ranks, preserving table and words.

    The table must hold distinct words, and each source word must be in
    the table for a valid rank. All borrowed work returns to zero.
    """
    _validate_layout(table, words, ranks, workspace, classical_bits)
    if len(table) == 1:
        return ()
    flag = workspace[0]
    carries = len(table[0]) - 1
    scratch = workspace[1 : 1 + carries]
    bits = classical_bits[:carries]
    operations: list[Instruction] = []
    for word, rank in zip(words, ranks, strict=True):
        for i, entry in enumerate(table[1:], start=1):
            equal = equality_ir(entry, word, flag, scratch, bits)
            operations.extend(equal)
            operations.extend(controlled_xor_constant_ir(flag, rank, i))
            operations.extend(equal)
    return tuple(operations)


def permutation_words_xor_ir(
    table: tuple[tuple[int, ...], ...],
    ranks: tuple[tuple[int, ...], ...],
    words: tuple[tuple[int, ...], ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """XOR the indexed quantum entries into words, preserving table and ranks.

    Out-of-range rank bit patterns select no entry. Repeating clears a
    previously computed word. All work returns to zero on every branch.
    """
    _validate_layout(table, words, ranks, workspace, classical_bits)
    if len(table) == 1:
        return xor_ir(table[0], words[0])
    flag, *scratch = workspace
    rank_carries = len(ranks[0]) - 1
    rank_work = tuple(scratch[:rank_carries])
    rank_bits = classical_bits[:rank_carries]
    copy_work = workspace[1:2]
    copy_bit = classical_bits[:1]
    operations: list[Instruction] = []
    for rank, word in zip(ranks, words, strict=True):
        for i, entry in enumerate(table):
            select = equality_constant_ir(rank, i, flag, rank_work, rank_bits)
            operations.extend(select)
            for source, target in zip(entry, word, strict=True):
                operations.extend(
                    multi_control_x_ir(
                        (flag, source),
                        (1, 1),
                        target,
                        copy_work,
                        copy_bit,
                    )
                )
            operations.extend(select)
    return tuple(operations)
