# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Output contract: Qualtran v0.7.0 SortInPlace.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/lists/sort_in_place.py
# Executable distinct-word construction: https://arxiv.org/pdf/2406.19378v1#page=34

"""In-place sorting of distinct unsigned words with permutation output."""

from ....core.ir import Instruction
from ..bits.bitwise import _validate_wires, xor_ir
from ..comparison.comparison_layout import comparison_network_requirements
from .bitonic_sort import bitonic_sort_history_count, bitonic_sort_ir
from .permutation_index import (
    permutation_index_requirements,
    permutation_rank_xor_ir,
    permutation_words_xor_ir,
)


def _network_shape(count: int, width: int, rank_width: int) -> tuple[int, int, int]:
    padded_count = 1 << rank_width
    key_width = width + int(padded_count != count)
    return padded_count, key_width, bitonic_sort_history_count(padded_count)


def in_place_sort_requirements(count: int, width: int) -> tuple[int, int, int]:
    """Return per-word rank width, total clean work and classical slots."""
    rank, index_work, index_slots = permutation_index_requirements(count, width)
    if count == 1:
        return 0, 0, 0
    padded, key_width, histories = _network_shape(count, width, rank)
    sort_work, sort_slots = comparison_network_requirements(key_width, histories)
    work = count * width + padded * key_width + histories + max(sort_work, index_work)
    return rank, work, max(sort_slots, index_slots)


def in_place_sort_ir(
    words: tuple[tuple[int, ...], ...],
    ranks: tuple[tuple[int, ...], ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Sort distinct big-endian words and output zero-based original-to-sorted ranks.

    Forward takes clean rank words. The adjoint requires ascending distinct
    words and their matching rank permutation, then restores original order
    and clears the ranks. Both directions restore all borrowed work. Classical
    slots may be overwritten. Duplicate values are outside this contract.
    """
    count = len(words)
    width = len(words[0]) if words else 0
    rank_width, work, slots = in_place_sort_requirements(count, width)
    if type(adjoint) is not bool:
        raise ValueError("In-place sort adjoint must be boolean")
    if (
        any(len(word) != width for word in words)
        or len(ranks) != count
        or any(len(rank) != rank_width for rank in ranks)
    ):
        raise ValueError("In-place sort requires matching word and rank layouts")
    if (
        len(workspace) != work
        or len(classical_bits) != slots
        or any(type(bit) is not int or bit < 0 for bit in classical_bits)
        or len(set(classical_bits)) != slots
    ):
        raise ValueError(
            "In-place sort requires clean work and distinct classical slots"
        )

    source = tuple(qubit for word in words for qubit in word)
    _validate_wires((*source, *(qubit for rank in ranks for qubit in rank), *workspace))
    if count == 1:
        return ()

    padded_count, key_width, history_count = _network_shape(count, width, rank_width)
    table_size = count * width
    keys_end = table_size + padded_count * key_width
    history_end = keys_end + history_count
    table_wires = workspace[:table_size]
    table = tuple(table_wires[i : i + width] for i in range(0, table_size, width))
    keys = tuple(
        workspace[i : i + key_width] for i in range(table_size, keys_end, key_width)
    )
    history = workspace[keys_end:history_end]
    pool = workspace[history_end:]

    key_payload = tuple(qubit for key in keys[:count] for qubit in key[-width:])
    # Padding keys have high bit 1 and data keys high bit 0. All payload words fit.
    prepare_keys = (
        *xor_ir(source, key_payload),
        *(Instruction("x", (key[0],)) for key in keys[count:]),
    )
    sort_work, sort_slots = comparison_network_requirements(key_width, history_count)
    sort = bitonic_sort_ir(keys, history, pool[:sort_work], classical_bits[:sort_slots])
    unsort = bitonic_sort_ir(
        keys, history, pool[:sort_work], classical_bits[:sort_slots], adjoint=True
    )
    sorted_xor = (
        *prepare_keys,
        *sort,
        *xor_ir(key_payload, table_wires),
        *unsort,
        *prepare_keys,
    )

    _, index_work, index_slots = permutation_index_requirements(count, width)
    scratch = pool[:index_work]
    bits = classical_bits[:index_slots]
    index = permutation_rank_xor_ir(table, words, ranks, scratch, bits)
    reconstruct = permutation_words_xor_ir(table, ranks, words, scratch, bits)
    into_words = xor_ir(table_wires, source)
    into_table = xor_ir(source, table_wires)
    if adjoint:
        return (*into_table, *into_words, *reconstruct, *index, *sorted_xor)
    return (*sorted_xor, *index, *reconstruct, *into_words, *into_table)
