# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Source signature and size model: Qualtran v0.7.0 SymmetricDifference.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/lists/symmetric_difference.py

"""Reversible symmetric difference of two sorted sets."""

from dataclasses import dataclass
from itertools import islice

from ...core.ir import Instruction
from .bitonic_merge import bitonic_merge_history_count, bitonic_merge_ir
from .bitonic_sort import bitonic_sort_history_count, bitonic_sort_ir
from .bitwise import _validate_wires, xor_ir
from .comparison_layout import comparison_network_requirements
from .equality import equality_constant_ir, equality_ir
from .hamming_weight import hamming_weight_ir, hamming_weight_requirements
from .multi_control import multi_control_x_ir


@dataclass(frozen=True)
class _DifferenceLayout:
    half: int
    padding: int
    tagging: int
    merge_history: int
    compact_history: int
    adjacent: int
    junk: int
    count: int
    pool: int
    slots: int


def _layout(left: int, right: int, width: int, size: int) -> _DifferenceLayout | None:
    if (
        any(type(value) is not int for value in (left, right, width, size))
        or not 0 <= right <= left
        or width < 1
        or size < 0
    ):
        raise ValueError(
            "Symmetric difference requires left >= right >= 0 and valid widths"
        )
    if left > 1 << width:
        raise ValueError("Symmetric difference set exceeds the word domain")
    total = left + right
    if right == 0 or size < left - right or size > total or (total - size) % 2:
        return None

    half = 1 << (left - 1).bit_length()
    padding = int(right != half)
    tagging = int(0 < size < total)
    merge_history = bitonic_merge_history_count(half)
    compact_history = bitonic_sort_history_count(2 * half) if tagging else 0
    merge_work, merge_slots = comparison_network_requirements(
        width + padding, merge_history
    )
    compact_work, compact_slots = comparison_network_requirements(
        width + padding + tagging, compact_history
    )
    adjacent = total - 1
    junk, count = hamming_weight_requirements(adjacent)
    return _DifferenceLayout(
        half,
        padding,
        tagging,
        merge_history,
        compact_history,
        adjacent,
        junk,
        count,
        max(merge_work, compact_work, width - 1, count - 1, 1),
        max(merge_slots, compact_slots, width - 1, count - 1, junk, 1),
    )


def _work_size(layout: _DifferenceLayout, width: int) -> int:
    return (
        2 * layout.half * (width + layout.padding + layout.tagging)
        + layout.merge_history
        + layout.compact_history
        + layout.adjacent
        + layout.junk
        + layout.count
        + 1
        + layout.pool
    )


def symmetric_difference_requirements(
    left: int, right: int, width: int, size: int
) -> tuple[int, int]:
    """Return the clean work and reusable classical slots for this set shape."""
    layout = _layout(left, right, width, size)
    return (_work_size(layout, width), layout.slots) if layout else (0, 0)


def symmetric_difference_ir(
    left: tuple[tuple[int, ...], ...],
    right: tuple[tuple[int, ...], ...],
    output: tuple[tuple[int, ...], ...],
    flag: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """XOR an ascending symmetric difference and a matching-size flag.

    Input sets hold distinct ascending, equal-width big-endian words.
    The left set is at least as long as the right. On a size mismatch,
    both XOR targets are unchanged. Inputs and clean work are restored.
    The logical operation is its own inverse.
    """
    words = (*left, *right)
    width = len(words[0]) if words else len(output[0]) if output else 1
    if any(len(word) != width for word in (*words, *output)):
        raise ValueError("Symmetric difference requires matching word widths")
    size = len(output)
    layout = _layout(len(left), len(right), width, size)
    work, slots = (_work_size(layout, width), layout.slots) if layout else (0, 0)
    if (
        len(workspace) != work
        or len(classical_bits) != slots
        or any(type(bit) is not int or bit < 0 for bit in classical_bits)
        or len(set(classical_bits)) != slots
    ):
        raise ValueError("Symmetric difference requires clean work and distinct slots")
    destination = tuple(qubit for word in output for qubit in word)
    source = tuple(qubit for word in words for qubit in word)
    _validate_wires((*source, *destination, flag, *workspace))
    if layout is None:
        if right or size != len(left):
            return ()
        copy = xor_ir(source, destination) if source else ()
        return (*copy, Instruction("x", (flag,)))

    key_width = width + layout.padding + layout.tagging
    allocation = iter(workspace)
    keys = tuple(tuple(islice(allocation, key_width)) for _ in range(2 * layout.half))
    merge_history = tuple(islice(allocation, layout.merge_history))
    compact_history = tuple(islice(allocation, layout.compact_history))
    equal_bits = tuple(islice(allocation, layout.adjacent))
    junk = tuple(islice(allocation, layout.junk))
    count = tuple(islice(allocation, layout.count))
    success = next(allocation)
    pool = tuple(allocation)

    active = (*keys[: len(left)], *keys[layout.half : layout.half + len(right)])
    pads = (*keys[len(left) : layout.half], *keys[layout.half + len(right) :])
    load = (
        *xor_ir(source, tuple(qubit for key in active for qubit in key[-width:])),
        *(Instruction("x", (key[0],)) for key in pads),
    )
    merge_keys = tuple((*key[: layout.padding], *key[-width:]) for key in keys)
    merge_work, merge_slots = comparison_network_requirements(
        width + layout.padding, len(merge_history)
    )
    merge_args = (
        merge_keys,
        merge_history,
        pool[:merge_work],
        classical_bits[:merge_slots],
    )
    payload = tuple(key[-width:] for key in keys[: len(words)])
    pairs = zip(payload[:-1], payload[1:], equal_bits, strict=True)
    compare = tuple(
        gate
        for a, b, target in pairs
        for gate in equality_ir(
            a, b, target, pool[: width - 1], classical_bits[: width - 1]
        )
    )
    tag: tuple[Instruction, ...] = ()
    compact: tuple[Instruction, ...] = ()
    uncomp: tuple[Instruction, ...] = ()
    if layout.tagging:
        tag = tuple(
            Instruction("cx", (bit, keys[j][layout.padding]))
            for i, bit in enumerate(equal_bits)
            for j in (i, i + 1)
        )
        compact_work, compact_slots = comparison_network_requirements(
            key_width, len(compact_history)
        )
        compact_args = (
            keys,
            compact_history,
            pool[:compact_work],
            classical_bits[:compact_slots],
        )
        compact = bitonic_sort_ir(*compact_args)
        uncomp = bitonic_sort_ir(*compact_args, adjoint=True)

    match = equality_constant_ir(
        count,
        (len(words) - size) // 2,
        success,
        pool[: len(count) - 1],
        classical_bits[: len(count) - 1],
    )
    copy = tuple(
        gate
        for entry, target_word in zip(payload[:size], output, strict=True)
        for wire, target in zip(entry, target_word, strict=True)
        for gate in multi_control_x_ir(
            (success, wire), (1, 1), target, pool[:1], classical_bits[:1]
        )
    )
    return (
        *load,
        *bitonic_merge_ir(*merge_args),
        *compare,
        *tag,
        *hamming_weight_ir(equal_bits, junk, count),
        *match,
        *compact,
        *copy,
        Instruction("cx", (success, flag)),
        *uncomp,
        *match,
        *hamming_weight_ir(
            equal_bits,
            junk,
            count,
            adjoint=True,
            classical_bits=classical_bits[: len(junk)],
        ),
        *tag,
        *compare,
        *bitonic_merge_ir(*merge_args, adjoint=True),
        *load,
    )
