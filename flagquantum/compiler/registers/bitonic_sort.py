# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 BitonicSort.build_composite_bloq and its adjoint.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/sorting.py
# Changes: native IR, original recursive/history order and shared clean work.

"""Original recursive bitonic sort over shared merge and compare/exchange."""

from ...core.ir import Instruction
from .bitonic_merge import bitonic_merge_ir
from .compare_exchange import compare_exchange_ir
from .comparison_layout import validate_comparison_layout


def bitonic_sort_history_count(count: int) -> int:
    """Return source logical history length for a positive power-of-two list."""
    if type(count) is not int or count < 1 or count & (count - 1):
        raise ValueError("Bitonic sort count must be a positive power of two")
    log = (count - 1).bit_length()
    return (count // 2) * ((log * (log + 1)) // 2)


def bitonic_sort_ir(
    words: tuple[tuple[int, ...], ...],
    histories: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Sort unsigned words retaining source histories, or undo correct outputs.

    List length is a positive power of two; words have equal positive big-endian
    widths. Forward histories start zero and remain logical output, concatenated
    left/right/merge. Inverse restores original words and clears correct history.
    One clean comparison pool is reused; slots retain latest outcomes. A single
    word has no histories/work/slots and emits the source empty circuit.
    """
    count = bitonic_sort_history_count(len(words))
    validate_comparison_layout(
        words,
        histories,
        workspace,
        classical_bits,
        history_count=count,
        adjoint=adjoint,
    )
    if len(words) == 1:
        return ()
    if len(words) == 2:
        return compare_exchange_ir(
            words[0], words[1], histories[0], workspace, classical_bits, adjoint=adjoint
        )
    half = len(words) // 2
    child = bitonic_sort_history_count(half)
    left = bitonic_sort_ir(
        words[:half], histories[:child], workspace, classical_bits, adjoint=adjoint
    )
    right = bitonic_sort_ir(
        words[half:],
        histories[child : 2 * child],
        workspace,
        classical_bits,
        adjoint=adjoint,
    )
    merge = bitonic_merge_ir(
        words, histories[2 * child :], workspace, classical_bits, adjoint=adjoint
    )
    return (*merge, *right, *left) if adjoint else (*left, *right, *merge)
