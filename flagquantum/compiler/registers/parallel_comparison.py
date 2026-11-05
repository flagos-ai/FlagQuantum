# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ParallelComparators.build_composite_bloq.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/sorting.py
# Changes: native IR, explicit history tuple and one reused clean comparison pool.

"""Offset comparison blocks with independent logical histories and shared work."""

from ...core.ir import Instruction
from .compare_exchange import compare_exchange_ir
from .comparison_layout import validate_comparison_layout


def parallel_comparison_pairs(count: int, offset: int) -> tuple[tuple[int, int], ...]:
    """Return source block/step order, retaining incomplete trailing blocks."""
    if type(count) is not int or count < 1 or type(offset) is not int or offset < 1:
        raise ValueError(
            "Parallel comparison count and offset must be positive integers"
        )
    pairs = []
    for start in range(0, count, 2 * offset):
        for step in range(offset):
            i, j = start + step, start + step + offset
            if j >= count:
                break
            pairs.append((i, j))
    return tuple(pairs)


def parallel_compare_ir(
    words: tuple[tuple[int, ...], ...],
    offset: int,
    histories: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Compare each source-offset pair, preserving a logical bit per exchange.

    All big-endian words have the same positive unsigned width. Each forward
    history starts zero and remains a logical output. The inverse consumes
    correct complete forward outputs and clears histories. A single comparison
    pool and classical slots are reused after each child cleans its work.
    A zero-pair block uses empty histories/work/slots and emits no instructions.
    """
    pairs = parallel_comparison_pairs(len(words), offset)
    validate_comparison_layout(
        words,
        histories,
        workspace,
        classical_bits,
        history_count=len(pairs),
        adjoint=adjoint,
    )
    stages = tuple(zip(pairs, histories, strict=True))
    return tuple(
        gate
        for (i, j), history in (reversed(stages) if adjoint else stages)
        for gate in compare_exchange_ir(
            words[i], words[j], history, workspace, classical_bits, adjoint=adjoint
        )
    )
