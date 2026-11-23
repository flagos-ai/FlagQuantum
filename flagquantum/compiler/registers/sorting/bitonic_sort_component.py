# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 BitonicSort.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/sorting.py

"""Compose the original recursive sort and ordered left/right/merge histories."""

from ....core.ir.program import ProgramRecord
from ..comparison.compare_exchange_component import (
    adjoint_comparison_network,
    compare_exchange_program,
)
from .bitonic_sort import bitonic_sort_history_count
from .network_graph import bitonic_sort_graph, empty_comparison_program


def bitonic_sort_program(
    count: int, width: int, *, adjoint: bool = False
) -> ProgramRecord:
    """Sort a power-of-two array while retaining every source comparison history.

    One word returns empty history. The inverse consumes a complete valid
    result/history and restores the original word array with histories cleared.
    Temporary work belongs to the compare/exchange children.
    """
    if type(adjoint) is not bool:
        raise ValueError("Bitonic sort direction requires a boolean")
    if not bitonic_sort_history_count(count):
        return empty_comparison_program(count, width, adjoint=adjoint)
    record = bitonic_sort_graph(count, compare_exchange_program(width))
    return adjoint_comparison_network(record, width) if adjoint else record
