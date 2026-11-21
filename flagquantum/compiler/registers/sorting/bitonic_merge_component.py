# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 BitonicMerge.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/sorting.py

"""Compose source cross-comparisons and offset rounds with owned histories."""

from ....core.ir.program import ProgramRecord
from ..comparison.compare_exchange_component import (
    adjoint_comparison_network,
    compare_exchange_program,
)
from .bitonic_merge import bitonic_merge_history_count
from .network_graph import bitonic_merge_graph


def bitonic_merge_program(
    half_length: int, width: int, *, adjoint: bool = False
) -> ProgramRecord:
    """Merge two sorted unsigned halves, retaining source-ordered logical history.

    Forward consumes xs/ys and returns result/junk. Inverse consumes the valid
    result/history and restores both input arrays. Half length is a positive
    power of two. Unsorted halves execute the same source network.
    """
    if type(adjoint) is not bool:
        raise ValueError("Bitonic merge direction requires a boolean")
    bitonic_merge_history_count(half_length)
    record = bitonic_merge_graph(half_length, compare_exchange_program(width))
    return adjoint_comparison_network(record, width) if adjoint else record
