# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ParallelComparators.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/sorting.py

"""Compose the existing offset pair order with directional history arrays."""

from ....core.ir.program import ProgramRecord
from ..sorting.network_graph import empty_comparison_program, parallel_comparison_graph
from .compare_exchange_component import (
    adjoint_comparison_network,
    compare_exchange_program,
)
from .parallel_comparison import parallel_comparison_pairs


def parallel_comparison_program(
    count: int, offset: int, width: int, *, adjoint: bool = False
) -> ProgramRecord:
    """Compare offset pairs, retaining one logical history per source-ordered pair.

    Zero-pair blocks return empty junk and require no comparison workspace.
    The inverse consumes histories in reverse source dependency order.
    """
    if type(adjoint) is not bool:
        raise ValueError("Parallel comparison direction requires a boolean")
    if not parallel_comparison_pairs(count, offset):
        return empty_comparison_program(count, width, adjoint=adjoint)
    record = parallel_comparison_graph(count, offset, compare_exchange_program(width))
    return adjoint_comparison_network(record, width) if adjoint else record
