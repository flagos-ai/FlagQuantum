# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 arithmetic/sorting.py.

"""Signatures and declared calls for the original sorting examples."""

from collections.abc import Hashable

from sympy import Symbol

from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.compiler.registers.comparison.ordering_declaration import (
    ComparatorDeclaration,
    ParallelComparatorsDeclaration,
)
from flagquantum.compiler.registers.sorting.bitonic_declaration import (
    BitonicMergeDeclaration,
    BitonicSortDeclaration,
)
from flagquantum.core.ir.program.signatures import Signature

SortingExample = tuple[str, Signature, dict[Hashable, SymbolicInt]]


def sorting_examples() -> tuple[SortingExample, ...]:
    """Consume the shared generic declarations for every original factory."""
    n = Symbol("n")
    declarations = (
        ("comparator", ComparatorDeclaration(3)),
        ("comparator_symbolic", ComparatorDeclaration(n)),
        ("bitonic_sort", BitonicSortDeclaration(8, 4)),
        ("bitonic_sort_symbolic", BitonicSortDeclaration(4, n)),
        ("parallel_comparators", ParallelComparatorsDeclaration(7, 2, 3)),
        ("bitonic_merge", BitonicMergeDeclaration(4, 7)),
    )
    return tuple(
        (name, declaration.signature, declaration.direct_calls())
        for name, declaration in declarations
    )


def sorting_example_graph(example: SortingExample) -> SymbolicCallGraph[Hashable]:
    """Build the exact one-level graph without constructing executable gates."""
    name, _, calls = example
    return symbolic_call_graph(name, {name: calls}, max_depth=1)
