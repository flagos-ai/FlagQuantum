# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 arithmetic/permutation.py.

"""Native metadata for the complete original permutation examples."""

from __future__ import annotations

from collections.abc import Hashable

from sympy import IndexedBase, Symbol

from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.compiler.registers.sorting.permutation_declaration import (
    CycleShape,
    PermutationCycleDeclaration,
    PermutationDeclaration,
)
from flagquantum.core.ir.program.signatures import Signature

PermutationExample = tuple[str, Signature, Hashable, dict[Hashable, SymbolicInt]]


def permutation_examples() -> tuple[PermutationExample, ...]:
    """Preserve every factory's cycles, unknown shapes, bounds and call identities."""
    size = Symbol("N", positive=True, integer=True)
    count = Symbol("k", positive=True, integer=True)
    length = Symbol("L", positive=True, integer=True)
    lengths = IndexedBase("d", shape=(4,))
    lower_n = Symbol("n", positive=True, integer=True)
    groups = (
        ("permutation", PermutationDeclaration.from_dense([1, 3, 0, 2])),
        ("unknown_cycles", PermutationDeclaration(size, CycleShape((count,)))),
        (
            "known_cycle_count",
            PermutationDeclaration.from_cycle_lengths(
                size, (lengths[i] for i in range(4))
            ),
        ),
        (
            "sparse_permutation",
            PermutationDeclaration.from_partial(16, {0: 1, 1: 3, 2: 8, 3: 15, 4: 12}),
        ),
        (
            "sparse_symbolic_bound",
            PermutationDeclaration.from_partial(size, {0: 1, 1: 3, 2: 4, 3: 7}),
        ),
    )
    examples: list[PermutationExample] = []
    for name, declaration in groups:
        cycles = declaration.cycles
        values = (
            cycles.shape[0]
            if isinstance(cycles, CycleShape)
            else tuple(
                cycle.shape[0] if isinstance(cycle, CycleShape) else cycle
                for cycle in cycles
            )
        )
        examples.append(
            (name, declaration.signature, values, declaration.direct_calls())
        )
    for name, cycle_declaration in (
        ("cycle_symbolic_bound", PermutationCycleDeclaration(lower_n, (3, 1, 2))),
        ("unknown_cycle", PermutationCycleDeclaration(size, CycleShape((length,)))),
        ("permutation_cycle", PermutationCycleDeclaration(4, (0, 1, 2))),
    ):
        cycle = cycle_declaration.cycle
        value = cycle.shape[0] if isinstance(cycle, CycleShape) else cycle
        examples.append(
            (name, cycle_declaration.signature, value, cycle_declaration.direct_calls())
        )
    return tuple(examples)


def permutation_example_graph(
    example: PermutationExample,
) -> SymbolicCallGraph[Hashable]:
    """Build the original direct resource view without binding or emitting wires."""
    name, _, _, calls = example
    return symbolic_call_graph(name, {name: calls}, max_depth=1)
