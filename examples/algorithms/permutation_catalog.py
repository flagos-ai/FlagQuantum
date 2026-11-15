# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 arithmetic/permutation.py.

"""Native metadata for the complete original permutation examples."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable
from typing import TypeAlias

from sympy import IndexedBase, Symbol

from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt, bit_length
from flagquantum.compiler.registers.sorting.cycle_decomposition import (
    partial_permutation_cycles,
    permutation_cycles,
)
from flagquantum.core.ir.program.register_types import bounded_unsigned_register_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature

Cycle: TypeAlias = "tuple[int, ...] | SymbolicInt"
PermutationExample = tuple[str, Signature, Hashable, dict[Hashable, SymbolicInt]]


def permutation_examples() -> tuple[PermutationExample, ...]:
    """Preserve every factory's cycles, unknown shapes, bounds and call identities."""
    size = Symbol("N", positive=True, integer=True)
    count = Symbol("k", positive=True, integer=True)
    length = Symbol("L", positive=True, integer=True)
    lengths = IndexedBase("d", shape=(4,))
    lower_n = Symbol("n", positive=True, integer=True)
    groups: tuple[tuple[str, SymbolicInt, tuple[Cycle, ...] | SymbolicInt], ...] = (
        ("permutation", 4, tuple(permutation_cycles([1, 3, 0, 2]))),
        ("unknown_cycles", size, count),
        ("known_cycle_count", size, tuple(lengths[i] for i in range(4))),
        (
            "sparse_permutation",
            16,
            tuple(partial_permutation_cycles({0: 1, 1: 3, 2: 8, 3: 15, 4: 12})),
        ),
        (
            "sparse_symbolic_bound",
            size,
            tuple(partial_permutation_cycles({0: 1, 1: 3, 2: 4, 3: 7})),
        ),
    )
    examples: list[PermutationExample] = []
    for name, bound, cycles in groups:
        width = bit_length(bound - 1)
        declared = cycles if isinstance(cycles, tuple) else (bound,)
        calls: dict[Hashable, SymbolicInt] = dict(
            Counter(("cycle", bound, width, cycle) for cycle in declared)
        )
        examples.append((name, _signature(width, bound), cycles, calls))
    for name, bound, cycle in (
        ("cycle_symbolic_bound", lower_n, (3, 1, 2)),
        ("unknown_cycle", size, length),
        ("permutation_cycle", 4, (0, 1, 2)),
    ):
        width = bit_length(bound - 1)
        examples.append(
            (name, _signature(width, bound), cycle, _cycle_calls(width, cycle))
        )
    return tuple(examples)


def _signature(width: SymbolicInt, bound: SymbolicInt) -> Signature:
    return Signature((Register("x", bounded_unsigned_register_type(width, bound)),))


def _cycle_calls(width: SymbolicInt, cycle: Cycle) -> dict[Hashable, SymbolicInt]:
    if not isinstance(cycle, tuple):
        return {
            ("equals", width, "unknown"): cycle + 1,
            ("controlled_xor", width, "unknown"): cycle,
        }
    calls: Counter[Hashable] = Counter({("allocate_marker",): 1})
    for value, successor in zip(cycle, (*cycle[1:], cycle[0]), strict=True):
        calls[("equals", width, value)] += 1
        calls[("controlled_xor", width, value ^ successor)] += 1
    calls[("equals", width, cycle[0])] += 1
    calls[("free_marker",)] = 1
    return dict(calls)


def permutation_example_graph(
    example: PermutationExample,
) -> SymbolicCallGraph[Hashable]:
    """Build the original direct resource view without binding or emitting wires."""
    name, _, _, calls = example
    return symbolic_call_graph(name, {name: calls}, max_depth=1)
