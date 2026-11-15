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
from flagquantum.compiler.registers.comparison.parallel_comparison import (
    parallel_comparison_pairs,
)
from flagquantum.compiler.registers.sorting.bitonic_merge import (
    bitonic_merge_history_count,
)
from flagquantum.compiler.registers.sorting.bitonic_sort import (
    bitonic_sort_history_count,
)
from flagquantum.core.ir.program.register_types import (
    quantum_bit_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature

SortingExample = tuple[str, Signature, dict[Hashable, SymbolicInt]]


def sorting_examples() -> tuple[SortingExample, ...]:
    """Preserve every factory's port directions, array shapes and child identities."""
    n = Symbol("n")
    bit = quantum_bit_type()
    examples: list[SortingExample] = []
    for name, width in (("comparator", 3), ("comparator_symbolic", n)):
        word = unsigned_register_type(width)
        signature = Signature(
            (
                Register("a", word),
                Register("b", word),
                Register("out", bit, side=RegisterSide.RIGHT),
            )
        )
        examples.append(
            (
                name,
                signature,
                {
                    ("allocate_bit",): 1,
                    ("greater_than", width, width): 1,
                    ("controlled_swap", width): 1,
                },
            )
        )
    for name, count, width, history in (
        ("bitonic_sort", 8, 4, bitonic_sort_history_count(8)),
        ("bitonic_sort_symbolic", 4, n, bitonic_sort_history_count(4)),
        ("parallel_comparators", 7, 3, len(parallel_comparison_pairs(7, 2))),
    ):
        signature = Signature(
            (
                Register("xs", unsigned_register_type(width), shape=(count,)),
                Register("junk", bit, shape=(history,), side=RegisterSide.RIGHT),
            )
        )
        examples.append((name, signature, {("comparator", width): history}))
    word = unsigned_register_type(7)
    history = bitonic_merge_history_count(4)
    signature = Signature(
        (
            Register("xs", word, shape=(4,), side=RegisterSide.LEFT),
            Register("ys", word, shape=(4,), side=RegisterSide.LEFT),
            Register("result", word, shape=(8,), side=RegisterSide.RIGHT),
            Register("junk", bit, shape=(history,), side=RegisterSide.RIGHT),
        )
    )
    examples.append(("bitonic_merge", signature, {("comparator", 7): history}))
    return tuple(examples)


def sorting_example_graph(example: SortingExample) -> SymbolicCallGraph[Hashable]:
    """Build the exact one-level graph without constructing executable gates."""
    name, _, calls = example
    return symbolic_call_graph(name, {name: calls}, max_depth=1)
