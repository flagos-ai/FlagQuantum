# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 arithmetic/sorting.py.

"""Qualify complete stored sorting definitions against their native owners."""

from examples.algorithms.sorting_decomposition import (
    sorting_symbols,
    symbolic_comparator_program,
)
from flagquantum.compiler.registers.comparison.compare_exchange_component import (
    compare_exchange_program,
)
from flagquantum.compiler.registers.comparison.ordering_declaration import (
    ComparatorDeclaration,
)
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.sorting.network_graph import (
    bitonic_merge_graph,
    bitonic_sort_graph,
    parallel_comparison_graph,
)
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.drawer.register_score_data import RegisterWireSymbolSupplier


def sorting_child_decomposition(
    record: ProgramRecord, entry: str
) -> tuple[str, ProgramRecord, RegisterWireSymbolSupplier] | None:
    """Infer finite constructor candidates, then match the entire reachable body.

    Parallel offset comes from the first named comparator operands. Width and
    array length come from typed ports. A two-word sort requires its enclosing
    exact recursive sort, since a two-word parallel block has the same body.
    Symbolic widths reuse the same finite network and declared comparison leaves.
    """
    function = next(f for f in record.functions if f.name == entry)
    registers = function.registers or ()
    if not registers or any(
        (type(r.bitsize) is int and r.bitsize < 1)
        or any(type(dimension) is not int for dimension in r.shape_symbolic)
        for r in registers
    ):
        return None
    ports = tuple(r.name for r in registers)
    if ports not in {("a", "b", "out"), ("xs", "junk"), ("xs", "ys", "result", "junk")}:
        return None
    width = registers[0].bitsize
    comparator = (
        compare_exchange_program(width)
        if type(width) is int
        else symbolic_comparator_program(ComparatorDeclaration(width).signature)
    )
    candidates: list[tuple[str, ProgramRecord]] = []
    if ports == ("a", "b", "out"):
        candidates.append(("Comparator", comparator))
    elif ports == ("xs", "ys", "result", "junk") and len(registers[0].shape) == 1:
        half = registers[0].shape[0]
        if half > 0 and half & (half - 1) == 0:
            candidates.append(("BitonicMerge", bitonic_merge_graph(half, comparator)))
    elif ports == ("xs", "junk") and len(registers[0].shape) == 1:
        count = registers[0].shape[0]
        sort_context = count >= 4 or (
            count == 2
            and any(
                len(parent.blocks) == 1
                and len(parent.blocks[0].operations) == 3
                and all(op.callee == entry for op in parent.blocks[0].operations[:2])
                and matches_component_definition(
                    record, parent.name, bitonic_sort_graph(2 * count, comparator)
                )
                for parent in record.functions
            )
        )
        if sort_context and count & (count - 1) == 0:
            candidates.append(("BitonicSort", bitonic_sort_graph(count, comparator)))
        if count > 2 and function.blocks and function.blocks[0].operations:
            first = function.blocks[0].operations[0]
            if first.opcode == "call":
                _, inputs, _ = next(iter_component_calls(record, entry))
                a, b = inputs.get("a"), inputs.get("b")
                if (
                    isinstance(a, RegisterBinding)
                    and isinstance(b, RegisterBinding)
                    and a.register.name == b.register.name == "xs"
                    and len(a.index) == len(b.index) == 1
                    and b.index[0] > a.index[0]
                ):
                    candidates.append(
                        (
                            "ParallelComparators",
                            parallel_comparison_graph(
                                count, b.index[0] - a.index[0], comparator
                            ),
                        )
                    )
    return next(
        (
            (label, template, sorting_symbols(template, label))
            for label, template in candidates
            if matches_component_definition(record, entry, template)
        ),
        None,
    )
