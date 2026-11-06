"""Aggregate declared calls and trace exclusive emitted instruction counts."""

# Modeled call graph adapted from Qualtran v0.7.0.
# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/_call_graph.py

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ...core.classical_control import instruction_condition_clauses
from ...core.ir import CircuitIR
from .resource_model import ComponentCost
from .trace import ComponentCall, ComponentTrace


def summarize_components(
    ir: CircuitIR, trace: ComponentTrace
) -> tuple[Counter[tuple[str, str]], Counter[tuple[str, str, bool]]]:
    """Return edge multiplicities and counts by component, opcode and condition.

    Instructions outside recorded roots belong to the circuit. A parent
    owns the instructions in its interval that are outside child intervals.
    """
    edges: Counter[tuple[str, str]] = Counter()
    gates: Counter[tuple[str, str, bool]] = Counter()

    def add_gates(name: str, start: int, stop: int) -> None:
        gates.update(
            (name, op.name, bool(instruction_condition_clauses(op)))
            for op in ir.instructions[start:stop]
        )

    def visit(
        parent: str, start: int, stop: int, calls: tuple[ComponentCall, ...]
    ) -> None:
        cursor = start
        for call in calls:
            if not cursor <= call.start <= call.stop <= stop:
                raise ValueError("Component intervals must be ordered and nested")
            add_gates(parent, cursor, call.start)
            edges[parent, call.name] += 1
            visit(call.name, call.start, call.stop, call.children)
            cursor = call.stop
        add_gates(parent, cursor, stop)

    visit("Circuit", 0, len(ir.instructions), tuple(trace.calls))
    return edges, gates


@dataclass(frozen=True)
class ModeledCallGraph:
    """Direct edge weights and weighted terminal counts of a declared model."""

    root: ComponentCost
    direct_calls: dict[tuple[ComponentCost, ComponentCost], int]
    leaf_counts: dict[ComponentCost, int]


CostGeneralizer = Callable[[ComponentCost], ComponentCost | None]


def modeled_call_graph(
    component: ComponentCost,
    *,
    generalizer: CostGeneralizer | Sequence[CostGeneralizer] | None = None,
    keep: Callable[[ComponentCost], bool] | None = None,
    max_depth: int | None = None,
) -> ModeledCallGraph:
    """Count declared calls with the locked source's first-visit cutoff semantics.

    Apply generalizers in order before combining identical callees. A child
    mapped to None is omitted. Structural equality identifies shared nodes.
    A zero edge remains in the graph and contributes zero terminal calls.
    """
    if max_depth is not None and (type(max_depth) is not int or max_depth < 0):
        raise ValueError("Call graph depth must be a nonnegative integer")

    if generalizer is None:
        generalizers: Sequence[CostGeneralizer] = ()
    elif callable(generalizer):
        generalizers = (generalizer,)
    else:
        generalizers = generalizer

    def generalize(node: ComponentCost) -> ComponentCost | None:
        current = node
        for transform in generalizers:
            updated = transform(current)
            if updated is None:
                return None
            current = updated
        return current

    root = generalize(component)
    if root is None:
        raise ValueError("Cannot remove the root of a call graph")

    graph: dict[ComponentCost, Counter[ComponentCost]] = {}
    active: set[ComponentCost] = set()
    postorder: list[ComponentCost] = []

    def visit(node: ComponentCost, depth: int) -> None:
        if node in active:
            raise ValueError("Modeled call graph contains a cycle")
        if node in graph:
            return
        children: Counter[ComponentCost] = Counter()
        graph[node] = children
        if (keep is None or not keep(node)) and (
            max_depth is None or depth < max_depth
        ):
            for call in node.calls:
                if type(call.repetitions) is not int or call.repetitions < 0:
                    raise ValueError("Call multiplicity must be a nonnegative integer")
                callee = generalize(call.component)
                if callee is not None:
                    children[callee] += call.repetitions
            active.add(node)
            for child in children:
                visit(child, depth + 1)
            active.remove(node)
        postorder.append(node)

    visit(root, 0)
    totals: Counter[ComponentCost] = Counter({root: 1})
    leaves: dict[ComponentCost, int] = {}
    for node in reversed(postorder):
        if not graph[node]:
            leaves[node] = totals[node]
        for child, count in graph[node].items():
            totals[child] += totals[node] * count

    return ModeledCallGraph(
        root,
        {
            (node, child): count
            for node, children in graph.items()
            for child, count in children.items()
        },
        leaves,
    )
