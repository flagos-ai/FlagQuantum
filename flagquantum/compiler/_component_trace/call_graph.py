"""Aggregate declared calls and trace exclusive emitted instruction counts."""

# Modeled call graph adapted from Qualtran v0.7.0.
# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/_call_graph.py

from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar, cast

from ...core.classical_control import instruction_condition_clauses
from ...core.ir import CircuitIR
from .resource_model import ComponentCost
from .symbolic_math import SymbolicInt
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


Node = TypeVar("Node", bound=Hashable)
Generalizer = Callable[[Node], Node | None]
CostGeneralizer = Generalizer[ComponentCost]


@dataclass(frozen=True)
class SymbolicCallGraph(Generic[Node]):
    """Declared direct calls and weighted terminal counts with symbolic weights."""

    root: Node
    direct_calls: dict[tuple[Node, Node], SymbolicInt]
    leaf_counts: dict[Node, SymbolicInt]

    def estimate_leaf_cost(
        self, unit_cost: Callable[[Node], SymbolicInt]
    ) -> SymbolicInt:
        """Sum one additive metric using a declared cost for each terminal."""
        return sum(count * unit_cost(node) for node, count in self.leaf_counts.items())


def compose_cost_generalizer(
    generalizer: Generalizer[Node] | Sequence[Generalizer[Node]] | None,
) -> Generalizer[Node]:
    """Apply ordered rewrites until a component is omitted."""
    if generalizer is None:
        generalizers: Sequence[Generalizer[Node]] = ()
    elif callable(generalizer):
        generalizers = (generalizer,)
    else:
        generalizers = generalizer

    def apply(node: Node) -> Node | None:
        current = node
        for transform in generalizers:
            updated = transform(current)
            if updated is None:
                return None
            current = updated
        return current

    return apply


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

    def calls(node: ComponentCost) -> Iterable[tuple[ComponentCost, int]]:
        for call in node.calls:
            if type(call.repetitions) is not int or call.repetitions < 0:
                raise ValueError("Call multiplicity must be a nonnegative integer")
            yield call.component, call.repetitions

    graph = _build_call_graph(component, calls, generalizer, keep, max_depth)
    return ModeledCallGraph(
        graph.root,
        cast(dict[tuple[ComponentCost, ComponentCost], int], graph.direct_calls),
        cast(dict[ComponentCost, int], graph.leaf_counts),
    )


def symbolic_call_graph(
    component: Node,
    declared_calls: Mapping[Node, Mapping[Node, SymbolicInt]],
    *,
    generalizer: Generalizer[Node] | Sequence[Generalizer[Node]] | None = None,
    keep: Callable[[Node], bool] | None = None,
    max_depth: int | None = None,
) -> SymbolicCallGraph[Node]:
    """Combine shape-dependent call multiplicities without constructing an IR.

    A node absent from declared_calls is terminal. Rewrites replace the node
    key before looking up its outgoing calls. Symbolic weights retain their
    expressions, including weights that may become zero after substitution.
    """

    def calls(node: Node) -> Iterable[tuple[Node, SymbolicInt]]:
        return declared_calls.get(node, {}).items()

    return _build_call_graph(component, calls, generalizer, keep, max_depth)


def _build_call_graph(
    component: Node,
    calls: Callable[[Node], Iterable[tuple[Node, SymbolicInt]]],
    generalizer: Generalizer[Node] | Sequence[Generalizer[Node]] | None,
    keep: Callable[[Node], bool] | None,
    max_depth: int | None,
) -> SymbolicCallGraph[Node]:
    if max_depth is not None and (type(max_depth) is not int or max_depth < 0):
        raise ValueError("Call graph depth must be a nonnegative integer")

    generalize = compose_cost_generalizer(generalizer)
    root = generalize(component)
    if root is None:
        raise ValueError("Cannot remove the root of a call graph")

    graph: dict[Node, dict[Node, SymbolicInt]] = {}
    active: set[Node] = set()
    postorder: list[Node] = []

    def visit(node: Node, depth: int) -> None:
        if node in active:
            raise ValueError("Modeled call graph contains a cycle")
        if node in graph:
            return
        children: dict[Node, SymbolicInt] = {}
        graph[node] = children
        if (keep is None or not keep(node)) and (
            max_depth is None or depth < max_depth
        ):
            for child, count in calls(node):
                callee = generalize(child)
                if callee is not None:
                    children[callee] = children.get(callee, 0) + count
            active.add(node)
            for child in children:
                visit(child, depth + 1)
            active.remove(node)
        postorder.append(node)

    visit(root, 0)
    totals: dict[Node, SymbolicInt] = {root: 1}
    leaves: dict[Node, SymbolicInt] = {}
    for node in reversed(postorder):
        if not graph[node]:
            leaves[node] = totals[node]
        for child, count in graph[node].items():
            totals[child] = totals.get(child, 0) + totals[node] * count

    return SymbolicCallGraph(
        root,
        {
            (node, child): count
            for node, children in graph.items()
            for child, count in children.items()
        },
        leaves,
    )
