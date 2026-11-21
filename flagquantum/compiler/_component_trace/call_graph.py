"""Aggregate declared calls and trace exclusive emitted instruction counts."""

# Modeled call graph adapted from Qualtran v0.7.0.
# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/_call_graph.py

from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeAlias, TypeVar, cast

from ...core.classical_control import instruction_condition_clauses
from ...core.ir import CircuitIR
from ...errors import CapabilityError
from .resource_model import ComponentCost
from .symbolic_math import SymbolicInt
from .trace import ComponentCall, ComponentTrace

if TYPE_CHECKING:
    from sympy import Symbol


class SymbolAllocator:
    """Allocate deterministic symbolic placeholders within one graph query."""

    def __init__(self) -> None:
        self._indices: dict[str, int] = {}

    def new_symbol(self, prefix: str) -> "Symbol":
        from sympy import Symbol

        index = self._indices.get(prefix, 0)
        self._indices[prefix] = index + 1
        return Symbol(f"_{prefix}{index}")


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
RawCallees: TypeAlias = Mapping[Node, SymbolicInt] | Iterable[tuple[Node, SymbolicInt]]
CallDeclaration: TypeAlias = Callable[[Node, SymbolAllocator], RawCallees[Node]]


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


# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/_generalization.py
def compose_cost_generalizer(
    generalizer: Generalizer[Node] | Sequence[Generalizer[Node]] | None,
) -> Callable[[Node | None], Node | None]:
    """Apply ordered rewrites until a component is omitted."""
    if generalizer is None:
        generalizers: Sequence[Generalizer[Node]] = ()
    elif callable(generalizer):
        generalizers = (generalizer,)
    else:
        generalizers = generalizer

    def apply(node: Node | None) -> Node | None:
        for transform in generalizers:
            if node is None:
                return None
            node = transform(node)
        return node

    return apply


def _component_calls(node: ComponentCost) -> Iterable[tuple[ComponentCost, int]]:
    for call in node.calls:
        if type(call.repetitions) is not int or call.repetitions < 0:
            raise ValueError("Call multiplicity must be a nonnegative integer")
        yield call.component, call.repetitions


def _generalize_callees(
    raw: RawCallees[Node], generalize: Generalizer[Node]
) -> dict[Node, SymbolicInt]:
    counts: dict[Node, SymbolicInt] = {}
    for child, count in raw.items() if isinstance(raw, Mapping) else raw:
        callee = generalize(child)
        if callee is not None:
            counts[callee] = counts.get(callee, 0) + count
    return counts


def _declared_callees(
    component: Node,
    declaration: CallDeclaration[Node],
    allocator: SymbolAllocator,
    *,
    ignore_unavailable: bool,
) -> RawCallees[Node]:
    try:
        raw = declaration(component, allocator)
        return raw if isinstance(raw, Mapping) else tuple(raw)
    except CapabilityError:
        if not ignore_unavailable:
            raise
        return ()


def component_callee_counts(
    component: Node,
    declaration: CallDeclaration[Node],
    *,
    generalizer: Generalizer[Node] | Sequence[Generalizer[Node]] | None = None,
    allocator: SymbolAllocator | None = None,
    ignore_unavailable: bool = True,
) -> dict[Node, SymbolicInt]:
    """Resolve and combine direct declarations in first occurrence order."""
    raw = _declared_callees(
        component,
        declaration,
        allocator if allocator is not None else SymbolAllocator(),
        ignore_unavailable=ignore_unavailable,
    )
    return _generalize_callees(raw, compose_cost_generalizer(generalizer))


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

    graph = _build_call_graph(component, _component_calls, generalizer, keep, max_depth)
    return ModeledCallGraph(
        graph.root,
        cast(dict[tuple[ComponentCost, ComponentCost], int], graph.direct_calls),
        cast(dict[ComponentCost, int], graph.leaf_counts),
    )


def symbolic_call_graph(
    component: Node,
    declared_calls: Mapping[Node, Mapping[Node, SymbolicInt]] | CallDeclaration[Node],
    *,
    generalizer: Generalizer[Node] | Sequence[Generalizer[Node]] | None = None,
    keep: Callable[[Node], bool] | None = None,
    max_depth: int | None = None,
    allocator: SymbolAllocator | None = None,
) -> SymbolicCallGraph[Node]:
    """Combine shape-dependent call multiplicities without constructing an IR.

    A node absent from declared_calls is terminal. Rewrites replace the node
    key before looking up its outgoing calls. Symbolic weights retain their
    expressions, including weights that may become zero after substitution.
    A callable declaration receives the same symbol allocator at each node.
    A declared capability refusal leaves that node terminal.
    """

    symbols = allocator if allocator is not None else SymbolAllocator()

    def calls(node: Node) -> RawCallees[Node]:
        if callable(declared_calls):
            return _declared_callees(
                node, declared_calls, symbols, ignore_unavailable=True
            )
        return declared_calls.get(node, {})

    return _build_call_graph(component, calls, generalizer, keep, max_depth)


def _call_graph_adjacency(
    components: Iterable[Node],
    calls: Callable[[Node], RawCallees[Node]],
    generalize: Callable[[Node | None], Node | None],
    keep: Callable[[Node], bool] | None,
    max_depth: int | None,
) -> tuple[dict[Node, dict[Node, SymbolicInt]], list[Node]]:
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
            children.update(_generalize_callees(calls(node), generalize))
            active.add(node)
            for child in children:
                visit(child, depth + 1)
            active.remove(node)
        postorder.append(node)

    for root in components:
        visit(root, 0)
    return graph, postorder


def symbolic_call_graph_union(
    components: Iterable[Node],
    declared_calls: CallDeclaration[Node],
    *,
    generalizer: Generalizer[Node] | Sequence[Generalizer[Node]] | None = None,
    keep: Callable[[Node], bool] | None = None,
) -> dict[Node, dict[Node, SymbolicInt]]:
    """Build the source library's shared first-visit graph of concrete roots.

    Roots retain their identities. Only callees are generalized. Each visited
    declaration receives a fresh allocator, matching the library source path.
    A shared node is expanded once across all roots. Cycles are rejected by the
    same traversal used for single-root graphs.
    """

    def calls(node: Node) -> RawCallees[Node]:
        return _declared_callees(
            node, declared_calls, SymbolAllocator(), ignore_unavailable=True
        )

    graph, _ = _call_graph_adjacency(
        components, calls, compose_cost_generalizer(generalizer), keep, None
    )
    return graph


def _build_call_graph(
    component: Node,
    calls: Callable[[Node], RawCallees[Node]],
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

    graph, postorder = _call_graph_adjacency(
        (root,), calls, generalize, keep, max_depth
    )
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
