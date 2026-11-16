# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CostKey, get_cost_value, and query_costs.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/_costing.py
# Success probability follows the product rule in the pinned source.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/_success_prob.py

"""Query declared component costs with independent caches for each metric."""

from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, cast

from .call_graph import (
    CallDeclaration,
    Generalizer,
    Node,
    SymbolAllocator,
    component_callee_counts,
    compose_cost_generalizer,
)
from .resource_model import (
    ComponentCost,
    GateCost,
    _active_calls,
    _component_width,
    _declared_success,
)
from .symbolic_math import SymbolicInt, SymbolicReal

Value = TypeVar("Value")


@dataclass(frozen=True)
class CostMetric(Generic[Node, Value]):
    """A recursive resource rule and the cost of an omitted component."""

    compute: Callable[[Node, Callable[[Node], Value]], Value]
    zero: Value
    validate_static: Callable[[Value], None] | None = None

    def cost_cache(
        self,
        component: Node,
        generalizer: Generalizer[Node] | Sequence[Generalizer[Node]] | None = None,
        *,
        cache: dict[Node, Value] | None = None,
        static_costs: Mapping[Node, Value] | None = None,
    ) -> dict[Node, Value]:
        """Return the visited costs, extending the same cache if one is supplied."""
        stored: dict[Node, Value] = {} if cache is None else cache
        query_component_cost(
            component,
            self,
            cache=stored,
            static_costs=static_costs,
            generalizer=generalizer,
        )
        return stored


class _BatchMetric(Protocol[Node]):
    def cost_cache(
        self, component: Node, generalizer: Generalizer[Node]
    ) -> Mapping[Node, object]: ...


def query_component_cost(
    component: Node,
    metric: CostMetric[Node, Value],
    *,
    cache: MutableMapping[Node, Value] | None = None,
    static_costs: Mapping[Node, Value] | None = None,
    generalizer: Generalizer[Node] | Sequence[Generalizer[Node]] | None = None,
) -> Value:
    """Generalize, then prefer a seeded cache, static override or recursive cost.

    A caller-provided cache is updated in place using generalized component
    keys. Use one cache per metric and generalizer/override configuration.
    """
    return _query_cost(
        component,
        metric.compute,
        metric.zero,
        compose_cost_generalizer(generalizer),
        cache,
        static_costs,
        validate_static=metric.validate_static,
    )


def _query_cost(
    root: Node,
    compute: Callable[[Node, Callable[[Node], Value]], Value],
    zero: Value,
    generalize: Generalizer[Node],
    cache: MutableMapping[Node, Value] | None,
    static_costs: Mapping[Node, Value] | None,
    *,
    validate_static: Callable[[Value], None] | None = None,
    is_composite: Callable[[Node], bool] | None = None,
) -> Value:
    stored: MutableMapping[Node, Value] = {} if cache is None else cache
    visiting: set[Node] = set()

    def visit(original: Node) -> Value:
        node = generalize(original)
        if node is None:
            return zero
        cacheable = is_composite is None or not is_composite(node)
        if cacheable and node in stored:
            return stored[node]
        if static_costs is not None and node in static_costs:
            value = static_costs[node]
            if validate_static is not None:
                validate_static(value)
            stored[node] = value
            return value
        if node in visiting:
            raise ValueError("Component cost query contains a cycle")
        visiting.add(node)
        try:
            value = compute(node, visit)
        finally:
            visiting.remove(node)
        if cacheable:
            stored[node] = value
        return value

    return visit(root)


def query_component_costs(
    component: Node,
    metrics: Iterable[_BatchMetric[Node]],
    *,
    generalizer: Generalizer[Node] | Sequence[Generalizer[Node]] | None = None,
) -> dict[Node, dict[_BatchMetric[Node], object]]:
    """Collect the visited component rows for each metric in request order.

    Each metric gets a fresh cache and may visit a different set of nodes.
    A row contains only metrics that evaluated or retrieved that node.
    """
    table: dict[Node, dict[_BatchMetric[Node], object]] = {}
    generalize = compose_cost_generalizer(generalizer)
    for metric in metrics:
        for node, value in metric.cost_cache(component, generalize).items():
            table.setdefault(node, {})[metric] = value
    return table


def _gate_cost(
    node: ComponentCost, child: Callable[[ComponentCost], GateCost]
) -> GateCost:
    gates = node.gates
    for call in _active_calls(node):
        gates = gates.plus(child(call.component), call.repetitions)
    return gates


def _width_cost(node: ComponentCost, child: Callable[[ComponentCost], int]) -> int:
    return max(
        _component_width(node),
        max(
            (
                child(call.component) + call.bystander_qubits
                for call in _active_calls(node)
            ),
            default=0,
        ),
    )


def _success_cost(
    node: ComponentCost, child: Callable[[ComponentCost], float]
) -> float:
    declared = _declared_success(node)
    if declared is not None:
        return declared
    return cast(
        float,
        _success_product(
            (child(call.component), call.repetitions) for call in _active_calls(node)
        ),
    )


def _success_product(
    callees: Iterable[tuple[SymbolicReal, SymbolicInt]],
) -> SymbolicReal:
    success: SymbolicReal = 1.0
    for probability, repetitions in callees:
        success = cast(SymbolicReal, success * probability**repetitions)
    return success


GATE_COST = CostMetric(_gate_cost, GateCost())
PEAK_LOGICAL_QUBITS = CostMetric(_width_cost, 0)
SUCCESS_PROBABILITY = CostMetric(_success_cost, 1.0)


def query_declared_success_probability(
    root: Node,
    declared_calls: Mapping[Node, Mapping[Node, SymbolicInt]] | CallDeclaration[Node],
    *,
    cache: MutableMapping[Node, SymbolicReal] | None = None,
    static_costs: Mapping[Node, SymbolicReal] | None = None,
    generalizer: Generalizer[Node] | Sequence[Generalizer[Node]] | None = None,
    allocator: SymbolAllocator | None = None,
) -> SymbolicReal:
    """Multiply declared probabilities with concrete or symbolic repetitions.

    Seeded and static costs use the existing recursive query semantics.
    Each declared callee is visited, including a zero-multiplicity callee.
    The result models success of a component and its calls.
    """
    symbols = allocator if allocator is not None else SymbolAllocator()

    def compute(node: Node, child: Callable[[Node], SymbolicReal]) -> SymbolicReal:
        counts = (
            component_callee_counts(node, declared_calls, allocator=symbols)
            if callable(declared_calls)
            else declared_calls.get(node, {})
        )
        return _success_product(
            (child(callee), count) for callee, count in counts.items()
        )

    return _query_cost(
        root,
        compute,
        SUCCESS_PROBABILITY.zero,
        compose_cost_generalizer(generalizer),
        cache,
        static_costs,
    )
