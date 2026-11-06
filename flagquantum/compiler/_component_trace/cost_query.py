# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CostKey, get_cost_value, and query_costs.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/_costing.py

"""Query declared component costs with independent caches for each metric."""

from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from .call_graph import CostGeneralizer, Generalizer, Node, compose_cost_generalizer
from .resource_model import (
    ComponentCost,
    GateCost,
    _active_calls,
    _component_width,
    _declared_success,
)

Value = TypeVar("Value")


@dataclass(frozen=True)
class CostMetric(Generic[Value]):
    """A recursive resource rule and the cost of an omitted component."""

    compute: Callable[[ComponentCost, Callable[[ComponentCost], Value]], Value]
    zero: Value

    def cost_cache(
        self, component: ComponentCost, generalizer: CostGeneralizer
    ) -> dict[ComponentCost, Value]:
        """Collect the values visited by this metric from an empty cache."""
        cache: dict[ComponentCost, Value] = {}
        query_component_cost(component, self, cache=cache, generalizer=generalizer)
        return cache


class _BatchMetric(Protocol):
    def cost_cache(
        self, component: ComponentCost, generalizer: CostGeneralizer
    ) -> Mapping[ComponentCost, object]: ...


def query_component_cost(
    component: ComponentCost,
    metric: CostMetric[Value],
    *,
    cache: MutableMapping[ComponentCost, Value] | None = None,
    static_costs: Mapping[ComponentCost, Value] | None = None,
    generalizer: CostGeneralizer | Sequence[CostGeneralizer] | None = None,
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
    )


def _query_cost(
    root: Node,
    compute: Callable[[Node, Callable[[Node], Value]], Value],
    zero: Value,
    generalize: Generalizer[Node],
    cache: MutableMapping[Node, Value] | None,
    static_costs: Mapping[Node, Value] | None,
) -> Value:
    stored: MutableMapping[Node, Value] = {} if cache is None else cache
    visiting: set[Node] = set()

    def visit(original: Node) -> Value:
        node = generalize(original)
        if node is None:
            return zero
        if node in stored:
            return stored[node]
        if static_costs is not None and node in static_costs:
            value = static_costs[node]
        else:
            if node in visiting:
                raise ValueError("Component cost query contains a cycle")
            visiting.add(node)
            value = compute(node, visit)
            visiting.remove(node)
        stored[node] = value
        return value

    return visit(root)


def query_component_costs(
    component: ComponentCost,
    metrics: Iterable[_BatchMetric],
    *,
    generalizer: CostGeneralizer | Sequence[CostGeneralizer] | None = None,
) -> dict[ComponentCost, dict[_BatchMetric, object]]:
    """Collect the visited component rows for each metric in request order.

    Each metric gets a fresh cache and may visit a different set of nodes.
    A row contains only metrics that evaluated or retrieved that node.
    """
    table: dict[ComponentCost, dict[_BatchMetric, object]] = {}
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
    success = 1.0
    for call in _active_calls(node):
        success *= child(call.component) ** call.repetitions
    return success


GATE_COST = CostMetric(_gate_cost, GateCost())
PEAK_LOGICAL_QUBITS = CostMetric(_width_cost, 0)
SUCCESS_PROBABILITY = CostMetric(_success_cost, 1.0)
