"""Verify scoped resource cache priority against the pinned source."""

from collections.abc import Callable

import pytest

from flagquantum.algorithms.primitives.rotation_preparation import RotationPreparation
from flagquantum.algorithms.primitives.rotation_resource_model import (
    rotation_preparation_cost,
)
from flagquantum.compiler._component_trace.cost_query import (
    GATE_COST,
    PEAK_LOGICAL_QUBITS,
    SUCCESS_PROBABILITY,
    CostMetric,
    query_component_cost,
    query_component_costs,
)
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
    estimate_component_resources,
)

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("seed", [None, 0, 9])
@pytest.mark.parametrize("has_static", [False, True])
def test_seed_static_and_recursive_priority(seed: int | None, has_static: bool) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.for_testing.costing import CostingBloq
    from qualtran.resource_counting import get_cost_value
    from qualtran.resource_counting._costing_test import TestCostKey

    atom = ComponentCost("atom", 1)
    alias = ComponentCost("alias", 1)
    omitted = ComponentCost("omitted", 1)
    branch = ComponentCost("branch", 2, calls=(CostCall(atom, 3), CostCall(alias, 4)))
    root = ComponentCost(
        "root", 3, calls=(CostCall(branch, 2), CostCall(atom), CostCall(omitted, 10))
    )
    visited: list[ComponentCost] = []

    def count(node: ComponentCost, child: Callable[[ComponentCost], int]) -> int:
        visited.append(node)
        return 1 + sum(call.repetitions * child(call.component) for call in node.calls)

    cache = {} if seed is None else {branch: seed}
    actual = query_component_cost(
        root,
        CostMetric(count, 0),
        cache=cache,
        static_costs={branch: 12} if has_static else {},
        generalizer=(
            lambda node: None if node == omitted else node,
            lambda node: atom if node == alias else node,
        ),
    )
    source_metric = TestCostKey()
    src_atom = CostingBloq("atom", 1)
    src_alias = CostingBloq("alias", 1)
    src_omitted = CostingBloq("omitted", 1)
    src_branch = CostingBloq(
        "branch",
        2,
        [(src_atom, 3), (src_alias, 4)],
        static_costs=[(source_metric, 12)] if has_static else [],
    )
    src_root = CostingBloq(
        "root", 3, [(src_branch, 2), (src_atom, 1), (src_omitted, 10)]
    )
    source_cache = {} if seed is None else {src_branch: seed}
    expected = get_cost_value(
        src_root,
        source_metric,
        costs_cache=source_cache,
        generalizer=(
            lambda node: None if node == src_omitted else node,
            lambda node: src_atom if node == src_alias else node,
        ),
    )
    assert actual == expected
    assert {node.name: n for node, n in cache.items()} == {
        node.name: n for node, n in source_cache.items()
    }
    assert [node.name for node in visited] == [node.name for node in source_metric._log]


def test_static_and_seeded_metrics_remain_independent() -> None:
    leaf = ComponentCost("leaf", 3, GateCost(t=5), success_probability=0.8)
    parent = ComponentCost(
        "parent", 4, calls=(CostCall(leaf, 3, 2),), success_probability=0.6
    )
    assert query_component_cost(
        parent, GATE_COST, static_costs={leaf: GateCost(t=2)}
    ) == GateCost(t=6)
    assert query_component_cost(parent, PEAK_LOGICAL_QUBITS) == 5
    assert query_component_cost(parent, PEAK_LOGICAL_QUBITS, cache={leaf: 7}) == 9
    assert query_component_cost(parent, SUCCESS_PROBABILITY) == 0.6
    assert (
        query_component_cost(
            parent, SUCCESS_PROBABILITY, cache={parent: 0.0}, static_costs={parent: 0.2}
        )
        == 0.0
    )
    assert query_component_cost(parent, GATE_COST) == GateCost(t=15)


def test_builtin_metrics_agree_with_combined_rotation_estimate() -> None:
    root = rotation_preparation_cost(RotationPreparation((0.5, 0.5j, -0.5, -0.5j), 8))
    report = estimate_component_resources(root)
    assert query_component_cost(root, GATE_COST) == report.gates
    assert query_component_cost(root, PEAK_LOGICAL_QUBITS) == report.peak_logical_qubits
    assert query_component_cost(root, SUCCESS_PROBABILITY) == report.success_probability


def test_removed_root_uses_identity_and_cycles_are_refused() -> None:
    root = ComponentCost("root", 1, calls=(CostCall(ComponentCost("leaf", 1)),))
    cache: dict[ComponentCost, float] = {}
    assert (
        query_component_cost(
            root, SUCCESS_PROBABILITY, cache=cache, generalizer=lambda _: None
        )
        == 1.0
    )
    assert not cache
    with pytest.raises(ValueError, match="cycle"):
        query_component_cost(root, GATE_COST, generalizer=lambda _: root)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("omit_branch", [False, True])
def test_batch_matches_source_metric_rows(omit_branch: bool) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates import TGate
    from qualtran.bloqs.for_testing.costing import CostingBloq
    from qualtran.resource_counting import (
        QECGatesCost,
        QubitCount,
        SuccessProb,
        query_costs,
    )

    leaf = ComponentCost("T", 1, GateCost(t=1))
    alias = ComponentCost("T†", 1, GateCost(t=1))
    branch = ComponentCost(
        "branch", 3, calls=(CostCall(leaf, 2), CostCall(alias)), success_probability=0.5
    )
    root = ComponentCost("root", 2, calls=(CostCall(branch, 3),))
    metrics = (GATE_COST, PEAK_LOGICAL_QUBITS, SUCCESS_PROBABILITY)
    actual = query_component_costs(
        root,
        iter(metrics),
        generalizer=(
            lambda node: None if omit_branch and node == branch else node,
            lambda node: leaf if node == alias else node,
        ),
    )

    t_gate = TGate()
    success = SuccessProb()
    source_branch = CostingBloq(
        "branch",
        3,
        [(t_gate, 2), (t_gate.adjoint(), 1)],
        static_costs=[(success, 0.5)],
    )
    source_root = CostingBloq("root", 2, [(source_branch, 3)])
    source_metrics = (QECGatesCost(), QubitCount(), success)
    expected = query_costs(
        source_root,
        source_metrics,
        generalizer=(
            lambda node: None if omit_branch and node == source_branch else node,
            lambda node: t_gate if node == t_gate.adjoint() else node,
        ),
    )

    source_nodes = {leaf: t_gate, branch: source_branch, root: source_root}
    assert [source_nodes[node] for node in actual] == list(expected)
    for node, row in actual.items():
        source_row = expected[source_nodes[node]]
        assert list(row) == [
            metric
            for metric, source_metric in zip(metrics, source_metrics, strict=True)
            if source_metric in source_row
        ]
        for metric, source_metric in zip(metrics, source_metrics, strict=True):
            if metric in row:
                value = source_row[source_metric]
                assert row[metric] == (
                    GateCost(t=value.t) if metric == GATE_COST else value
                )
    if not omit_branch:
        assert SUCCESS_PROBABILITY not in actual[leaf]
        assert actual[root][SUCCESS_PROBABILITY] == 0.125


def test_batch_with_no_evaluated_nodes_has_no_rows() -> None:
    root = ComponentCost("root", 1)
    assert query_component_costs(root, ()) == {}
    assert query_component_costs(root, (GATE_COST,), generalizer=lambda _: None) == {}
