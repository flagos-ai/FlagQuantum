"""Locked source parity for direct and weighted terminal component calls."""

from dataclasses import replace

import pytest

from flagquantum.compiler._component_trace.call_graph import modeled_call_graph
from flagquantum.compiler._component_trace.resource_model import ComponentCost, CostCall

pytestmark = pytest.mark.unit


def diamond(*, deep_first: bool = True) -> ComponentCost:
    leaf = ComponentCost("leaf", 1)
    shared = ComponentCost("shared", 1, calls=(CostCall(leaf, 7),))
    left = ComponentCost("left", 1, calls=(CostCall(shared, 3),))
    right = ComponentCost("right", 1, calls=(CostCall(shared, 5),))
    calls = (CostCall(left, 2), CostCall(right, 4), CostCall(shared, 1))
    return ComponentCost(
        "root",
        1,
        calls=(
            *(calls if deep_first else calls[::-1]),
            CostCall(ComponentCost("zero", 1), 0),
        ),
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("depth", [None, 0, 1, 2, 3])
@pytest.mark.parametrize("deep_first", [False, True])
def test_weighted_shared_graph_and_first_visit_depth(
    depth: int | None, deep_first: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.for_testing.costing import CostingBloq
    from qualtran.resource_counting import get_bloq_call_graph

    model = diamond(deep_first=deep_first)

    def source(node: ComponentCost) -> CostingBloq:
        return CostingBloq(
            node.name,
            node.qubits,
            [(source(call.component), call.repetitions) for call in node.calls],
        )

    expected_graph, sigma = get_bloq_call_graph(source(model), max_depth=depth)
    graph = modeled_call_graph(model, max_depth=depth)
    assert {
        (a.name, b.name): count for (a, b), count in graph.direct_calls.items()
    } == {(a.name, b.name): data["n"] for a, b, data in expected_graph.edges.data()}
    assert {node.name: count for node, count in graph.leaf_counts.items()} == {
        node.name: count for node, count in sigma.items()
    }


@pytest.mark.ftqc_reference
def test_generalizer_chain_omission_merging_and_keep() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.for_testing.costing import CostingBloq
    from qualtran.resource_counting import get_bloq_call_graph

    terminal = ComponentCost("terminal", 1)
    branch = ComponentCost("branch", 1, calls=(CostCall(terminal, 8),))
    root = ComponentCost(
        "root",
        1,
        calls=(
            CostCall(replace(branch, name="a"), 2),
            CostCall(replace(branch, name="b"), 3),
            CostCall(terminal, 20),
        ),
    )

    def filter_node(node: ComponentCost) -> ComponentCost | None:
        return None if node.name == "terminal" else node

    def merge_node(node: ComponentCost) -> ComponentCost:
        return branch if node.name in {"a", "b"} else node

    graph = modeled_call_graph(
        root,
        generalizer=(filter_node, merge_node),
        keep=lambda node: node.name == "branch",
    )
    src_terminal = CostingBloq("terminal", 1)
    src_branch = CostingBloq("branch", 1, [(src_terminal, 8)])
    src_root = CostingBloq(
        "root",
        1,
        [
            (CostingBloq("a", 1, [(src_terminal, 8)]), 2),
            (CostingBloq("b", 1, [(src_terminal, 8)]), 3),
            (src_terminal, 20),
        ],
    )
    expected, sigma = get_bloq_call_graph(
        src_root,
        generalizer=(
            lambda node: None if node.name == "terminal" else node,
            lambda node: src_branch if node.name in {"a", "b"} else node,
        ),
        keep=lambda node: node.name == "branch",
    )
    assert {(a.name, b.name): n for (a, b), n in graph.direct_calls.items()} == {
        (a.name, b.name): data["n"] for a, b, data in expected.edges.data()
    }
    assert (
        {node.name: n for node, n in graph.leaf_counts.items()}
        == {node.name: n for node, n in sigma.items()}
        == {"branch": 5}
    )


def test_reject_root_removal_cycles_and_invalid_counts() -> None:
    root = diamond()
    with pytest.raises(ValueError, match="remove the root"):
        modeled_call_graph(root, generalizer=lambda _: None)
    with pytest.raises(ValueError, match="cycle"):
        modeled_call_graph(root, generalizer=lambda _: root)
    with pytest.raises(ValueError, match="depth"):
        modeled_call_graph(root, max_depth=-1)
    with pytest.raises(ValueError, match="multiplicity"):
        modeled_call_graph(replace(root, calls=(CostCall(root, -1),)))


def test_rotation_lookup_leaf_counts_from_declared_model() -> None:
    from collections import Counter

    from flagquantum.algorithms.primitives.rotation_preparation import (
        RotationPreparation,
    )
    from flagquantum.algorithms.primitives.rotation_resource_model import (
        rotation_preparation_cost,
    )

    model = rotation_preparation_cost(RotationPreparation((0.5, 0.5j, -0.5, -0.5j), 8))
    graph = modeled_call_graph(model)
    totals: Counter[str] = Counter()
    for component, count in graph.leaf_counts.items():
        totals[component.name] += count
    assert totals == {"QROM": 6, "AddIntoPhaseGrad": 3}
    stopped = modeled_call_graph(
        model, keep=lambda node: node.name == "PhaseGradientLookup"
    )
    assert sum(stopped.leaf_counts.values()) == 3
    assert {node.name for node in stopped.leaf_counts} == {"PhaseGradientLookup"}
