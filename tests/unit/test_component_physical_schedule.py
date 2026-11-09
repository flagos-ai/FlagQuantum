"""Compare whole-graph physical visitation with the locked source scheduler."""

import pytest

from flagquantum.compiler._component_trace.width_schedule import (
    WidthConnection,
    WidthStep,
    schedule_width_steps,
)

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "case",
    ["multi_alloc", "independent", "serial", "interior", "partial_free", "joined_free"],
)
def test_global_queue_matches_actual_source_graph_insertion_and_greedy_order(case):
    from qualtran import BloqBuilder, QAny, Signature
    from qualtran._infra.binst_graph_iterators import greedy_topological_sort
    from qualtran._infra.binst_graph_iterators_test import MultiAlloc
    from qualtran._infra.quantum_graph import LeftDangle, RightDangle
    from qualtran.bloqs.bookkeeping import Allocate, Free
    from qualtran.bloqs.for_testing.interior_alloc import InteriorAlloc
    from qualtran.bloqs.for_testing.with_decomposition import (
        TestIndependentParallelCombo,
        TestSerialCombo,
    )

    if case in {"partial_free", "joined_free"}:
        builder, inputs = BloqBuilder.from_signature(Signature.build(data=1))
        if case == "partial_free":
            word = builder.allocate(dtype=QAny(3))
            bits = builder.split(word)
            for bit in reversed(bits):
                builder.free(bit)
        else:
            first = builder.allocate(dtype=QAny(1))
            second = builder.allocate(dtype=QAny(2))
            joined = builder.join([*builder.split(second), *builder.split(first)])
            builder.free(joined)
        composite = builder.finalize(**inputs)
    else:
        source = {
            "multi_alloc": MultiAlloc(),
            "independent": TestIndependentParallelCombo(),
            "serial": TestSerialCombo(),
            "interior": InteriorAlloc(3),
        }[case]
        composite = source.decompose_bloq()
    graph = composite._binst_graph
    index = {node: i for i, node in enumerate(graph.nodes)}
    steps = []
    for node in index:
        if node == LeftDangle:
            steps.append(WidthStep("input"))
        elif node == RightDangle:
            steps.append(WidthStep("output"))
        elif node.bloq_is(Allocate):
            steps.append(WidthStep("allocate"))
        elif node.bloq_is(Free):
            steps.append(WidthStep("free"))
        else:
            steps.append(WidthStep("component"))
    edges = [
        WidthConnection(index[start], index[end], connection.shape)
        for start, end, data in graph.edges(data=True)
        for connection in data["cxns"]
    ]
    assert schedule_width_steps(steps, edges, group_components=False) == (
        tuple(index[node] for node in greedy_topological_sort(graph)),
    )


def test_global_order_can_interleave_independent_components_without_changing_default():
    steps = [
        WidthStep("allocate"),
        WidthStep("free"),
        WidthStep("input"),
        WidthStep("output"),
    ]
    edges = [WidthConnection(0, 1, 1), WidthConnection(2, 3, 1)]
    assert schedule_width_steps(steps, edges) == ((0, 1), (2, 3))
    assert schedule_width_steps(steps, edges, group_components=False) == ((2, 3, 0, 1),)


def test_global_queue_preserves_isolated_nodes_parallel_edges_and_empty_graph():
    steps = [WidthStep("component"), WidthStep("component"), WidthStep("component")]
    edges = [WidthConnection(0, 1, 1), WidthConnection(0, 1, 2)]
    assert schedule_width_steps(steps, edges, group_components=False) == ((2, 0, 1),)
    assert schedule_width_steps([], [], group_components=False) == ()
    with pytest.raises(ValueError, match="cycle"):
        schedule_width_steps(
            steps, [*edges, WidthConnection(1, 0, 1)], group_components=False
        )


@pytest.mark.parametrize("grouping", [None, 0, "global"])
def test_invalid_grouping_is_rejected_at_entry(grouping):
    with pytest.raises(ValueError, match="built-in boolean"):
        schedule_width_steps([], [], group_components=grouping)
