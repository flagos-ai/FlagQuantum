"""Verify connection liveness and greedy scheduling against the locked model."""

import pytest

from flagquantum.compiler._component_trace.width_schedule import (
    WidthConnection,
    WidthStep,
    estimate_connection_width,
    schedule_width_steps,
)

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "case",
    [
        "symbolic",
        "interior",
        "independent",
        "serial",
        "many",
        "once",
        "abstracted",
        "multi_alloc",
        "symbolic_sided",
    ],
)
def test_width_matches_source_connections_and_local_costs(case: str) -> None:
    pytest.importorskip("qualtran")
    import networkx as nx
    from qualtran import BloqBuilder
    from qualtran._infra.binst_graph_iterators import greedy_topological_sort
    from qualtran._infra.binst_graph_iterators_test import MultiAlloc
    from qualtran._infra.quantum_graph import LeftDangle, RightDangle
    from qualtran.bloqs.basic_gates import IntState, Swap
    from qualtran.bloqs.bookkeeping import Allocate, Free
    from qualtran.bloqs.for_testing.interior_alloc import InteriorAlloc
    from qualtran.bloqs.for_testing.qubit_count_many_alloc import (
        TestManyAllocAbstracted,
        TestManyAllocMany,
        TestManyAllocOnce,
    )
    from qualtran.bloqs.for_testing.with_decomposition import (
        TestIndependentParallelCombo,
        TestSerialCombo,
    )
    from qualtran.resource_counting import QubitCount, get_cost_value
    from qualtran.resource_counting._qubit_counts import _cbloq_max_width
    from sympy import Symbol

    n = Symbol("n", positive=True, integer=True)
    if case == "symbolic_sided":
        builder = BloqBuilder()
        x = builder.add(IntState(5, n))
        y = builder.add_register("y", n)
        x, y = builder.add(Swap(n), x=x, y=y)
        x, y = builder.add(Swap(n), x=x, y=y)
        composite, known_peak = builder.finalize(x=x, y=y), 2 * n
    else:
        source, known_peak = {
            "symbolic": (InteriorAlloc(n), 3 * n),
            "interior": (InteriorAlloc(10), 30),
            "independent": (TestIndependentParallelCombo(), 1),
            "serial": (TestSerialCombo(), 1),
            "many": (TestManyAllocMany(10), 11),
            "once": (TestManyAllocOnce(10), 11),
            "abstracted": (TestManyAllocAbstracted(10), 11),
            "multi_alloc": (MultiAlloc(), 2),
        }[case]
        composite = source.decompose_bloq()
    graph = composite._binst_graph
    index = {node: i for i, node in enumerate(graph.nodes)}
    edges = [
        WidthConnection(index[start], index[end], edge.shape)
        for start, end, data in graph.edges(data=True)
        for edge in data["cxns"]
    ]

    def width_step(node, *, use_local_cost: bool) -> WidthStep:
        if node == LeftDangle:
            return WidthStep("input")
        if node == RightDangle:
            return WidthStep("output")
        width = get_cost_value(node.bloq, QubitCount()) if use_local_cost else 0
        if node.bloq_is(Allocate):
            return WidthStep("allocate", width)
        if node.bloq_is(Free):
            return WidthStep("free", width)
        return WidthStep("component", width)

    raw = [width_step(node, use_local_cost=False) for node in index]
    modeled = [width_step(node, use_local_cost=True) for node in index]
    assert schedule_width_steps(raw, edges) == tuple(
        tuple(
            index[node] for node in greedy_topological_sort(graph.subgraph(component))
        )
        for component in nx.weakly_connected_components(graph)
    )
    assert estimate_connection_width(raw, edges) == _cbloq_max_width(graph)
    actual = estimate_connection_width(modeled, edges)
    assert actual == get_cost_value(composite, QubitCount()) == known_peak
    if case == "symbolic":
        assert actual.subs(n, 10) == 30


def test_ready_steps_prefer_shrinking_and_break_ties_by_insertion() -> None:
    # Two parallel calls expand one input to five outputs each. The first
    # call's peak includes eight local qubits, so tie order is observable.
    steps = [
        WidthStep("input"),
        WidthStep("component", 8),
        WidthStep("component", 5),
        WidthStep("output"),
    ]
    edges = [
        WidthConnection(0, 1, 1),
        WidthConnection(0, 2, 1),
        WidthConnection(1, 3, 5),
        WidthConnection(2, 3, 5),
    ]
    assert estimate_connection_width(steps, edges) == 10
    steps[1], steps[2] = steps[2], steps[1]
    assert estimate_connection_width(steps, edges) == 13
    edges[-1] = WidthConnection(2, 3, 0)
    assert estimate_connection_width(steps, edges) == 9


def test_invalid_width_graphs_are_rejected() -> None:
    component = WidthStep("component", 1)
    assert estimate_connection_width([], []) == 0
    with pytest.raises(ValueError, match="outside"):
        estimate_connection_width([component], [WidthConnection(0, 1, 1)])
    with pytest.raises(ValueError, match="nonnegative"):
        estimate_connection_width([component], [WidthConnection(0, 0, -1)])
    with pytest.raises(ValueError, match="cycle"):
        estimate_connection_width([component], [WidthConnection(0, 0, 1)])


@pytest.mark.ftqc_reference
def test_native_add_model_with_bystander_matches_source_composite() -> None:
    pytest.importorskip("qualtran")
    from qualtran import BloqBuilder, QUInt, Signature
    from qualtran.bloqs.arithmetic import Add
    from qualtran.resource_counting import QubitCount, get_cost_value

    from flagquantum.compiler.registers.arithmetic.arithmetic_resource_model import (
        source_add_cost,
    )

    adder = source_add_cost(2, 2)
    steps = [
        WidthStep("input"),
        WidthStep("component", adder.qubits),
        WidthStep("output"),
    ]
    connections = [
        WidthConnection(0, 1, 2),
        WidthConnection(0, 1, 2),
        WidthConnection(1, 2, 2),
        WidthConnection(1, 2, 2),
        WidthConnection(0, 2, 3),
    ]
    builder, ports = BloqBuilder.from_signature(Signature.build(a=2, b=2, spectator=3))
    ports["a"], ports["b"] = builder.add(
        Add(QUInt(2), QUInt(2)), a=ports["a"], b=ports["b"]
    )
    source = builder.finalize(**ports)
    assert (
        estimate_connection_width(steps, connections)
        == get_cost_value(source, QubitCount())
        == 8
    )
