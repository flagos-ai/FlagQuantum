"""Original greedy-sort tests run against the existing native queue owner."""

from collections.abc import Iterator
from hashlib import sha1
from pathlib import Path
from typing import Any

import pytest
import sympy

from flagquantum.compiler._component_trace.width_schedule import (
    WidthConnection,
    WidthStep,
    _priority,
    schedule_width_steps,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def native_sort(graph: Any) -> Iterator[Any]:
    """Translate a reference graph only inside the differential test boundary."""
    import networkx as nx
    from qualtran import LeftDangle, RightDangle
    from qualtran.bloqs.bookkeeping import Allocate, Free

    assert isinstance(graph, nx.DiGraph)
    nodes = tuple(graph.nodes)
    index = {node: position for position, node in enumerate(nodes)}
    steps = []
    for node in nodes:
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
    (order,) = schedule_width_steps(steps, edges, group_components=False)
    yield from (nodes[position] for position in order)


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    from qualtran._infra import binst_graph_iterators

    for name, blob in (
        ("binst_graph_iterators.py", "d22ebbcc5ac642964cc61f7793b24d6dcdc13261"),
        ("binst_graph_iterators_test.py", "dabd2b34806ff8f1c8db33b92af054a4bd00b1e8"),
    ):
        raw = (Path(binst_graph_iterators.__file__).parent / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


def test_both_original_tests_with_native_scheduler_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qualtran._infra import binst_graph_iterators_test as original

    monkeypatch.setattr(original, "greedy_topological_sort", native_sort)
    original.test_greedy_topological_sort()
    original.test_topo_sort_with_symbolic_registers()


def test_all_source_priority_branches_with_complete_signature_edges() -> None:
    from qualtran import BloqInstance, LeftDangle, QAny, RightDangle
    from qualtran._infra.binst_graph_iterators import _priority as source_priority
    from qualtran.bloqs.basic_gates import CNOT, IntEffect, IntState
    from qualtran.bloqs.bookkeeping import Allocate, Free

    n = sympy.Symbol("n", integer=True, positive=True)
    fixtures = (
        (LeftDangle, WidthStep("input"), (), ()),
        (RightDangle, WidthStep("output"), (), ()),
        (BloqInstance(Allocate(QAny(3)), 0), WidthStep("allocate"), (), ()),
        (BloqInstance(Free(QAny(3)), 1), WidthStep("free"), (), ()),
        (
            BloqInstance(IntState(1, 3), 2),
            WidthStep("component"),
            (),
            (WidthConnection(0, 1, 3),),
        ),
        (
            BloqInstance(IntEffect(1, 3), 3),
            WidthStep("component"),
            (WidthConnection(0, 1, 3),),
            (),
        ),
        (
            BloqInstance(CNOT(), 4),
            WidthStep("component"),
            (WidthConnection(0, 1, 2),),
            (WidthConnection(1, 2, 2),),
        ),
        (
            BloqInstance(IntState(1, n), 5),
            WidthStep("component"),
            (),
            (WidthConnection(0, 1, n),),
        ),
    )
    for node, step, incoming, outgoing in fixtures:
        assert _priority(step, incoming, outgoing) == source_priority(node)


def test_source_mutation_failure_and_native_finite_snapshot_boundary() -> None:
    import networkx as nx
    from qualtran import BloqInstance
    from qualtran._infra.binst_graph_iterators import greedy_topological_sort
    from qualtran.bloqs.basic_gates import XGate

    first, second = (BloqInstance(XGate(), i) for i in range(2))
    source = nx.DiGraph()
    source.add_edge(first, second, cxns=())
    stream = greedy_topological_sort(source)
    assert next(stream) == first
    source.remove_node(second)
    with pytest.raises(RuntimeError, match="Graph changed during iteration"):
        tuple(stream)
    # Native execution consumes validated finite immutable Program graphs.
    assert schedule_width_steps(
        (WidthStep("component"), WidthStep("component")),
        (WidthConnection(0, 1, 1),),
        group_components=False,
    ) == ((0, 1),)
