"""Compare full native resource fixture graphs and override precedence."""

from dataclasses import asdict

import pytest
from sympy import Symbol

from flagquantum.compiler._component_trace.call_graph import symbolic_call_graph
from flagquantum.compiler._component_trace.cost_query import (
    PEAK_LOGICAL_QUBITS,
    query_component_cost,
)
from flagquantum.compiler._component_trace.qec_gate_cost import (
    QECGateCost,
    query_qec_gate_cost,
)
from flagquantum.compiler._component_trace.resource_generalization import (
    DeclaredResourceCalls,
    PhaseCall,
    ResourceCall,
)
from tests.unit.declared_cost_fixtures import (
    DeclaredCostFixture,
    StaticKey,
    costing_example,
    debug_edges,
    fixture_label,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_complete_declared_fixture_matches_locked_graph() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.for_testing.costing import make_example_costing_bloqs
    from qualtran.resource_counting import (
        QECGatesCost,
        QubitCount,
        format_call_graph_debug_text,
        get_cost_value,
    )

    root, *children = costing_example()
    source = make_example_costing_bloqs()
    edges, counts = source.call_graph()
    declarations: DeclaredResourceCalls = {
        node: node.declared_calls() for node in (root, *children)
    }
    root_key: ResourceCall = root
    graph = symbolic_call_graph(root_key, declarations)
    assert debug_edges(graph) == format_call_graph_debug_text(edges)
    assert {
        fixture_label(node): count for node, count in graph.leaf_counts.items()
    } == {str(node): count for node, count in counts.items()}
    assert get_cost_value(source, QubitCount()) == root.signature.n_qubits() == 100
    modeled = query_qec_gate_cost(root, declared_calls=declarations)
    assert {key: value for key, value in asdict(modeled).items() if value} == (
        get_cost_value(source, QECGatesCost()).asdict()
    )
    assert modeled == QECGateCost(t=40, toffoli=100, clifford=20)


def test_source_duplicate_conversion_and_static_override() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates import TGate
    from qualtran.bloqs.for_testing.costing import CostingBloq
    from qualtran.resource_counting import (
        GateCounts,
        QECGatesCost,
        QubitCount,
        format_call_graph_debug_text,
        get_cost_value,
    )

    n = Symbol("n", nonnegative=True, integer=True)
    leaf = PhaseCall("t")
    pairs = [(leaf, 2), (leaf, n)]
    costs: list[tuple[StaticKey, QECGateCost | int]] = [
        ("qec", QECGateCost(t=5)),
        ("qec", QECGateCost(t=11)),
        ("width", 8),
    ]
    node = DeclaredCostFixture("override", 0, callees=pairs, static_costs=costs)
    source = CostingBloq(
        "override",
        0,
        [(TGate(), 2), (TGate(), n)],
        [
            (QECGatesCost(), GateCounts(t=5)),
            (QECGatesCost(), GateCounts(t=11)),
            (QubitCount(), 8),
        ],
    )
    pairs.clear()
    costs.clear()
    assert isinstance(node.callees, tuple) and isinstance(node.static_costs, tuple)
    assert node != DeclaredCostFixture("override", 0, callees=[(leaf, n)])
    source_calls, _ = source.call_graph()
    actual: DeclaredResourceCalls = {node: node.declared_calls()}
    root_key: ResourceCall = node
    assert debug_edges(symbolic_call_graph(root_key, actual)) == (
        format_call_graph_debug_text(source_calls)
    )
    assert query_qec_gate_cost(node, declared_calls=actual).t == n
    overrides = dict(node.static_costs)
    qec, width = overrides["qec"], overrides["width"]
    assert isinstance(qec, QECGateCost) and isinstance(width, int)
    assert query_qec_gate_cost(
        node, declared_calls=actual, static_costs={node: qec}
    ).t == (get_cost_value(source, QECGatesCost()).t)
    assert query_component_cost(
        node, PEAK_LOGICAL_QUBITS, static_costs={node: width}
    ) == (get_cost_value(source, QubitCount()))
    assert tuple(node.signature) == tuple(source.signature) == ()
