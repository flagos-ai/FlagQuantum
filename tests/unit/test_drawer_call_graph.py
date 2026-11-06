"""Check declared call-graph presentation against locked source formatters."""

from dataclasses import asdict
from typing import get_args

import pytest

from flagquantum.compiler._component_trace.call_graph import (
    modeled_call_graph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from flagquantum.drawer.call_graph import (
    GateAggregation,
    export_call_graph_dot,
    export_modeled_call_graph_dot,
    format_call_graph_markdown,
    render_call_graph_svg,
)

pytestmark = pytest.mark.unit


def test_svg_requires_graphviz() -> None:
    with pytest.raises(ImportError, match="requires Graphviz"):
        render_call_graph_svg(
            "digraph counts {}", executable="/nonexistent-flagquantum-graphviz/dot"
        )


def test_shared_leaf_weights_and_distinct_node_ids() -> None:
    leaf = ComponentCost("leaf", 1)
    first = ComponentCost("branch", 2, calls=(CostCall(leaf, 2),))
    second = ComponentCost("branch", 3, calls=(CostCall(leaf, 4),))
    root = ComponentCost("<root>", 4, calls=(CostCall(first, 3), CostCall(second, 5)))
    graph = modeled_call_graph(root)

    assert format_call_graph_markdown(
        graph.root, graph.direct_calls, graph.leaf_counts, title=lambda node: node.name
    ).endswith("#### Counts totals:\n - `leaf`: 26")
    dot = export_modeled_call_graph_dot(root)
    assert dot.count('COLSPAN="2">branch</TD>') == 2
    assert "&lt;root&gt;" in dot
    assert 'n1 -> n3 [label="2"]' in dot
    assert 'n2 -> n3 [label="4"]' in dot


@pytest.mark.ftqc_reference
def test_markdown_matches_locked_multiand() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.mcmt.and_bloq import MultiAnd
    from qualtran.drawing import format_counts_graph_markdown, format_counts_sigma

    graph, totals = MultiAnd(cvs=(1,) * 6).call_graph()
    root = next(iter(graph))
    direct = {(src, dst): data["n"] for src, dst, data in graph.edges(data=True)}
    assert format_call_graph_markdown(root, direct, totals) == (
        format_counts_graph_markdown(graph) + format_counts_sigma(totals)
    )


@pytest.mark.ftqc_reference
def test_symbolic_export_preserves_expressions_and_zero_calls() -> None:
    pytest.importorskip("qualtran")
    import networkx as nx
    import pydot
    import sympy
    from qualtran.drawing import (
        GraphvizCallGraph,
        format_counts_graph_markdown,
        format_counts_sigma,
    )

    n = sympy.Symbol("n", integer=True)
    graph = symbolic_call_graph(
        "root", {"root": {"left": n + 1, "right": 0}, "left": {"leaf": 2 * n}}
    )
    source = nx.DiGraph()
    source.add_edges_from(
        (caller, callee, {"n": count})
        for (caller, callee), count in graph.direct_calls.items()
    )
    assert format_call_graph_markdown(
        graph.root, graph.direct_calls, graph.leaf_counts
    ) == format_counts_graph_markdown(source) + format_counts_sigma(graph.leaf_counts)
    [parsed] = pydot.graph_from_dot_data(
        export_call_graph_dot(graph.root, graph.direct_calls)
    )
    assert parsed.get_rankdir() == "TB"
    reference = GraphvizCallGraph(source).get_graph()
    assert [edge.get_label().strip('"') for edge in parsed.get_edges()] == [
        edge.get_label().strip('"') for edge in reference.get_edges()
    ]


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("aggregation", get_args(GateAggregation))
def test_truncated_graph_annotates_full_model(aggregation: GateAggregation) -> None:
    pytest.importorskip("qualtran")
    from qualtran.drawing import GraphvizCallGraph
    from qualtran.resource_counting import GateCounts

    gates = GateCost(
        t=2, toffoli=1, cswap=2, and_bloq=3, clifford=4, rotation=5, measurement=6
    )
    child = ComponentCost("hidden", 5, gates)
    root = ComponentCost("root", 1, calls=(CostCall(child, bystander_qubits=3),))
    dot = export_modeled_call_graph_dot(root, max_depth=0, aggregation=aggregation)
    reference = GraphvizCallGraph.format_qec_gates_cost(
        GateCounts(**asdict(gates)), agg=aggregation
    )
    assert "hidden" not in dot
    assert "<TD>Qubits</TD><TD>8</TD>" in dot
    for label, count in reference.items():
        assert f"<TD>{label}</TD><TD>{count}</TD>" in dot
    assert dot.count("<TR>") == 2 + len(reference)
