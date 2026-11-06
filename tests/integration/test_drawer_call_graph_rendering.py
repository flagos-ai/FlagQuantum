"""Exercise real Graphviz rendering and the locked Qualtran drawing fixture."""

from __future__ import annotations

import shutil
from xml.etree import ElementTree

import pytest

from flagquantum.algorithms.primitives.multiplexers.thc_resource_model import (
    select_thc_cost,
)
from flagquantum.drawer.call_graph import (
    export_call_graph_dot,
    export_modeled_call_graph_dot,
    render_call_graph_svg,
)

pytestmark = pytest.mark.integration

_SVG = {"svg": "http://www.w3.org/2000/svg"}


@pytest.fixture
def dot() -> str:
    executable = shutil.which("dot")
    if executable is None:
        pytest.skip("Real call-graph rendering requires Graphviz")
    return executable


def _svg_labels(
    svg: bytes,
) -> tuple[dict[str, tuple[str, ...]], dict[tuple[str, str], tuple[str, ...]]]:
    tree = ElementTree.fromstring(svg)
    nodes = {}
    titles = {}
    edges = {}
    for group in tree.findall(".//svg:g[@class='node']", _SVG):
        node_id = group.findtext("svg:title", namespaces=_SVG)
        assert node_id is not None
        title, *details = tuple(
            text.text or "" for text in group.findall("svg:text", _SVG)
        )
        titles[node_id] = title
        nodes[title] = tuple(details)
    for group in tree.findall(".//svg:g[@class='edge']", _SVG):
        edge_id = group.findtext("svg:title", namespaces=_SVG)
        assert edge_id is not None
        caller, callee = edge_id.split("->")
        edges[titles[caller], titles[callee]] = tuple(
            text.text or "" for text in group.findall("svg:text", _SVG)
        )
    return nodes, edges


def test_svg_rendering_thc_resource_cutoff(dot: str) -> None:
    model = select_thc_cost(2, 4, 8, 1)
    source = export_modeled_call_graph_dot(model, max_depth=0, aggregation="total_t")
    nodes, edges = _svg_labels(render_call_graph_svg(source, executable=dot))
    assert nodes == {"SelectTHC": ("Qubits", "24", "Ts", "472")}
    assert edges == {}


def test_svg_surfaces_graphviz_diagnostic(dot: str) -> None:
    with pytest.raises(RuntimeError, match="Graphviz SVG rendering failed:"):
        render_call_graph_svg("not a DOT graph", executable=dot)


@pytest.mark.ftqc_reference
def test_svg_labels_match_locked_graphviz_fixture(dot: str) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.for_testing import TestBloqWithCallGraph
    from qualtran.drawing import GraphvizCallGraph

    reference = GraphvizCallGraph.from_bloq(
        TestBloqWithCallGraph(), max_depth=1, agg_gate_counts="t_and_ccz"
    )
    direct = {
        (caller, callee): data["n"]
        for caller, callee, data in reference.g.edges(data=True)
    }
    root = next(iter(reference.g))
    source = export_call_graph_dot(root, direct, annotations=reference.bloq_data)
    actual = _svg_labels(render_call_graph_svg(source, executable=dot))
    expected = _svg_labels(reference.get_svg_bytes())
    assert actual == expected
    nodes, edges = actual
    assert len(nodes) > 1
    assert edges
    assert nodes["TestBloqWithCallGraph"] == (
        "Qubits",
        "3",
        "Ts",
        "100*_n0 + 600",
        "CCZs",
        "0",
    )
