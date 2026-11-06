"""Check instance ports and dependencies against the locked Graphviz adapter."""

import shutil
from collections import Counter
from xml.etree import ElementTree

import pytest

from flagquantum.drawer.call_graph import render_call_graph_svg
from flagquantum.drawer.register_graph import (
    RegisterConnection,
    export_register_graph_dot,
    register_connections,
)
from flagquantum.drawer.register_score import (
    RegisterStep,
    layout_register_score,
    multi_and_register_score,
)

pytestmark = pytest.mark.unit
_SVG = {"svg": "http://www.w3.org/2000/svg"}


def test_instance_graph_tracks_distinct_lifetimes_and_through_ports() -> None:
    score = layout_register_score(
        {"control": 1, "a<&>": 3, "b": 2},
        ("control", "a<&>"),
        (
            RegisterStep("replace <old>", ("a<&>", "control"), ("b", "control")),
            RegisterStep("finish", ("b",), ()),
        ),
    )
    assert register_connections(score) == (
        RegisterConnection("a<&>", 3, None, 0),
        RegisterConnection("control", 1, None, 0),
        RegisterConnection("b", 2, 0, 1),
        RegisterConnection("control", 1, 0, None),
    )
    dot = export_register_graph_dot(score)
    assert "replace &lt;old&gt;" in dot
    assert "a&lt;&amp;&gt;" in dot
    assert dot.count('COLSPAN="2" PORT=') == 1


@pytest.mark.ftqc_reference
def test_multi_and_register_graph_matches_locked_instance_connections() -> None:
    pytest.importorskip("qualtran")
    import pydot
    from qualtran import BloqInstance
    from qualtran.bloqs.mcmt import MultiAnd
    from qualtran.drawing.graphviz import GraphDrawer
    from qualtran.drawing.musical_score import _cbloq_musical_score

    values = (1, 1, 0, 1)
    composite = MultiAnd(values).decompose_bloq()
    source_drawer = GraphDrawer(composite)
    _, positions, _ = _cbloq_musical_score(composite.signature, composite._binst_graph)
    score = multi_and_register_score(values)
    by_lane = {line.lane: line.register for line in score.lines}
    expected = Counter(
        RegisterConnection(
            by_lane[positions[edge.left].y],
            edge.shape,
            edge.left.binst.i if isinstance(edge.left.binst, BloqInstance) else None,
            edge.right.binst.i if isinstance(edge.right.binst, BloqInstance) else None,
        )
        for edge in composite.connections
    )
    assert Counter(register_connections(score)) == expected

    [actual] = pydot.graph_from_dot_data(export_register_graph_dot(score))
    reference = source_drawer.get_graph()
    assert actual.get_rankdir() == reference.get_rankdir() == "LR"
    assert len(actual.get_nodes()) == len(reference.get_nodes()) == 3
    assert len(actual.get_edges()) == len(reference.get_edges()) == 13
    assert [len(group.get_nodes()) for group in actual.get_subgraphs()] == [4, 7]
    node = actual.get_node("n1")[0].get_label()
    assert node.index("junk[0]") < node.index("ctrl[2]") < node.index("junk[1]")


def test_real_multi_and_graphviz_renders_all_register_connections() -> None:
    if shutil.which("dot") is None:
        pytest.skip("Real register graph rendering requires Graphviz")
    graph = export_register_graph_dot(multi_and_register_score((1, 1, 0, 1)))
    svg = ElementTree.fromstring(render_call_graph_svg(graph))
    nodes = svg.findall(".//svg:g[@class='node']", _SVG)
    edges = svg.findall(".//svg:g[@class='edge']", _SVG)
    assert len(nodes) == 14
    assert len(edges) == 13
    labels = [text.text for node in nodes for text in node.findall("svg:text", _SVG)]
    assert labels.count("And") == 3
    assert "junk[0]" in labels and "junk[1]" in labels and "target" in labels
