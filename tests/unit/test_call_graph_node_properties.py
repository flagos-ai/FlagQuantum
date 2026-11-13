"""Compare callback-owned call-graph labels and properties with locked drawers."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pydot
import pytest

from flagquantum.compiler._component_trace.resource_generalization import (
    ControlledSwapCall,
    ResourceCall,
)
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from flagquantum.drawer.call_graph import (
    export_call_graph_dot,
    export_modeled_call_graph_dot,
    export_resource_call_graph_dot,
    render_call_graph_svg,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
REFERENCE = Path(__file__).resolve().parents[3] / "ftqc-reference-env"


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    with (REFERENCE.parent / "FlagQuantum-Qualtran-Kickmix-移植进度.csv").open(
        encoding="utf-8-sig"
    ) as stream:
        blobs = {row["源文件"]: row["源对象"] for row in csv.DictReader(stream)}
    raw = (REFERENCE / "qualtran/drawing/bloq_counts_graph.py").read_bytes()
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == blobs["qualtran/drawing/bloq_counts_graph.py"]
    )


def properties_from_dot(dot: str) -> list[dict[str, str]]:
    graphs = pydot.graph_from_dot_data(dot)
    assert graphs is not None and len(graphs) == 1
    graph: Any = graphs[0]
    result = []
    for node in graph.get_nodes():
        attributes = node.get_attributes()
        if not attributes:
            continue
        result.append(
            {
                json.loads(key) if key.startswith('"') else key: (
                    json.loads(value) if value.startswith('"') else value
                )
                for key, value in attributes.items()
            }
        )
    return result


@pytest.mark.parametrize("title", ["root", '<x & "y">', "Unicode θ", "same\ntitle"])
@pytest.mark.parametrize("detail", ["", "T count = 4", '<a & "b">\nsecond line'])
def test_rectangular_details_match_unmodified_base(title: str, detail: str) -> None:
    import networkx as nx
    from qualtran.drawing.bloq_counts_graph import _CallGraphDrawerBase

    def source_title(self: Any, node: object) -> str:
        return title

    def source_detail(self: Any, node: object) -> str:
        return detail

    # Subclass the untyped external drawer at the reference boundary.
    reference: Any = type(
        "Reference",
        (_CallGraphDrawerBase,),
        {"get_node_title": source_title, "get_node_details": source_detail},
    )(nx.DiGraph([(0, 1, {"n": 2})]))
    calls: list[tuple[str, int]] = []

    def title_for(node: int) -> str:
        calls.append(("title", node))
        return title

    def detail_for(node: int) -> str:
        calls.append(("detail", node))
        return detail

    dot = export_call_graph_dot(0, {(0, 1): 2}, title=title_for, details=detail_for)
    expected = [reference.get_node_properties(node) for node in (0, 1)]
    assert properties_from_dot(dot) == expected
    assert calls == [("title", 0), ("detail", 0), ("title", 1), ("detail", 1)]
    native_svg = ElementTree.fromstring(render_call_graph_svg(dot))
    source_svg = ElementTree.fromstring(reference.get_svg_bytes())
    assert [
        element.text for element in native_svg.iter() if element.tag.endswith("}text")
    ] == [
        element.text for element in source_svg.iter() if element.tag.endswith("}text")
    ]


@pytest.mark.parametrize(
    "properties",
    [
        {"shape": "ellipse", "color": "blue", "label": "custom"},
        {
            "shape": "box",
            "style": "filled",
            "fillcolor": "#aabbcc",
            "label": 'a "quote"',
        },
        {"label": "first\nsecond", "tooltip": 'a "quote" and \\ path'},
        {"label": "<<B>Bold &amp; safe</B>>", "shape": "plain"},
        {"label": "custom", "xlabel": "<<I>outside</I>>", "fontcolor": "red"},
        {"label": "", "shape": "point"},
    ],
)
def test_complete_properties_replace_source_labels_without_mutation(
    properties: dict[str, str],
) -> None:
    import networkx as nx
    from qualtran.drawing import GraphvizCallGraph

    def source_properties(self: Any, node: object) -> dict[str, str]:
        return properties

    reference_type = type(
        "Reference", (GraphvizCallGraph,), {"get_node_properties": source_properties}
    )

    def forbidden(node: str) -> str:
        pytest.fail("A complete property replacement must bypass label callbacks")

    visited = []
    before = dict(properties)

    def attributes(node: str) -> Mapping[str, str]:
        visited.append(node)
        return properties

    dot = export_call_graph_dot(
        "root",
        {("root", "leaf"): 3},
        title=forbidden,
        details=forbidden,
        annotations={"root": {"ignored": "row"}},
        node_properties=attributes,
    )
    reference = reference_type(nx.DiGraph([("root", "leaf")]))
    reference.g.edges["root", "leaf"]["n"] = 3
    assert properties_from_dot(dot) == [
        reference.get_node_properties(n) for n in ("root", "leaf")
    ]
    assert visited == ["root", "leaf"] and properties == before
    native_svg = ElementTree.fromstring(render_call_graph_svg(dot))
    source_svg = ElementTree.fromstring(reference.get_svg_bytes())
    assert [
        element.text for element in native_svg.iter() if element.tag.endswith("}text")
    ] == [
        element.text for element in source_svg.iter() if element.tag.endswith("}text")
    ]


def test_property_values_cannot_create_additional_nodes_or_edges() -> None:
    text = 'x"]; injected -> other; //\n\\ tail'
    dot = export_call_graph_dot(
        "root", {}, node_properties=lambda node: {"label": text, "tooltip": text}
    )
    assert properties_from_dot(dot) == [{"label": text, "tooltip": text}]
    graphs = pydot.graph_from_dot_data(dot)
    assert graphs is not None
    assert len(graphs[0].get_nodes()) == 1 and not graphs[0].get_edges()
    ElementTree.fromstring(render_call_graph_svg(dot))


@pytest.mark.parametrize("depth", [0, 1, None])
@pytest.mark.parametrize("mode", ["detail", "properties"])
def test_typed_and_concrete_factories_forward_only_visible_callbacks(
    depth: int | None,
    mode: str,
) -> None:
    root: ResourceCall = ControlledSwapCall(3, approximate=True)
    visited: list[object] = []

    def detail(node: object) -> str:
        visited.append(node)
        return "declared cost"

    def attributes(node: object) -> Mapping[str, str]:
        visited.append(node)
        return {"label": "custom cost", "shape": "box"}

    dot = export_resource_call_graph_dot(
        root,
        logical_width=lambda node: 7 if node == root else None,
        max_depth=depth,
        details=detail if mode == "detail" else None,
        node_properties=attributes if mode == "properties" else None,
    )
    assert len(visited) == (1 if depth == 0 else 4)
    assert visited[0] == root and len(visited) == len(set(visited))
    ElementTree.fromstring(render_call_graph_svg(dot))
    visited.clear()
    leaf = ComponentCost("leaf", 1, GateCost(t=2))
    concrete = ComponentCost("root", 1, calls=(CostCall(leaf, 3),))
    concrete_dot = export_modeled_call_graph_dot(
        concrete,
        max_depth=depth,
        details=detail if mode == "detail" else None,
        node_properties=attributes if mode == "properties" else None,
    )
    assert len(visited) == (1 if depth == 0 else 2)
    assert visited[0] == concrete
    ElementTree.fromstring(render_call_graph_svg(concrete_dot))


@pytest.mark.parametrize("mode", ["title", "detail", "properties"])
def test_callback_failure_propagates_and_retry_owns_fresh_state(mode: str) -> None:
    failure = RuntimeError("callback failed")

    def bad(node: str) -> Any:
        raise failure

    with pytest.raises(RuntimeError) as actual:
        export_call_graph_dot(
            "root",
            {},
            title=bad if mode == "title" else str,
            details=bad if mode == "detail" else None,
            node_properties=bad if mode == "properties" else None,
        )
    assert actual.value is failure
    dot = export_call_graph_dot("root", {}, details=lambda node: "fresh")
    assert properties_from_dot(dot)[0]["shape"] == "rect"


def test_cycles_fail_before_display_callbacks() -> None:
    def bad(node: str) -> Mapping[str, str]:
        pytest.fail("Cyclic graph must fail before rendering nodes")

    with pytest.raises(ValueError):
        export_call_graph_dot("root", {("root", "root"): 1}, node_properties=bad)
