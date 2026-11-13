"""Audit the complete locked call-graph notebook through native graph owners."""

import builtins
import hashlib
import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pytest
from sympy import Symbol, simplify

from examples.algorithms.resource_call_graphs import annotated_example, declared_example
from flagquantum.compiler._component_trace.call_graph import symbolic_call_graph
from flagquantum.compiler._component_trace.qec_gate_cost import (
    QECGateCost,
    query_qec_gate_cost,
)
from flagquantum.compiler._component_trace.resource_generalization import ResourceCall
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.drawer.call_graph import export_call_graph_dot, render_call_graph_svg

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT.parent / "ftqc-reference-env"


def execute(notebook: dict[str, Any]) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            exec(
                compile("".join(cell["source"]), f"call-graph-cell-{index}", "exec"),
                namespace,
            )
    return namespace


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    import csv

    with (REFERENCE.parent / "FlagQuantum-Qualtran-Kickmix-移植进度.csv").open(
        encoding="utf-8-sig"
    ) as stream:
        blobs = {row["源文件"]: row["源对象"] for row in csv.DictReader(stream)}
    for path in (
        "drawing/drawing_call_graph.ipynb",
        "drawing/bloq_counts_graph.py",
        "bloqs/for_testing/with_call_graph.py",
        "bloqs/for_testing/with_decomposition.py",
        "bloqs/for_testing/atom.py",
    ):
        raw = (REFERENCE / "qualtran" / path).read_bytes()
        assert (
            hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
            == blobs["qualtran/" + path]
        )


def native_name(node: ResourceCall) -> str:
    assert isinstance(node, ComponentCost)
    return node.name


def source_names() -> dict[str, str]:
    return {
        "TestBloqWithCallGraph": "Declared program",
        "TestParallelCombo": "Parallel",
        "TestSerialCombo": "Serial",
        "TestAtom": "Atom",
        "Split": "Split",
        "Join": "Join",
        **{f"TestAtom('atom{index}')": f"Atom {index}" for index in range(3)},
    }


def test_entire_original_and_native_notebooks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import IPython.display
    import qualtran.drawing
    from qualtran.drawing import GraphvizCallGraph

    source = json.loads(
        (REFERENCE / "qualtran/drawing/drawing_call_graph.ipynb").read_text()
    )
    source_renders: list[bytes] = []
    renderer = GraphvizCallGraph.get_svg_bytes

    def capture(drawer: Any) -> bytes:
        svg: bytes = renderer(drawer)
        source_renders.append(svg)
        return svg

    def show(graph: Any) -> None:
        GraphvizCallGraph(graph).get_svg_bytes()

    with monkeypatch.context() as patch:
        patch.setattr(GraphvizCallGraph, "get_svg_bytes", capture)
        patch.setattr(qualtran.drawing, "show_call_graph", show)
        original = execute(source)
    native_notebook = json.loads(
        (ROOT / "examples/notebooks/resource_call_graphs.ipynb").read_text()
    )
    assert len(source["cells"]) == len(native_notebook["cells"]) == 7
    assert [c["cell_type"] for c in source["cells"]] == [
        c["cell_type"] for c in native_notebook["cells"]
    ]
    assert sum(c["cell_type"] == "code" for c in source["cells"]) == 3
    assert len({c["id"] for c in native_notebook["cells"]}) == 7
    assert all(
        not c["metadata"] and not c.get("outputs") and not c.get("attachments")
        for c in native_notebook["cells"]
    )
    import_function = builtins.__import__

    def native_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq", "quimb", "networkx", "pydot"}:
            raise AssertionError(f"Native example imported {name}")
        return import_function(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", native_import)
        patch.setattr(IPython.display, "display", lambda *args: None)
        native = execute(native_notebook)
    assert len(source_renders) == len(native["renders"]) == 3
    for prefix, renders in (("source", source_renders), ("native", native["renders"])):
        for index, svg in enumerate(renders):
            tree = ElementTree.fromstring(svg)
            groups = tree.findall(".//{http://www.w3.org/2000/svg}g")
            assert sum(g.get("class") == "node" for g in groups) == 9
            assert sum(g.get("class") == "edge" for g in groups) == 9
            (tmp_path / f"{prefix}-{index}.svg").write_bytes(svg)
    names = source_names()
    assert native["annotations"] == {
        next(
            node
            for node in native["annotations"]
            if native_name(node) == names[str(old)]
        ): {key: str(value) for key, value in values.items()}
        for old, values in original["bloq_data"].items()
    }


@pytest.mark.parametrize("depth", [0, 1, 2, 3, None])
def test_complete_graph_and_leaf_counts_match_source(depth: int | None) -> None:
    from qualtran.bloqs.for_testing.with_call_graph import TestBloqWithCallGraph

    source, leaves = TestBloqWithCallGraph().call_graph(max_depth=depth)
    root, declarations = declared_example()
    graph = symbolic_call_graph(root, declarations, max_depth=depth)
    names = source_names()
    assert [
        (native_name(a), native_name(b), count)
        for (a, b), count in graph.direct_calls.items()
    ] == [
        (names[str(a)], names[str(b)], data["n"])
        for a, b, data in source.edges(data=True)
    ]
    assert {native_name(node): count for node, count in graph.leaf_counts.items()} == {
        names[str(node)]: count for node, count in leaves.items()
    }


@pytest.mark.parametrize("repetitions", [0, 1, 2, 7, 1000])
def test_full_recursive_costs_and_annotations_match_source(repetitions: int) -> None:
    from qualtran.bloqs.for_testing.with_call_graph import TestBloqWithCallGraph

    graph, annotations = annotated_example()
    source, _ = TestBloqWithCallGraph().call_graph()
    names = source_names()
    for node in source:
        cost = node.t_complexity()
        native = next(n for n in annotations if native_name(n) == names[str(node)])
        assert annotations[native] == {
            "T count": str(cost.t),
            "clifford": str(cost.clifford),
            "rot": str(cost.rotations),
        }
    root, declarations = declared_example()
    cache: dict[ResourceCall, QECGateCost] = {}
    cost = query_qec_gate_cost(root, declared_calls=declarations, cache=cache)
    assert len(cache) == 9
    n = Symbol("_n0")
    assert simplify(cost.t - (100 * n + 600)) == 0
    assert int((100 * n + 600).subs(n, repetitions)) == 100 * repetitions + 600
    assert (
        graph.estimate_leaf_cost(
            lambda node: 100 if native_name(node).startswith("Atom") else 0
        )
        == cost.t
    )


def test_dot_tables_edges_and_customization_match_source() -> None:
    import pydot
    from qualtran.bloqs.for_testing.with_call_graph import TestBloqWithCallGraph
    from qualtran.drawing import GraphvizCallGraph

    graph, annotations = annotated_example()
    source, _ = TestBloqWithCallGraph().call_graph()
    source_data = {}
    for node in source:
        cost = node.t_complexity()
        source_data[node] = {
            "T count": cost.t,
            "clifford": cost.clifford,
            "rot": cost.rotations,
        }
    parsed = pydot.graph_from_dot_data(
        export_call_graph_dot(
            graph.root, graph.direct_calls, title=native_name, annotations=annotations
        )
    )
    assert parsed is not None and len(parsed) == 1
    native = parsed[0]
    reference = GraphvizCallGraph(source, bloq_data=source_data).get_graph()
    assert native.get("rankdir") == reference.get("rankdir") == "TB"
    for new, old in zip(native.get_nodes(), reference.get_nodes(), strict=True):
        table = ElementTree.fromstring(new.get("label")[1:-1])
        source_table = ElementTree.fromstring(old.get("label")[1:-1])
        assert table.tag == "TABLE" and source_table.tag == "font"
        [source_table] = list(source_table)
        assert source_table.tag == "table"
        rows = [tuple("".join(cell.itertext()) for cell in row) for row in table]
        expected = [
            tuple("".join(cell.itertext()) for cell in row) for row in source_table
        ]
        assert rows[0] == (source_names()[expected[0][0]],)
        assert rows[1:] == expected[1:]
    assert [e.get("label").strip('"') for e in native.get_edges()] == [
        e.get("label").strip('"') for e in reference.get_edges()
    ]
    customized = export_call_graph_dot(
        graph.root,
        graph.direct_calls,
        title=lambda _: "<custom>",
        annotations={graph.root: {"note": "A&B"}},
    )
    assert "&lt;custom&gt;" in customized and "A&amp;B" in customized
    ElementTree.fromstring(render_call_graph_svg(customized))


def test_standalone_example_normal_entry(capsys: pytest.CaptureFixture[str]) -> None:
    import runpy

    runpy.run_path(
        str(ROOT / "examples/algorithms/resource_call_graphs.py"), run_name="__main__"
    )
    output = capsys.readouterr().out
    assert "100*_n0 + 600" in output
    assert "Rendered SVG bytes" in output
