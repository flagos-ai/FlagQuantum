# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Execute unchanged assertions from the Qualtran v0.7.0 call-graph tests.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/drawing/bloq_counts_graph_test.py

"""Audit the complete source test asset through native graph and cost owners."""

import ast
import csv
import hashlib
import html
import runpy
from collections.abc import Callable, Mapping
from pathlib import Path
from types import FunctionType
from typing import Any
from xml.etree import ElementTree

import pytest

from examples.algorithms.resource_call_graphs import declared_example
from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    _build_call_graph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.cost_query import _query_cost
from flagquantum.compiler._component_trace.gate_budget import t_and_ccz_budget
from flagquantum.compiler._component_trace.qec_gate_cost import (
    QECGateCost,
    query_qec_gate_cost,
)
from flagquantum.compiler._component_trace.resource_generalization import (
    ConjunctionCall,
    ResourceCall,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.compiler._component_trace.terminal_t_cost import (
    _terminal_resource_callees,
)
from flagquantum.drawer.call_graph import (
    export_call_graph_dot,
    format_call_graph_markdown,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
REFERENCE = Path(__file__).resolve().parents[3] / "ftqc-reference-env"
SOURCE = REFERENCE / "qualtran/drawing/bloq_counts_graph_test.py"
FUNCTIONS = (
    "test_format_counts_sigma",
    "test_format_counts_graph_markdown",
    "test_graphviz_call_graph_no_data",
    "test_graphviz_call_graph_with_data",
    "test_graphviz_call_graph_from_bloq",
    "test_graphviz_call_graph_from_bloq_agg",
    "test_graphviz_call_graph_from_bloq_max_depth",
)


@pytest.fixture(scope="module", autouse=True)
def source_tests() -> dict[str, Any]:
    with (REFERENCE.parent / "FlagQuantum-Qualtran-Kickmix-移植进度.csv").open(
        encoding="utf-8-sig"
    ) as stream:
        blobs = {row["源文件"]: row["源对象"] for row in csv.DictReader(stream)}
    for path in (
        "drawing/bloq_counts_graph_test.py",
        "drawing/bloq_counts_graph.py",
        "bloqs/for_testing/with_call_graph.py",
        "bloqs/for_testing/with_decomposition.py",
        "bloqs/for_testing/atom.py",
        "bloqs/mcmt/and_bloq.py",
    ):
        raw = (REFERENCE / "qualtran" / path).read_bytes()
        assert (
            hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
            == blobs["qualtran/" + path]
        )
    functions = [
        n.name
        for n in ast.parse(SOURCE.read_bytes()).body
        if isinstance(n, ast.FunctionDef)
    ]
    assert functions == [
        *FUNCTIONS[:2],
        "_get_node_labels_from_pydot_graph",
        *FUNCTIONS[2:],
    ]
    return runpy.run_path(str(SOURCE))


def reference_title(node: ResourceCall) -> str:
    """Translate only display names at the reference test boundary."""
    if isinstance(node, ConjunctionCall):
        assert isinstance(node.controls, tuple)
        return "And" if len(node.controls) == 2 else f"MultiAnd(n={len(node.controls)})"
    assert isinstance(node, ComponentCost)
    names = {
        "Declared program": "TestBloqWithCallGraph",
        "Parallel": "TestParallelCombo",
        "Serial": "TestSerialCombo",
        "Atom": "TestAtom",
    }
    if node.name.startswith("Atom "):
        return f"TestAtom('atom{node.name[-1]}')"
    return names.get(node.name, node.name)


def native_conjunction_graph(
    call: ConjunctionCall,
) -> tuple[SymbolicCallGraph[ResourceCall], dict[ResourceCall, SymbolicInt]]:
    root: ResourceCall = call
    graph = _build_call_graph(
        root, lambda node: _terminal_resource_callees(node, None), None, None, None
    )
    return graph, dict(graph.leaf_counts)


def native_sigma(sigma: Mapping[ResourceCall, SymbolicInt]) -> str:
    # Select the totals section from the existing combined native formatter.
    return format_call_graph_markdown(
        ConjunctionCall((1, 1)), {}, sigma, title=reference_title
    )


def native_markdown(graph: SymbolicCallGraph[ResourceCall]) -> str:
    text = format_call_graph_markdown(
        graph.root, graph.direct_calls, graph.leaf_counts, title=reference_title
    )
    return text.split("#### Counts totals:", 1)[0]


def native_annotated_dot(
    max_depth: int | None, *, aggregate: bool = False
) -> tuple[SymbolicCallGraph[ResourceCall], str]:
    """Compose the fixture's zero-bystander declared width and QEC costs.

    The width callback models this fixture's sequential declarations. It claims
    no arbitrary register-graph liveness inference.
    """
    root, declarations = declared_example()
    graph = symbolic_call_graph(root, declarations, max_depth=max_depth)
    costs: dict[ResourceCall, QECGateCost] = {}
    query_qec_gate_cost(root, declared_calls=declarations, cache=costs)
    widths: dict[ResourceCall, int] = {}

    def width(node: ResourceCall, child: Callable[[ResourceCall], int]) -> int:
        assert isinstance(node, ComponentCost)
        return max([node.qubits, *(child(c) for c in declarations.get(node, {}))])

    _query_cost(root, width, 0, lambda node: node, widths, None)
    annotations = {}
    for node, cost in costs.items():
        assert all(
            value == 0
            for value in (
                cost.clifford,
                cost.rotation,
                cost.toffoli,
                cost.cswap,
                cost.and_bloq,
                cost.measurement,
            )
        )
        record = {"Qubits": str(widths[node])}
        if cost.t:
            record["Ts"] = str(cost.t)
        if aggregate:
            budget = t_and_ccz_budget(cost)
            record["Ts"] = str(budget["n_t"])
            record["CCZs"] = str(budget["n_ccz"])
        annotations[node] = record
    return graph, export_call_graph_dot(
        graph.root, graph.direct_calls, title=reference_title, annotations=annotations
    )


def source_html_table(label: str) -> str:
    """Adapt outer presentation while preserving every native computed cell."""
    table = ElementTree.fromstring(label[1:-1])
    rows = [list(row) for row in table]
    assert table.tag == "TABLE" and len(rows[0]) == 1
    title = html.escape("".join(rows[0][0].itertext()))
    result = (
        '<<font point-size="10"><table border="0" cellborder="1" cellspacing="0" cellpadding="5">\n'
        f'<tr><td colspan="2"><font point-size="10">{title}</font></td></tr>\n'
    )
    for row in rows[1:]:
        assert len(row) == 2
        result += (
            "<tr>"
            + "".join(
                f"<td>{html.escape(''.join(cell.itertext()))}</td>" for cell in row
            )
            + "</tr>"
        )
    return result + "</table></font>>"


class NativeView:
    """Test-only inspection shape for unchanged source consumers."""

    def __init__(
        self,
        graph: object,
        bloq_data: Mapping[object, Mapping[str, object]] | None = None,
    ) -> None:
        root, declarations = declared_example()
        self.native = symbolic_call_graph(root, declarations)
        annotations = None
        if bloq_data is not None:
            by_name = {str(node): values for node, values in bloq_data.items()}
            nodes = {
                root,
                *self.native.leaf_counts,
                *(callee for _, callee in self.native.direct_calls),
            }
            annotations = {
                node: {
                    key: str(value)
                    for key, value in by_name[reference_title(node)].items()
                }
                for node in nodes
            }
        self.dot = export_call_graph_dot(
            root,
            self.native.direct_calls,
            title=reference_title,
            annotations=annotations,
        )

    @classmethod
    def from_bloq(
        cls,
        bloq: object,
        *,
        max_depth: int | None = None,
        agg_gate_counts: str | None = None,
    ) -> "NativeView":
        assert agg_gate_counts in (None, "t_and_ccz")
        result = cls(None)
        result.native, result.dot = native_annotated_dot(
            max_depth, aggregate=agg_gate_counts is not None
        )
        return result

    @property
    def g(self) -> Any:
        import networkx as nx

        graph = nx.DiGraph()
        graph.add_node(self.native.root)
        graph.add_edges_from(self.native.direct_calls)
        return graph

    def get_graph(self) -> Any:
        import pydot

        graphs = pydot.graph_from_dot_data(self.dot)
        assert graphs is not None
        [graph] = graphs
        for node in graph.get_nodes():
            node.set("label", source_html_table(node.get("label")))
        return graph


@pytest.mark.parametrize("name", FUNCTIONS)
def test_original_assertions_source_only(
    source_tests: dict[str, Any], name: str
) -> None:
    source_tests[name]()


@pytest.mark.parametrize("name", FUNCTIONS)
def test_original_assertions_native_replacement(
    source_tests: dict[str, Any], name: str
) -> None:
    original = source_tests[name]
    namespace = {
        **original.__globals__,
        "GraphvizCallGraph": NativeView,
        "MultiAnd": lambda cvs: ConjunctionCall(cvs),
        "get_bloq_call_graph": native_conjunction_graph,
        "format_counts_sigma": native_sigma,
        "format_counts_graph_markdown": native_markdown,
    }
    replacement = FunctionType(original.__code__, namespace)
    assert replacement.__code__ is original.__code__
    replacement()


@pytest.mark.parametrize("depth,generations", [(0, 1), (1, 2), (2, 3), (None, 3)])
@pytest.mark.parametrize("aggregate", [False, True])
def test_full_native_annotations_and_visibility(
    depth: int | None, generations: int, aggregate: bool
) -> None:
    import networkx as nx
    import pydot
    from qualtran.bloqs.for_testing import TestBloqWithCallGraph
    from qualtran.drawing import GraphvizCallGraph

    graph, dot = native_annotated_dot(depth, aggregate=aggregate)
    source = GraphvizCallGraph.from_bloq(
        TestBloqWithCallGraph(),
        max_depth=depth,
        agg_gate_counts="t_and_ccz" if aggregate else None,
    )
    parsed = pydot.graph_from_dot_data(dot)
    assert parsed is not None
    [native] = parsed
    assert (
        len(native.get_nodes())
        == len(source.g)
        == (1 if depth == 0 else 4 if depth == 1 else 9)
    )
    assert (
        len(
            list(
                nx.topological_generations(
                    NativeView.from_bloq(None, max_depth=depth).g
                )
            )
        )
        == generations
    )
    assert {source_html_table(n.get("label")) for n in native.get_nodes()} == {
        n.get("label") for n in source.get_graph().get_nodes()
    }
    assert len(graph.direct_calls) == len(source.g.edges)


def test_original_assertions_detect_wrong_native_cost(
    source_tests: dict[str, Any],
) -> None:
    class WrongCost(NativeView):
        @classmethod
        def from_bloq(
            cls,
            bloq: object,
            *,
            max_depth: int | None = None,
            agg_gate_counts: str | None = None,
        ) -> "NativeView":
            result = super().from_bloq(
                bloq, max_depth=max_depth, agg_gate_counts=agg_gate_counts
            )
            result.dot = result.dot.replace("100*_n0 + 600", "100*_n0 + 601")
            return result

    original = source_tests["test_graphviz_call_graph_from_bloq"]
    replacement = FunctionType(
        original.__code__, {**original.__globals__, "GraphvizCallGraph": WrongCost}
    )
    with pytest.raises(AssertionError):
        replacement()
