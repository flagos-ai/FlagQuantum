# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Assertions executed from the unchanged Qualtran v0.7.0 Graphviz tests.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/drawing/graphviz_test.py

"""Replace graph test construction at the test boundary, retaining assertions."""

import ast
import hashlib
import runpy
from collections.abc import Callable
from pathlib import Path
from types import FunctionType
from typing import Any
from xml.etree import ElementTree

import pytest

from examples.algorithms.register_graphs import composite_dot
from flagquantum.drawer.call_graph import render_call_graph_svg

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / "ftqc-reference-env/qualtran/drawing/graphviz_test.py"


@pytest.fixture
def source_tests(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Require the frozen complete file before loading its unmodified functions."""
    raw = SOURCE.read_bytes()
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == (
        "28e17f850887d55af6472cd6574d006285397401"
    )
    functions = [
        node.name for node in ast.parse(raw).body if isinstance(node, ast.FunctionDef)
    ]
    assert functions == ["test_assign_ids", "test_graphviz", "test_notebook"]
    request.config.addinivalue_line("markers", "notebook: Locked source notebook test")
    return runpy.run_path(str(SOURCE))


@pytest.mark.parametrize("pretty", [False, True])
def test_original_graph_assertions_with_native_construction(
    source_tests: dict[str, Any], pretty: bool
) -> None:
    import IPython.display
    import pydot

    class NativeExample:
        def decompose_bloq(self) -> str:
            return composite_dot(pretty=pretty)

    class NativeView:
        """Test-only view supplies the original consumer's inspection methods."""

        def __init__(self, dot: str) -> None:
            self.dot = dot

        def get_graph(self) -> Any:
            graphs = pydot.graph_from_dot_data(self.dot)
            assert graphs is not None
            [graph] = graphs
            return graph

        def get_svg_bytes(self) -> bytes:
            return render_call_graph_svg(self.dot)

        def get_svg(self) -> IPython.display.SVG:
            factory: Callable[[bytes], IPython.display.SVG] = IPython.display.SVG
            return factory(self.get_svg_bytes())

    original = source_tests["test_graphviz"]
    namespace = {**original.__globals__, "TestParallelCombo": NativeExample}
    replacement = FunctionType(original.__code__, namespace)
    assert replacement.__code__ is original.__code__
    replacement(NativeView)


def test_original_id_assertions_and_native_endpoint_integrity(
    source_tests: dict[str, Any],
) -> None:
    import pydot

    source_tests["test_assign_ids"]()
    graphs = pydot.graph_from_dot_data(composite_dot())
    assert graphs is not None
    [graph] = graphs
    nodes = list(graph.get_nodes())
    boundaries = [node for group in graph.get_subgraphs() for node in group.get_nodes()]
    names = [node.get_name() for node in nodes + boundaries]
    assert len(names) == len(set(names)) == 7
    declared: set[tuple[str, str]] = set()
    for node in nodes:
        label = node.get_attributes()["label"]
        assert isinstance(label, str)
        table = ElementTree.fromstring(label[1:-1])
        local_ports = [
            cell.attrib["PORT"] for cell in table.iter() if "PORT" in cell.attrib
        ]
        assert len(local_ports) == len(set(local_ports))
        declared.update((node.get_name(), port) for port in local_ports)
    declared.update((node.get_name(), "") for node in boundaries)
    # Thirteen live endpoints correspond to thirteen source Soquets. The five
    # source group IDs are unused drawing reservations and have no native IDs.
    assert len(declared) == 13
    for edge in graph.get_edges():
        for endpoint in (edge.get_source(), edge.get_destination()):
            assert isinstance(endpoint, str)
            parts = endpoint.split(":")
            key = (parts[0], parts[1] if len(parts) == 3 else "")
            assert key in declared
    assert len(graph.get_edges()) == 8
