"""Compare the three owned component graph presentations with the locked source."""

import hashlib
from importlib import import_module
from pathlib import Path
from xml.etree import ElementTree

import pytest

from examples.algorithms.register_graphs import parallel_program
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.call_graph import render_call_graph_svg
from flagquantum.drawer.component_graph import export_component_graph_dot
from tests.unit.test_graphviz_notebook import normalized_table

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
SOURCE = (
    Path(__file__).resolve().parents[3]
    / "ftqc-reference-env/qualtran/drawing/graphviz.py"
)


@pytest.mark.parametrize(
    "drawer_name,pretty,typed",
    [
        ("GraphDrawer", False, False),
        ("PrettyGraphDrawer", True, False),
        ("TypedGraphDrawer", True, True),
    ],
)
def test_three_styles_against_source(
    drawer_name: str, pretty: bool, typed: bool
) -> None:
    source = SOURCE.read_bytes()
    assert (
        hashlib.sha1(f"blob {len(source)}\0".encode() + source).hexdigest()
        == "b22e5252ec474f629be155f8da1f27750eb5896c"
    )
    examples = import_module("qualtran.bloqs.for_testing")
    drawing = import_module("qualtran.drawing.graphviz")
    pydot = import_module("pydot")
    example = examples.TestParallelCombo().decompose_bloq()
    drawer = getattr(drawing, drawer_name)(example)
    expected = drawer.get_graph()
    dot = export_component_graph_dot(parallel_program(), pretty=pretty, typed=typed)
    actual = pydot.graph_from_dot_data(dot)[0]
    for instance in drawer._binsts:
        source_table = expected.get_node(drawer.ids[instance])[0].get_label()
        native_table = actual.get_node(f"n{instance.i}")[0].get_label()
        assert normalized_table(native_table) == normalized_table(
            source_table.replace("TestAtom", "atom")
        )

    def normalize_dtype(label: str) -> str:
        if "encoding='opaque'" in label:
            return "QAny(3)"
        if "encoding='bit'" in label:
            return "QBit()"
        return label.strip('"')

    source_nodes = {
        drawer.ids[instance]: f"n{instance.i}" for instance in drawer._binsts
    }
    for dangle, namespace in (("LeftDangle", "in"), ("RightDangle", "out")):
        for soquet in drawer._soquets:
            if str(soquet.binst) == dangle:
                source_nodes[drawer.ids[soquet]] = namespace

    def native_node(endpoint: str) -> str:
        node, *_ = endpoint.split(":")
        return node.split("_", 1)[0]

    def edge_snapshot(graph: object, source_graph: bool) -> list[tuple[object, ...]]:
        result = []
        for edge in getattr(graph, "get_edges")():
            start, end = edge.get_source(), edge.get_destination()
            pair = (
                (source_nodes[start.split(":")[0]], source_nodes[end.split(":")[0]])
                if source_graph
                else (native_node(start), native_node(end))
            )
            attributes = edge.get_attributes()
            attributes["label"] = normalize_dtype(attributes["label"])
            result.append(
                (
                    *pair,
                    tuple(
                        sorted(
                            (k, str(v).strip('"').lower())
                            for k, v in attributes.items()
                        )
                    ),
                )
            )
        return sorted(result)

    assert edge_snapshot(actual, False) == edge_snapshot(expected, True)
    svg = ElementTree.fromstring(render_call_graph_svg(dot))
    assert len(svg.findall(".//{*}g[@class='edge']")) == 8


def test_boundary_uses_formal_output_name_and_multiplicity() -> None:
    bit = quantum_bit_type()
    signature = Signature((Register("published<&", bit, side=RegisterSide.RIGHT),))
    builder = ComponentProgramBuilder(signature)
    record = builder.finalize({"published<&": builder.allocate(dtype=bit)})
    dot = export_component_graph_dot(record, pretty=False)
    assert "label=<published&lt;&amp;>" in dot
    assert "call0:" not in dot
    svg = ElementTree.fromstring(render_call_graph_svg(dot))
    assert "published<&" in [element.text for element in svg.findall(".//{*}text")]
    with pytest.raises(ValueError, match="requires compact"):
        export_component_graph_dot(record, pretty=False, typed=True)


def test_directional_array_boundaries_follow_formal_spelling() -> None:
    register = Register("input", quantum_bit_type(), (1, 2))
    builder = ComponentProgramBuilder(Signature((register,)))
    record = builder.finalize({"input": builder.inputs["input"]})
    svg = ElementTree.fromstring(
        render_call_graph_svg(export_component_graph_dot(record))
    )
    labels = [element.text for element in svg.findall(".//{*}text")]
    assert labels.count("input[0, 0]") == labels.count("input[0, 1]") == 2
