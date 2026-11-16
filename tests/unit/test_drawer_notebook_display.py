"""Exercise every notebook display path with the locked reference renderers."""

import hashlib
import shutil
from importlib import import_module
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

import pytest

from examples.algorithms.register_graphs import parallel_program
from flagquantum import Circuit
from flagquantum.compiler._component_trace.call_graph import SymbolicCallGraph
from flagquantum.compiler._component_trace.resource_generalization import FixedTCall
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.layout.component_cast import cast_word_function
from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer import draw_latex
from flagquantum.drawer.call_graph import GateAggregation
from flagquantum.drawer.component_graph import export_component_graph_dot
from flagquantum.drawer.notebook_display import (
    show_call_graph,
    show_counts,
    show_flame_graph,
    show_latex_diagram,
    show_register_program,
    show_register_programs,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
REFERENCE = Path(__file__).resolve().parents[3] / "ftqc-reference-env"


@pytest.fixture(scope="module", autouse=True)
def _verify_frozen_source() -> None:
    for name, checksum in (
        ("__init__.py", "78692d24b174743b04a4e8c33180d628d14c3be3"),
        ("_show_funcs.py", "93b1a6b6cd1c6c2341789414156479d6a8db9ad4"),
    ):
        source = (REFERENCE / "qualtran" / "drawing" / name).read_bytes()
        assert (
            hashlib.sha1(f"blob {len(source)}\0".encode() + source).hexdigest()
            == checksum
        )


def _svg_text(source: str, xpath: str = ".//{*}text") -> list[str]:
    return [
        "".join(node.itertext())
        for node in ElementTree.fromstring(source).findall(xpath)
    ]


def test_graph_dtype_and_musical_score(monkeypatch: pytest.MonkeyPatch) -> None:
    display = import_module("IPython.display")
    reference = import_module("qualtran.drawing")
    source_examples = import_module("qualtran.bloqs.for_testing")
    pyplot = import_module("matplotlib.pyplot")
    displayed: list[object] = []
    monkeypatch.setattr(display, "display", displayed.append)
    figures_before = pyplot.get_fignums()
    program = parallel_program()
    with patch("matplotlib.figure.Figure.show") as show:
        try:
            show_register_program(program, "musical_score")
            show.assert_called_once()
        finally:
            for number in set(pyplot.get_fignums()) - set(figures_before):
                pyplot.close(number)
    for mode in ("graph", "DTYPE"):
        reference.show_bloq(source_examples.TestParallelCombo().decompose_bloq(), mode)
        show_register_program(program, mode)
        expected = getattr(displayed[-2], "data")
        actual = getattr(displayed[-1], "data")
        assert (
            len(ElementTree.fromstring(expected).findall(".//{*}g[@class='edge']")) == 8
        )
        assert (
            len(ElementTree.fromstring(actual).findall(".//{*}g[@class='edge']")) == 8
        )
        assert "atom" in _svg_text(actual)
        assert any(
            ("QBit()" in label if mode == "DTYPE" else label == "1")
            for label in _svg_text(expected)
        )
        assert any(
            ("encoding='bit'" in label if mode == "DTYPE" else label == "1")
            for label in _svg_text(actual)
        )


def test_typed_boundary_differs_by_endpoint() -> None:
    opaque = opaque_register_type(3)
    unsigned = unsigned_register_type(3)
    root = Signature((Register("q", opaque),))
    to_unsigned = cast_word_function("to_unsigned", opaque, unsigned)
    to_opaque = cast_word_function("to_opaque", unsigned, opaque)
    builder = ComponentProgramBuilder(root, (to_unsigned, to_opaque))
    intermediate = builder.call_dict("to_unsigned", {"reg": builder.inputs["q"]})
    returned = builder.call_dict("to_opaque", {"reg": intermediate["reg"]})
    dot = export_component_graph_dot(
        builder.finalize({"q": returned["reg"]}), typed=True
    )
    assert dot.count('fontcolor="black"') == 3
    assert dot.count("encoding='opaque'") == 2
    assert dot.count("encoding='uint'") == 1


def test_grouped_programs_and_negative_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    display = import_module("IPython.display")
    source = import_module("qualtran.drawing")
    gates = import_module("qualtran.bloqs.basic_gates")
    displayed: list[object] = []
    monkeypatch.setattr(display, "display", displayed.append)
    program = parallel_program()
    labels = (None, "**Example**")
    source.show_bloqs([gates.Toffoli(), gates.Toffoli()], labels)
    show_register_programs((program, program), labels)
    for box in displayed:
        first, second = getattr(box, "children")
        assert [len(first.outputs), len(second.outputs)] == [1, 2]
        assert second.outputs[0]["data"]["text/markdown"] == "**Example**"
        assert "<svg" in second.outputs[1]["data"]["image/svg+xml"]
    with pytest.raises(ValueError, match="exactly one"):
        show_register_programs((program,), labels)
    with pytest.raises(ValueError, match="Display mode"):
        show_register_program(program, "pdf")
    with pytest.raises(TypeError, match="LatexFigure"):
        show_register_program(program, "latex")


@pytest.mark.parametrize(
    "aggregation", ["factored", "total_t", "t_and_ccz", "beverland"]
)
def test_call_graph_profiles(
    monkeypatch: pytest.MonkeyPatch, aggregation: GateAggregation
) -> None:
    display = import_module("IPython.display")
    source = import_module("qualtran.drawing")
    gates = import_module("qualtran.bloqs.basic_gates")
    outputs: list[object] = []
    monkeypatch.setattr(display, "display", outputs.append)
    source.show_call_graph(gates.Toffoli(), max_depth=0, agg_gate_counts=aggregation)
    show_call_graph(FixedTCall("toffoli"), max_depth=0, aggregation=aggregation)
    old, new = (_svg_text(getattr(output, "data")) for output in outputs)
    assert old[1:] == new[1:]
    assert "Qubits" in new
    assert "3" in new


def test_prebuilt_graph_and_symbolic_markdown(monkeypatch: pytest.MonkeyPatch) -> None:
    display = import_module("IPython.display")
    sympy = import_module("sympy")
    source = import_module("qualtran.drawing")
    gates = import_module("qualtran.bloqs.basic_gates")
    outputs: list[object] = []
    monkeypatch.setattr(display, "display", outputs.append)
    exponent = sympy.Symbol("n")
    graph = SymbolicCallGraph("root", {("root", "leaf"): exponent}, {"leaf": exponent})
    show_call_graph(graph, max_depth=0, aggregation="total_t")
    assert "leaf" in _svg_text(getattr(outputs[0], "data"))
    source.show_counts_sigma({gates.CNOT(): exponent, gates.Toffoli(): 2})
    show_counts({"CNOT": exponent, "Toffoli": 2})
    assert getattr(outputs[-2], "data") == getattr(outputs[-1], "data")


def test_real_flame_svg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    display = import_module("IPython.display")
    source = import_module("qualtran.drawing")
    gates = import_module("qualtran.bloqs.basic_gates")
    outputs: list[object] = []
    monkeypatch.setattr(display, "display", outputs.append)
    reference_svg = tmp_path / "source.svg"
    native_svg = tmp_path / "native.svg"
    source.show_flame_graph(gates.Toffoli(), file_path=reference_svg, keep=None)
    show_flame_graph(
        FixedTCall("toffoli"),
        executable=str(REFERENCE / "qualtran/third_party/flamegraph/flamegraph.pl"),
        keep=None,
        file_path=native_svg,
    )
    for output, path in zip(outputs, (reference_svg, native_svg), strict=True):
        assert (
            "4 TCounts" in getattr(output, "data") and "4 TCounts" in path.read_text()
        )


def test_real_latex_png(monkeypatch: pytest.MonkeyPatch) -> None:
    if not any(
        shutil.which(tool) for tool in ("tectonic", "pdflatex", "xelatex", "lualatex")
    ):
        pytest.skip("A local TeX engine is required for real PNG display")
    if not any(shutil.which(tool) for tool in ("pdftoppm", "pdftocairo")):
        pytest.skip("Poppler is required for real PNG display")
    display = import_module("IPython.display")
    outputs: list[object] = []
    monkeypatch.setattr(display, "display", outputs.append)
    figure = draw_latex(Circuit(2).h(0).cx(0, 1), timeout=20)
    show_register_program(figure, "LaTeX")
    show_latex_diagram(figure, width=500, height=200)
    for output, size in zip(outputs[-2:], [(1000, 400), (500, 200)], strict=True):
        assert getattr(output, "data").startswith(b"\x89PNG\r\n\x1a\n")
        assert (getattr(output, "width"), getattr(output, "height")) == size
