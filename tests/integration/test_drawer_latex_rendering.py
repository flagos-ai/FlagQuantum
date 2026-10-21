"""Opt-in real compiler and Notebook formatter conformance."""

from __future__ import annotations

import shutil
import struct
from pathlib import Path

import pytest

import flagquantum as fq
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
from flagquantum.drawer import ClassicalLayer, LatexFigure, draw_hybrid, draw_latex

pytestmark = pytest.mark.integration


def test_real_pdf_fragment_bundle_and_notebook_display(tmp_path: Path) -> None:
    if not shutil.which("tectonic") or not shutil.which("pdftoppm"):
        pytest.skip("Real rendering requires Tectonic and Poppler")
    ipython = pytest.importorskip("IPython.core.formatters")
    parameter = fq.Parameter("theta_0")
    circuit = fq.Circuit(2).h(0).cx(0, 1).ry(1, parameter)
    fragment = draw_latex(circuit, standalone=False)
    assert fragment.render().startswith(b"%PDF-")
    assert fragment.render("png").startswith(b"\x89PNG")
    source = fragment.export_bundle(tmp_path / "bundle")
    assert source.is_file()
    figure = draw_hybrid(circuit, encoder_features=2, head_outputs=2, training=True)
    bundle, _ = ipython.DisplayFormatter().format(figure)
    assert any(mime in bundle for mime in ("image/png", "image/svg+xml"))
    assert figure.save(tmp_path / "hybrid.pdf").stat().st_size > 1000


def test_real_registered_unitary_gate_gallery() -> None:
    if not shutil.which("tectonic"):
        pytest.skip("Real gate-gallery compilation requires Tectonic")
    instructions = tuple(
        Instruction(
            schema.opcode,
            tuple(range(schema.arity)),
            dict.fromkeys(schema.parameters, 0.25),
        )
        for schema in OPERATOR_SCHEMAS.values()
        if schema.unitary
    )
    figure = draw_latex(CircuitIR(3, instructions))
    assert figure.render().startswith(b"%PDF-")


def test_real_detailed_layers_stack_circuit_panels_below_architecture() -> None:
    if not shutil.which("tectonic") or not shutil.which("pdftoppm"):
        pytest.skip("Real rendering requires Tectonic and Poppler")
    circuit = fq.Circuit(3)
    for index in range(24):
        circuit.ry(index % 3, fq.Parameter(f"theta_{index}"))
    layers = (
        ClassicalLayer("Conv2d", (1, 8, 8), (3, 8, 8), True),
        ClassicalLayer("Pool", (3, 8, 8), (3, 1, 1)),
        ClassicalLayer("Flatten", (3, 1, 1), (3,)),
    )
    detailed = draw_hybrid(circuit, encoder=layers, gates_per_panel=6, standalone=False)
    overview = draw_hybrid(circuit, encoder=layers, quantum_view="overview")
    width, height = struct.unpack(">II", detailed.render("png")[16:24])
    overview_width, overview_height = struct.unpack(
        ">II", overview.render("png")[16:24]
    )
    # The 181.5 mm canvas plus 2 pt margins fits a 183 mm double-column figure.
    assert width <= 1153  # 183 mm at the renderer's 160 dpi.
    # Panels add height, not side-by-side columns or scaled-down circuit text.
    assert width <= overview_width + 32
    assert overview_height < height <= 600  # Compact four-panel layout, under 95 mm.
    assert r"\resizebox" not in detailed.source
    assert detailed.render("pdf").startswith(b"%PDF-")


def test_oversized_circuit_panel_fails_instead_of_shrinking_text() -> None:
    if not shutil.which("tectonic"):
        pytest.skip("Width guard requires Tectonic")
    circuit = fq.Circuit(1).ry(0, fq.Parameter("theta_" + "a" * 160))
    figure = draw_hybrid(circuit)
    with pytest.raises(RuntimeError, match="Figure panel exceeds 181.5 mm"):
        figure.render()
    # Overview remains usable without omitting validation of circuit semantics.
    assert draw_hybrid(circuit, quantum_view="overview").render().startswith(b"%PDF-")


@pytest.mark.parametrize("training", [False, True])
def test_short_cnn_uses_one_compact_forward_strip(training: bool) -> None:
    if not shutil.which("tectonic") or not shutil.which("pdftoppm"):
        pytest.skip("Real rendering requires Tectonic and Poppler")
    circuit = fq.Circuit(3)
    for i in range(3):
        circuit.ry(i, fq.Parameter(f"x_{i}"))
    circuit.cx(0, 1).cx(1, 2)
    for i in range(3):
        circuit.ry(i, fq.Parameter(f"theta_{i}"))
    layers = (
        ClassicalLayer("Conv 3x3", (1, 8, 8), (3, 8, 8), True),
        ClassicalLayer("Tanh", (3, 8, 8), (3, 8, 8)),
        ClassicalLayer("Avg. pool", (3, 8, 8), (3, 1, 1)),
        ClassicalLayer("Flatten", (3, 1, 1), (3,)),
    )
    figure = draw_hybrid(
        circuit,
        encoder=layers,
        head=[ClassicalLayer("Linear", (3,), (2,), True)],
        readout="Z expectations",
        readout_features=3,
        training=training,
    )
    width, height = struct.unpack(">II", figure.render("png")[16:24])
    assert width <= 1153
    assert height <= (355 if training else 230)  # 56/36 mm at 160 dpi.
    assert figure.source.count(r"\begin{yquant}") == 1


def test_tall_detail_fails_without_reducing_register_font() -> None:
    if not shutil.which("tectonic"):
        pytest.skip("Height guard requires Tectonic")
    with pytest.raises(RuntimeError, match="Figure exceeds 170 mm height"):
        draw_hybrid(fq.Circuit(60), quantum_view="detail").render()
    assert (
        draw_hybrid(fq.Circuit(60), quantum_view="overview")
        .render()
        .startswith(b"%PDF-")
    )


def test_wide_circuit_panels_stack_without_global_scaling() -> None:
    if not shutil.which("tectonic") or not shutil.which("pdftoppm"):
        pytest.skip("Real rendering requires Tectonic and Poppler")
    short, wide = fq.Circuit(1), fq.Circuit(1)
    for i in range(12):
        short.ry(0, fq.Parameter(f"t_{i}"))
        wide.ry(0, fq.Parameter(f"weight_block_{i}"))
    compact = draw_hybrid(short, gates_per_panel=6)
    stacked = draw_hybrid(wide, gates_per_panel=6)
    compact_width, compact_height = struct.unpack(">II", compact.render("png")[16:24])
    wide_width, wide_height = struct.unpack(">II", stacked.render("png")[16:24])
    assert compact_width == wide_width <= 1153
    assert wide_height > compact_height + 25


def test_tensor_matrix_and_large_vectors_keep_declared_counts() -> None:
    if not shutil.which("tectonic"):
        pytest.skip("Tensor glyph rendering requires Tectonic")
    figure = draw_hybrid(
        fq.Circuit(1).h(0),
        encoder=[ClassicalLayer("Reshape", (2, 3), (6,))],
        head=[ClassicalLayer("Linear", (6,), (5,))],
        readout="Z features",
        readout_features=6,
        encoder_trainable=False,
        quantum_trainable=False,
        head_trainable=False,
        training=True,
    )
    assert "6 values" in figure.source and r"\vdots" in figure.source
    assert figure.render().startswith(b"%PDF-")


def test_large_image_shapes_wrap_without_reducing_type_size() -> None:
    if not shutil.which("tectonic") or not shutil.which("pdftoppm"):
        pytest.skip("Real rendering requires Tectonic and Poppler")
    figure = draw_hybrid(
        fq.Circuit(3).ry(0, fq.Parameter("theta_0")),
        encoder=[
            ClassicalLayer("Conv+Tanh", (3, 224, 224), (64, 224, 224), True),
            ClassicalLayer("Global pool", (64, 224, 224), (64, 1, 1)),
            ClassicalLayer("Flatten", (64, 1, 1), (64,)),
            ClassicalLayer("Linear", (64,), (3,), True),
        ],
        head=[ClassicalLayer("Linear", (3,), (2,), True)],
        readout="Z expectations",
        readout_features=3,
    )
    width, height = struct.unpack(">II", figure.render("png")[16:24])
    assert width <= 1153 and height <= 300


def test_figure_package_preserves_the_papers_sans_serif_default() -> None:
    if not shutil.which("tectonic"):
        pytest.skip("Preamble isolation requires Tectonic")
    source = r"""\documentclass{standalone}
\let\beforeFigureFamily\sfdefault
\usepackage{flaghybrid}
\ifx\beforeFigureFamily\sfdefault\else
\PackageError{figure-test}{The diagram package changed the paper font}{Keep font selection local.}
\fi
\begin{document}\sffamily Preserved document font\end{document}
"""
    assert LatexFigure(source).render().startswith(b"%PDF-")
