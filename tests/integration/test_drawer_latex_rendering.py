"""Opt-in real compiler and Notebook formatter conformance."""

from __future__ import annotations

import shutil
import struct
from pathlib import Path

import pytest

import flagquantum as fq
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
from flagquantum.drawer import ClassicalLayer, draw_hybrid, draw_latex

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
    for index in range(18):
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
    # Panels add height, not side-by-side columns or scaled-down circuit text.
    assert width <= overview_width + 32
    assert height > overview_height
    assert r"\resizebox" not in detailed.source
    assert detailed.render("pdf").startswith(b"%PDF-")
