"""Opt-in real compiler and Notebook formatter conformance."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import flagquantum as fq
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
from flagquantum.drawer import draw_hybrid, draw_latex

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
