"""Test the original qpic fixture and native logical wire and rendering contracts."""

import hashlib
import os
import shutil
from importlib import import_module
from pathlib import Path

import pytest

from flagquantum.core.ir.program.register_types import (
    bounded_unsigned_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.circuit_diagram import (
    export_register_diagram_qpic,
    qpic_latex_figure,
)
from flagquantum.drawer.register_score import RegisterStep, layout_register_score
from flagquantum.drawer.register_score_data import (
    PortSide,
    SymbolClass,
    adjoint_wire_symbol,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
REFERENCE = Path(__file__).resolve().parents[3] / "ftqc-reference-env"


def _reflection_diagram() -> str:
    selection = Register("selection", bounded_unsigned_register_type(2, 4))
    step = RegisterStep(
        "ReflectionUsingPrepare",
        ("selection",),
        ("selection",),
        input_registers=(selection,),
        output_registers=(selection,),
    )
    score = layout_register_score(
        {"selection": 2},
        ("selection",),
        (step,),
        input_registers=(selection,),
        output_registers=(selection,),
    )

    def reflection_symbol(
        step: RegisterStep, wire: str | None, side: PortSide | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        return "TextBox", {"text": "R_L"}

    return export_register_diagram_qpic(
        score, Signature((selection,)), wire_symbol=reflection_symbol
    )


def test_original_reflection_lines_and_frozen_source() -> None:
    for name, digest in (
        ("qpic_diagram.py", "a11a9eecc1bde23f43918ae177b728a01b667dc1"),
        ("qpic_diagram_test.py", "c5aeb0ed27cc84ef72e64a99a6da399d3cf7807b"),
    ):
        data = (REFERENCE / "qualtran/drawing" / name).read_bytes()
        assert hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == digest
    source = import_module("qualtran.drawing.qpic_diagram")
    alias = import_module(
        "qualtran.bloqs.state_preparation.state_preparation_alias_sampling"
    )
    reflection = import_module("qualtran.bloqs.reflections.reflection_using_prepare")
    prepare = alias.StatePreparationAliasSampling.from_probabilities(
        [0.1, 0.2, 0.3, 0.4], precision=0.1
    )
    bloq = reflection.ReflectionUsingPrepare(prepare, global_phase=-1j)
    assert _reflection_diagram().splitlines() == source.get_qpic_data(bloq)


def test_controls_lifetimes_adjoint_and_wireless_box() -> None:
    bit = quantum_bit_type()
    first, second = Register("control", bit), Register("target", bit)
    steps = (
        RegisterStep("cx", ("control", "target"), ("control", "target"), gate="cx"),
        *(
            step
            for identity in ("first", "second")
            for step in (
                RegisterStep(
                    "alloc",
                    ("target",),
                    ("target", identity),
                    input_registers=(second,),
                    output_registers=(
                        second,
                        Register("work", bit, side=RegisterSide.RIGHT),
                    ),
                ),
                RegisterStep(
                    "free",
                    ("target", identity),
                    ("target",),
                    input_registers=(
                        second,
                        Register("work", bit, side=RegisterSide.LEFT),
                    ),
                    output_registers=(second,),
                ),
            )
        ),
        RegisterStep("GlobalPhase"),
        RegisterStep("adjoint", ("target",), ("target",)),
    )
    score = layout_register_score(
        dict.fromkeys(("control", "target", "first", "second"), 1),
        ("control", "target"),
        steps,
        input_registers=(first, second),
    )

    def symbols(
        step: RegisterStep, wire: str | None, side: PortSide | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        if step.name == "adjoint":
            return adjoint_wire_symbol("TextBox", {"text": "A&B_2"})
        if step.gate == "cx":
            return (
                ("ModPlus", {}) if wire == "target" else ("Circle", {"filled": False})
            )
        if side == "THRU":
            return "TextBox", {"text": step.name}
        return ("LarrowTextBox" if side == "RIGHT" else "RarrowTextBox"), {"text": ""}

    lines = export_register_diagram_qpic(
        score, Signature((first, second)), wire_symbol=symbols
    ).splitlines()
    assert "-control +target" in lines
    assert "_empty_wire W off" in lines
    assert any("shape=8" in line and "GlobalPhase" in line for line in lines)
    assert any(r"A\&B\_2$^\dagger$" in line and "width=29" in line for line in lines)
    assert lines.count("work W off") == 1
    assert sum("work:on G:width=5:shape=<" in line for line in lines) == 2
    assert sum("work:off G:width=5:shape=>" in line for line in lines) == 2
    assert "work_1" not in "\n".join(lines)


def test_real_qpic_converter_and_native_figure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = REFERENCE / "qpic-tool"
    command = root / "bin/qpic"
    if not command.is_file():
        pytest.skip(
            "The isolated qpic reference tool is required for the real conversion"
        )
    old = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", f"{root}:{old}")
    monkeypatch.setenv("PATH", f"{root / 'bin'}:{os.environ['PATH']}")
    source = import_module("qualtran.drawing.qpic_diagram")
    source_path = tmp_path / "reflection.qpic"
    source_tex = tmp_path / "source.tex"
    source_path.write_text(_reflection_diagram())
    source.qpic_input_to_diagram(source_path, source_tex, output_type="tex")
    figure = qpic_latex_figure(_reflection_diagram(), executable=str(command))
    assert figure.source.strip() in source_tex.read_text()
    assert r"\begin{tikzpicture}" in figure.source
    assert r"R\_L" in figure.source and "BQUInt(2, 4)" in figure.source
    assert figure.save(tmp_path / "reflection.tex").read_text() == figure.source
    if not shutil.which("tectonic") or not shutil.which("pdftoppm"):
        pytest.skip("Tectonic and Poppler are required for the real PDF and PNG")
    assert figure.render("pdf").startswith(b"%PDF-")
    output = figure.save(tmp_path / "reflection.png")
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_missing_converter_has_specific_diagnostic() -> None:
    with pytest.raises(ImportError, match="Logical qpic conversion requires"):
        qpic_latex_figure("", executable="flagquantum-missing-qpic")
