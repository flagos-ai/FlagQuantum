"""Render verified scores and compare pixels with the locked Matplotlib path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.synthesis_local_registers import local_composition
from flagquantum.drawer.register_score import register_program_score
from flagquantum.drawer.register_score_data import register_score_data
from flagquantum.drawer.register_score_mpl import draw_register_score

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("max_width,max_height", [(4.0, 8.0), (100.0, 2.0)])
def test_register_score_matches_locked_agg(
    tmp_path: Path, max_width: float, max_height: float
) -> None:
    pytest.importorskip("qualtran")
    np = pytest.importorskip("numpy")
    pyplot = pytest.importorskip("matplotlib.pyplot")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from qualtran import BloqBuilder, CUInt, QBit, Register, Signature
    from qualtran.bloqs.basic_gates import CNOT, Hadamard
    from qualtran.drawing import dump_musical_score, get_musical_score_data
    from qualtran.drawing.musical_score import draw_musical_score

    signature = Signature([Register("ctrl", QBit()), Register("bits", CUInt(3))])
    builder, inputs = BloqBuilder.from_signature(signature)
    ctrl = builder.add(Hadamard(), q=inputs["ctrl"])
    work = builder.allocate(dtype=QBit())
    for _ in range(2):
        ctrl, work = builder.add(CNOT(), ctrl=ctrl, target=work)
    builder.free(work)
    composite = builder.finalize(ctrl=ctrl, bits=inputs["bits"])
    reference = get_musical_score_data(composite)
    path = tmp_path / "locked"
    dump_musical_score(reference, name=str(path))
    records = json.loads(path.with_suffix(".json").read_text())
    options = {"max_width": max_width, "max_height": max_height}
    with pyplot.rc_context({"figure.dpi": 100, "font.family": "DejaVu Sans"}):
        expected, expected_ax = draw_musical_score(reference, **options)
        actual, actual_ax = draw_register_score(records, **options)
        try:
            expected_canvas = FigureCanvasAgg(expected)
            actual_canvas = FigureCanvasAgg(actual)
            expected_canvas.draw()
            actual_canvas.draw()
            assert actual_ax.get_xlim() == expected_ax.get_xlim()
            assert actual_ax.get_ylim() == expected_ax.get_ylim()
            assert np.array_equal(
                actual_canvas.buffer_rgba(), expected_canvas.buffer_rgba()
            )
            assert np.array_equal(actual.get_size_inches(), expected.get_size_inches())
        finally:
            pyplot.close(actual)
            pyplot.close(expected)


def test_native_register_score_exports_png(tmp_path: Path) -> None:
    pyplot = pytest.importorskip("matplotlib.pyplot")
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    program, functions = local_composition(overlapping=False, helpers=True)
    score = register_program_score(program, functions)
    data = register_score_data(score)
    fig, ax = draw_register_score(data)
    try:
        path = tmp_path / "score.png"
        FigureCanvasAgg(fig).print_png(path)
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        labels = {text.get_text() for text in ax.texts}
        assert {"alloc", "free", "uncompute"} <= labels
        assert len(ax.collections) == len(data["hlines"]) + len(data["vlines"])
    finally:
        pyplot.close(fig)
