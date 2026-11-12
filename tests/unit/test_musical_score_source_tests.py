# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Assertions executed from the unchanged Qualtran v0.7.0 musical-score tests.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/drawing/musical_score_test.py

"""Keep original dump assertions while replacing construction with native scores."""

import ast
import hashlib
import json
import runpy
from pathlib import Path
from types import FunctionType
from typing import Any

import numpy as np
import pytest

from flagquantum.drawer.register_score import (
    RegisterLine,
    RegisterScore,
    layout_register_score,
    multi_and_register_score,
)
from flagquantum.drawer.register_score_data import (
    MusicalScoreData,
    ScoreHorizontalLine,
    export_register_score_json,
    register_score_data,
)
from flagquantum.drawer.register_score_mpl import draw_register_score

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / "ftqc-reference-env/qualtran/drawing/musical_score_test.py"


@pytest.fixture(scope="module", autouse=True)
def source_tests(request: pytest.FixtureRequest) -> dict[str, Any]:
    blobs = {
        "musical_score.py": "3863bb66cdba17c4654657e275a8485d94b8da75",
        "musical_score_test.py": "db35be86a8fc4c1f3dc25b4154a89776ec519565",
        "musical_score.ipynb": "7205fa53b168a8b0de5429345dc25d2081979446",
    }
    for name, blob in blobs.items():
        data = (SOURCE.parent / name).read_bytes()
        assert hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == blob
    request.config.addinivalue_line("markers", "notebook: Locked source notebook test")
    raw = SOURCE.read_bytes()
    assert [
        node.name for node in ast.parse(raw).body if isinstance(node, ast.FunctionDef)
    ] == ["test_dump_json", "test_notebook"]
    return runpy.run_path(str(SOURCE))


def test_original_dump_assertions_with_native_construction(
    source_tests: dict[str, Any], tmp_path: Path
) -> None:
    class NativeLine:
        """Test-only view of the original record inspection protocol."""

        def __init__(self, y: int, seq_x_start: int, seq_x_end: int) -> None:
            line = RegisterLine("wire", 1, y, seq_x_start, seq_x_end)
            self.data = register_score_data(RegisterScore((), (line,), 0))["hlines"][0]

        def json_dict(self) -> ScoreHorizontalLine:
            return self.data

    class NativeMultiAnd:
        def __init__(self, values: tuple[int, ...]) -> None:
            self.values = values

        def decompose_bloq(self) -> RegisterScore:
            return multi_and_register_score(self.values)

    def dump(data: MusicalScoreData, name: str) -> None:
        Path(f"{name}.json").write_text(
            export_register_score_json(data), encoding="utf-8"
        )

    original = source_tests["test_dump_json"]
    namespace = {
        **original.__globals__,
        "HLine": NativeLine,
        "MultiAnd": NativeMultiAnd,
        "get_musical_score_data": register_score_data,
        "dump_musical_score": dump,
    }
    replacement = FunctionType(original.__code__, namespace)
    assert replacement.__code__ is original.__code__
    source_dir, native_dir = tmp_path / "source", tmp_path / "native"
    source_dir.mkdir()
    native_dir.mkdir()
    original(source_dir)
    replacement(native_dir)
    expected = json.loads((source_dir / "musical_score_example.json").read_text())
    actual = json.loads((native_dir / "musical_score_example.json").read_text())
    for key in ("max_x", "max_y", "hlines", "vlines"):
        assert actual[key] == expected[key]
    assert sorted(
        json.dumps({k: v for k, v in p.items() if k != "ident"}, sort_keys=True)
        for p in actual["soqs"]
    ) == sorted(
        json.dumps({k: v for k, v in p.items() if k != "ident"}, sort_keys=True)
        for p in expected["soqs"]
    )


@pytest.mark.parametrize(
    "symbol,attributes",
    [
        ("TextBox", {"text": "G"}),
        ("Text", {"text": "G†", "fontsize": 13}),
        ("RarrowTextBox", {"text": "work"}),
        ("LarrowTextBox", {"text": "work"}),
        ("Circle", {"filled": True}),
        ("Circle", {"filled": False}),
        ("ModPlus", {}),
    ],
)
def test_all_source_symbols_render_exactly_through_native_records(
    symbol: str, attributes: dict[str, str | int | bool], tmp_path: Path
) -> None:
    import qualtran.drawing.musical_score as source

    pyplot = pytest.importorskip("matplotlib.pyplot")
    agg = pytest.importorskip("matplotlib.backends.backend_agg")
    glyph = getattr(source, symbol)(**attributes)
    expected_data = source.MusicalScoreData(
        max_x=1,
        max_y=0,
        soqs=[source.SoqData(glyph, source.RegPosition(0, 0, 1), "glyph")],
        hlines=[source.HLine(0, -1, 1)],
        vlines=[source.VLine(0, 0, 0, source.Text("connector", fontsize=11))],
    )
    # The existing JSON owner is the only interchange representation. Render
    # source records directly without rebuilding or approximating their symbols.
    data: MusicalScoreData = json.loads(
        json.dumps(expected_data, cls=source.MusicalScoreEncoder)
    )
    with pyplot.rc_context({"figure.dpi": 100, "font.family": "DejaVu Sans"}):
        expected, _ = source.draw_musical_score(expected_data)
        actual, _ = draw_register_score(data)
        try:
            original_canvas, native_canvas = agg.FigureCanvasAgg(
                expected
            ), agg.FigureCanvasAgg(actual)
            original_canvas.draw()
            native_canvas.draw()
            assert np.array_equal(
                original_canvas.buffer_rgba(), native_canvas.buffer_rgba()
            )
            assert np.array_equal(expected.get_size_inches(), actual.get_size_inches())
            native_canvas.print_png(tmp_path / f"{symbol}.png")
        finally:
            pyplot.close(actual)
            pyplot.close(expected)


def test_empty_graph_boundary_is_explicit() -> None:
    original = pytest.importorskip("qualtran")
    source = pytest.importorskip("qualtran.drawing.musical_score")
    with pytest.raises(ValueError, match=r"max\(\).*empty"):
        source.get_musical_score_data(original.BloqBuilder().finalize())
    assert register_score_data(layout_register_score({}, (), ())) == {
        "max_x": 0,
        "max_y": 0,
        "soqs": [],
        "hlines": [],
        "vlines": [],
    }


def test_native_layout_has_no_source_default_hundred_line_limit() -> None:
    original = pytest.importorskip("qualtran")
    source = pytest.importorskip("qualtran.drawing.musical_score")
    signature = original.Signature.build(
        **dict.fromkeys((f"q{i}" for i in range(101)), 1)
    )
    builder, inputs = original.BloqBuilder.from_signature(signature)
    graph = builder.finalize(**inputs)
    with pytest.raises(IndexError, match="index out of range"):
        source.get_musical_score_data(graph)
    names = tuple(register.name for register in signature)
    actual = register_score_data(
        layout_register_score(dict.fromkeys(names, 1), names, ())
    )
    expected = source.get_musical_score_data(graph, manager=source.LineManager(101))
    assert actual["max_y"] == expected.max_y == 100
    assert actual["hlines"] == [line.json_dict() for line in expected.hlines]
    assert len(actual["soqs"]) == len(expected.soqs) == 202
