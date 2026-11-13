"""Replace native port painting against frozen source custom symbol dispatch."""

from __future__ import annotations

import json
from hashlib import sha1
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from flagquantum.drawer.register_score_data import (
    MusicalScoreData,
    ScorePort,
    SymbolClass,
    export_register_score_json,
)
from flagquantum.drawer.register_score_mpl import (
    draw_register_score,
    draw_register_score_port,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    raw = (
        Path(__file__).resolve().parents[3]
        / "ftqc-reference-env/qualtran/drawing/musical_score.py"
    ).read_bytes()
    assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == (
        "3863bb66cdba17c4654657e275a8485d94b8da75"
    )


def source_badge(events: list[tuple[str, int, int]]) -> Any:
    """A caller-defined source subclass exports a portable builtin fallback."""
    import attrs
    from qualtran.drawing.musical_score import WireSymbol

    def draw(self: Any, ax: Axes, x: int, y: int) -> None:
        events.append((self.text, x, y))
        ax.text(
            x,
            -y,
            self.text,
            transform=ax.transData,
            fontsize=11,
            rotation=15,
            ha="center",
            va="center",
            bbox={"boxstyle": "round4", "fc": "lightyellow", "ec": "purple"},
        )

    def json_dict(self: Any) -> dict[str, object]:
        return {"symb_cls": "TextBox", "symb_attributes": {"text": self.text}}

    return attrs.frozen(
        type(
            "Badge",
            (WireSymbol,),
            {
                "__annotations__": {"text": str},
                "draw": draw,
                "json_dict": json_dict,
            },
        )
    )


def source_score(glyph: Any) -> Any:
    import qualtran.drawing.musical_score as source

    return source.MusicalScoreData(
        max_x=2,
        max_y=2,
        soqs=[
            source.SoqData(
                source.Text("in", fontsize=8), source.RegPosition(1, -1, 0), "in"
            ),
            source.SoqData(glyph, source.RegPosition(1, 0, 1), "badge"),
            source.SoqData(source.ModPlus(), source.RegPosition(2, 0, 1), "target"),
            source.SoqData(
                source.Text("out", fontsize=8), source.RegPosition(1, 2, 2), "out"
            ),
        ],
        hlines=[
            source.HLine(1, -1, 2),
            source.HLine(2, -1, 2, source.HLineFlavor.CLASSICAL),
        ],
        vlines=[source.VLine(0, 2, 1, source.Text("caption", fontsize=13))],
    )


def native_records(original: Any) -> MusicalScoreData:
    from qualtran.drawing.musical_score import MusicalScoreEncoder

    records: MusicalScoreData = json.loads(
        json.dumps(original, cls=MusicalScoreEncoder)
    )
    return records


def assert_pixels(actual: Figure, expected: Figure) -> None:
    # Supported Matplotlib releases leave these Agg methods unannotated.
    native_canvas: Any = FigureCanvasAgg(actual)
    source_canvas: Any = FigureCanvasAgg(expected)
    native_canvas.draw()
    source_canvas.draw()
    assert np.array_equal(native_canvas.buffer_rgba(), source_canvas.buffer_rgba())
    assert np.array_equal(actual.get_size_inches(), expected.get_size_inches())


@pytest.mark.parametrize("false_valued", [False, True])
@pytest.mark.parametrize("max_width,max_height", [(10.0, 8.0), (2.0, 8.0), (10.0, 1.0)])
def test_custom_draw_dispatch_and_builtin_delegation_match_source(
    false_valued: bool,
    max_width: float,
    max_height: float,
    tmp_path: Path,
) -> None:
    from qualtran.drawing.musical_score import draw_musical_score

    expected_events: list[tuple[str, int, int]] = []
    actual_events: list[tuple[str, int, int]] = []
    visited: list[str] = []
    badge = source_badge(expected_events)("θ†")
    assert badge.adjoint() is badge
    original = source_score(badge)
    records = native_records(original)
    before = export_register_score_json(records)

    class Painter:
        def __bool__(self) -> bool:
            return not false_valued

        def __call__(self, ax: Axes, port: ScorePort) -> None:
            visited.append(port["ident"])
            if port["ident"] != "badge":
                draw_register_score_port(ax, port)
                return
            text = str(port["symb_attributes"]["text"])
            x, y = port["seq_x"], port["y"]
            actual_events.append((text, x, y))
            ax.text(
                x,
                -y,
                text,
                transform=ax.transData,
                fontsize=11,
                rotation=15,
                ha="center",
                va="center",
                bbox={"boxstyle": "round4", "fc": "lightyellow", "ec": "purple"},
            )

    with plt.rc_context({"figure.dpi": 100, "font.family": "DejaVu Sans"}):
        expected, expected_ax = draw_musical_score(
            original, max_width=max_width, max_height=max_height
        )
        actual, actual_ax = draw_register_score(
            records, max_width=max_width, max_height=max_height, port_drawer=Painter()
        )
        try:
            assert actual_events == expected_events == [("θ†", 0, 1)]
            assert visited == [p["ident"] for p in records["soqs"]]
            assert actual_ax.get_xlim() == expected_ax.get_xlim()
            assert actual_ax.get_ylim() == expected_ax.get_ylim()
            assert_pixels(actual, expected)
            assert before == export_register_score_json(records)
            assert set(records["soqs"][1]) == {
                "symb_cls",
                "symb_attributes",
                "ident",
                "y",
                "seq_x",
                "topo_gen",
            }
            assert records["soqs"][1]["symb_cls"] == "TextBox"
            actual.savefig(tmp_path / "custom-symbol.png")
        finally:
            plt.close(actual)
            plt.close(expected)


@pytest.mark.parametrize(
    ("glyph", "attributes"),
    [
        ("Text", {"text": "θ†", "fontsize": 13}),
        ("TextBox", {"text": "θ†"}),
        ("RarrowTextBox", {"text": "state"}),
        ("LarrowTextBox", {"text": "effect"}),
        ("Circle", {"filled": True}),
        ("Circle", {"filled": False}),
        ("ModPlus", {}),
    ],
)
def test_explicit_builtin_replacement_retains_all_source_pixels(
    glyph: SymbolClass,
    attributes: dict[str, str | int | bool],
) -> None:
    import qualtran.drawing.musical_score as source

    original = source_score(getattr(source, glyph)(**attributes))
    data = native_records(original)
    expected, _ = source.draw_musical_score(original)
    actual, _ = draw_register_score(data, port_drawer=draw_register_score_port)
    default, _ = draw_register_score(data)
    try:
        assert_pixels(actual, expected)
        assert_pixels(default, actual)
    finally:
        plt.close(expected)
        plt.close(actual)
        plt.close(default)


def test_callback_failure_closes_owned_figure_and_retry_preserves_records() -> None:
    import qualtran.drawing.musical_score as source

    records = native_records(source_score(source.TextBox("badge")))
    before = export_register_score_json(records)
    figures = plt.get_fignums()
    failure = RuntimeError("custom paint failed")
    seen: list[str] = []

    def fail(ax: Axes, port: ScorePort) -> None:
        seen.append(port["ident"])
        draw_register_score_port(ax, port)
        if port["ident"] == "badge":
            raise failure

    with pytest.raises(RuntimeError) as caught:
        draw_register_score(records, port_drawer=fail)
    assert caught.value is failure
    assert seen == ["in", "badge"]
    assert plt.get_fignums() == figures
    assert export_register_score_json(records) == before
    badge = source_badge([])

    def source_fail(self: Any, ax: Axes, x: int, y: int) -> None:
        raise failure

    badge.draw = source_fail
    try:
        with pytest.raises(RuntimeError) as source_caught:
            source.draw_musical_score(source_score(badge("broken")))
        assert source_caught.value is failure
        assert len(set(plt.get_fignums()) - set(figures)) == 1
    finally:
        for number in set(plt.get_fignums()) - set(figures):
            plt.close(number)
    retried, _ = draw_register_score(records)
    plt.close(retried)
    assert plt.get_fignums() == figures


def test_empty_score_has_no_port_callback_and_returns_owned_figure() -> None:
    data: MusicalScoreData = {
        "max_x": 0,
        "max_y": 0,
        "soqs": [],
        "hlines": [],
        "vlines": [],
    }

    def unexpected(ax: Axes, port: ScorePort) -> None:
        pytest.fail("empty score has no port draw")

    fig, ax = draw_register_score(data, port_drawer=unexpected)
    try:
        assert ax.figure is fig
        assert plt.gcf() is fig
    finally:
        plt.close(fig)
