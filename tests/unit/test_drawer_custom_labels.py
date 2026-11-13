"""Replace connector label drawing against frozen source Text subclass dispatch."""

from __future__ import annotations

from hashlib import sha1
from pathlib import Path
from typing import Any

import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from flagquantum.drawer.register_score_data import (
    MusicalScoreData,
    ScorePort,
    ScoreVerticalLine,
    export_register_score_json,
)
from flagquantum.drawer.register_score_mpl import (
    draw_register_score,
    draw_register_score_label,
    draw_register_score_port,
)
from tests.unit.test_drawer_custom_symbols import assert_pixels, native_records

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
Event = tuple[str, str, int, float, int]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    raw = (
        Path(__file__).resolve().parents[3]
        / "ftqc-reference-env/qualtran/drawing/musical_score.py"
    ).read_bytes()
    assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == (
        "3863bb66cdba17c4654657e275a8485d94b8da75"
    )


def source_labels(events: list[Event], bottom: int) -> Any:
    import qualtran.drawing.musical_score as source

    def label_draw(self: Any, ax: Axes, x: int, y: float) -> None:
        events.append(("label", self.text, x, y, len(ax.collections)))
        if self.text == "ordinary":
            source.Text.draw(self, ax, x, y)
            return
        ax.text(
            x,
            -y,
            self.text,
            transform=ax.transData,
            fontsize=self.fontsize,
            rotation=-12,
            color="darkgreen",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.25", "fc": "lightyellow", "ec": "green"},
        )

    def port_draw(self: Any, ax: Axes, x: int, y: int) -> None:
        events.append(("port", self.text, x, y, len(ax.collections)))
        source.Text.draw(self, ax, x, y)

    def port_json(self: Any) -> dict[str, object]:
        return {
            "symb_cls": "Text",
            "symb_attributes": {"text": self.text, "fontsize": self.fontsize},
        }

    label_class = type("CustomLabel", (source.Text,), {"draw": label_draw})
    port_class = type(
        "ObservedPort", (source.Text,), {"draw": port_draw, "json_dict": port_json}
    )
    return source.MusicalScoreData(
        max_x=3,
        max_y=bottom + 1,
        soqs=[
            source.SoqData(
                port_class("in", fontsize=8), source.RegPosition(bottom, -1, 0), "in"
            ),
            source.SoqData(
                port_class("out", fontsize=8), source.RegPosition(bottom, 3, 2), "out"
            ),
        ],
        hlines=[
            source.HLine(bottom, -1, 3),
            source.HLine(bottom + 1, -1, 3, source.HLineFlavor.CLASSICAL),
        ],
        vlines=[
            source.VLine(x, bottom + 1, bottom, label_class(text, fontsize=13))
            for x, text in enumerate(("", " ", "θ†", "ordinary"))
        ],
    )


@pytest.mark.parametrize("false_valued", [False, True])
@pytest.mark.parametrize(
    ("bottom", "max_width", "max_height"),
    [(0, 10.0, 8.0), (2, 2.0, 8.0), (5, 10.0, 1.0)],
)
def test_custom_labels_match_pixels_order_coordinates_and_empty_gate(
    false_valued: bool,
    bottom: int,
    max_width: float,
    max_height: float,
    tmp_path: Path,
) -> None:
    from qualtran.drawing.musical_score import draw_musical_score

    source_events: list[Event] = []
    actual_events: list[Event] = []
    original = source_labels(source_events, bottom)
    data = native_records(original)
    before = export_register_score_json(data)
    seen: list[ScoreVerticalLine] = []

    class LabelPainter:
        def __bool__(self) -> bool:
            return not false_valued

        def __call__(self, ax: Axes, connector: ScoreVerticalLine) -> None:
            seen.append(connector)
            label = connector["label"]
            x, y = connector["x"], connector["bottom_y"] - 0.5
            actual_events.append(("label", label["text"], x, y, len(ax.collections)))
            if label["text"] == "ordinary":
                draw_register_score_label(ax, connector)
                return
            ax.text(
                x,
                -y,
                label["text"],
                transform=ax.transData,
                fontsize=label["fontsize"],
                rotation=-12,
                color="darkgreen",
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.25", "fc": "lightyellow", "ec": "green"},
            )

    def port_painter(ax: Axes, port: ScorePort) -> None:
        actual_events.append(
            (
                "port",
                str(port["symb_attributes"]["text"]),
                port["seq_x"],
                port["y"],
                len(ax.collections),
            )
        )
        draw_register_score_port(ax, port)

    with plt.rc_context({"figure.dpi": 100, "font.family": "DejaVu Sans"}):
        expected, expected_ax = draw_musical_score(
            original, max_width=max_width, max_height=max_height
        )
        actual, actual_ax = draw_register_score(
            data,
            max_width=max_width,
            max_height=max_height,
            port_drawer=port_painter,
            label_drawer=LabelPainter(),
        )
        try:
            assert (
                actual_events
                == source_events
                == [
                    ("label", " ", 1, bottom - 0.5, 4),
                    ("label", "θ†", 2, bottom - 0.5, 5),
                    ("label", "ordinary", 3, bottom - 0.5, 6),
                    ("port", "in", -1, bottom, 6),
                    ("port", "out", 3, bottom, 6),
                ]
            )
            assert all(a is b for a, b in zip(seen, data["vlines"][1:], strict=True))
            assert actual_ax.get_xlim() == expected_ax.get_xlim()
            assert actual_ax.get_ylim() == expected_ax.get_ylim()
            assert_pixels(actual, expected)
            assert export_register_score_json(data) == before
            assert all(
                set(line["label"]) == {"text", "fontsize"} for line in data["vlines"]
            )
            actual.savefig(tmp_path / "custom-label.png")
        finally:
            plt.close(actual)
            plt.close(expected)


@pytest.mark.parametrize("fontsize", [8, 13])
def test_builtin_label_replacement_matches_source_and_default(fontsize: int) -> None:
    import qualtran.drawing.musical_score as source

    original = source_labels([], 0)
    original.vlines = [
        source.VLine(
            connector.x,
            connector.top_y,
            connector.bottom_y,
            source.Text(connector.label.text, fontsize=fontsize),
        )
        for connector in original.vlines
    ]
    data = native_records(original)
    expected, _ = source.draw_musical_score(original)
    actual, _ = draw_register_score(data, label_drawer=draw_register_score_label)
    default, _ = draw_register_score(data)
    try:
        assert_pixels(actual, expected)
        assert_pixels(default, actual)
    finally:
        plt.close(expected)
        plt.close(actual)
        plt.close(default)


@pytest.mark.parametrize("exception_type", [RuntimeError, KeyboardInterrupt])
def test_label_failure_stops_before_ports_closes_native_figure_and_retries(
    exception_type: type[BaseException],
) -> None:
    import qualtran.drawing.musical_score as source

    original = source_labels([], 0)
    data = native_records(original)
    before = export_register_score_json(data)
    figures = plt.get_fignums()
    failure = exception_type("label paint failed")
    calls: list[str] = []

    def fail(ax: Axes, connector: ScoreVerticalLine) -> None:
        calls.append(connector["label"]["text"])
        draw_register_score_label(ax, connector)
        raise failure

    def unexpected(ax: Axes, port: ScorePort) -> None:
        pytest.fail("port drawing must stop after label failure")

    with pytest.raises(exception_type) as caught:
        draw_register_score(data, label_drawer=fail, port_drawer=unexpected)
    assert caught.value is failure
    assert calls == [" "]
    assert plt.get_fignums() == figures
    assert export_register_score_json(data) == before

    def source_fail(self: Any, ax: Axes, x: int, y: float) -> None:
        raise failure

    broken = type("BrokenLabel", (source.Text,), {"draw": source_fail})
    line = original.vlines[1]
    original.vlines[1] = source.VLine(line.x, line.top_y, line.bottom_y, broken(" "))
    try:
        with pytest.raises(exception_type) as source_caught:
            source.draw_musical_score(original)
        assert source_caught.value is failure
        assert len(set(plt.get_fignums()) - set(figures)) == 1
    finally:
        for number in set(plt.get_fignums()) - set(figures):
            plt.close(number)
    retried, _ = draw_register_score(data)
    plt.close(retried)
    assert plt.get_fignums() == figures


def test_empty_score_returns_owned_figure_without_label_callback() -> None:
    data: MusicalScoreData = {
        "max_x": 0,
        "max_y": 0,
        "soqs": [],
        "hlines": [],
        "vlines": [],
    }

    def unexpected(ax: Axes, connector: ScoreVerticalLine) -> None:
        pytest.fail("empty score has no label draw")

    fig, ax = draw_register_score(data, label_drawer=unexpected)
    try:
        assert ax.figure is fig
        assert plt.gcf() is fig
    finally:
        plt.close(fig)
