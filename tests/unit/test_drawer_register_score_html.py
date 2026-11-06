"""Check offline score embedding and dependency-view lane ownership."""

import json

import pytest

from flagquantum.drawer.register_score import RegisterStep, layout_register_score
from flagquantum.drawer.register_score_data import register_score_data
from flagquantum.drawer.register_score_html import export_register_score_html

pytestmark = pytest.mark.unit


def test_dependency_view_repacks_lanes_when_lifetimes_meet() -> None:
    score = layout_register_score(
        {"a": 1, "b": 1, "first": 1, "second": 1},
        ("a", "b"),
        (
            RegisterStep("open", ("a",), ("a", "first")),
            RegisterStep("prep", ("b",), ("b",)),
            RegisterStep("close", ("a", "first"), ("a",)),
            RegisterStep("open", ("b",), ("b", "second")),
            RegisterStep("close", ("b", "second"), ("b",)),
        ),
    )
    data = register_score_data(score)
    html = export_register_score_html(data)
    payload = html.split('type="application/json">', 1)[1].split("</script>", 1)[0]
    views = json.loads(payload)
    assert views["sequential"] == data

    def port(view: str, identity: str) -> dict:
        return next(row for row in views[view]["soqs"] if row["ident"] == identity)

    first = "2,first,Side.LEFT,()"
    second = "3,second,Side.RIGHT,()"
    assert port("sequential", first)["y"] == port("sequential", second)["y"]
    end, start = port("dependency", first), port("dependency", second)
    assert end["seq_x"] == start["seq_x"] == 2
    assert end["y"] != start["y"]


def test_score_page_escapes_embedded_script_data() -> None:
    register = "</script><script>throw 1</script>"
    score = layout_register_score({register: 1}, (register,), ())
    html = export_register_score_html(register_score_data(score))
    assert register not in html
    assert "\\u003c/script>" in html
    assert "<script src=" not in html
