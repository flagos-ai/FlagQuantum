"""Compare physical SVG and viewport data with locked graphical output."""

import json
import re
from pathlib import Path
from xml.etree.ElementTree import Element, ParseError, fromstring

import pytest

from flagquantum.compiler.formats.kmb import decode_kmb, encode_kmb
from flagquantum.drawer.reversible_phase_graphics import (
    render_reversible_phase_html,
    render_reversible_phase_svg,
)

_CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "kickmix_circuit_graphics.json").read_text()
)
_PAYLOAD = re.compile(
    r'<script type="application/json" id="circuit-data">(.*?)</script>'
)


def _xml_equal(actual: Element, expected: Element) -> None:
    assert (actual.tag, actual.attrib, actual.text) == (
        expected.tag,
        expected.attrib,
        expected.text,
    )
    for left, right in zip(actual, expected, strict=True):
        _xml_equal(left, right)


def _markup_equal(actual: str, source: str) -> None:
    # Source puts a bare ampersand in the guarded BIT_STORE0 label.
    _xml_equal(
        fromstring(f"<g>{actual}</g>"),
        fromstring(f"<g>{source.replace('&=!', '&amp;=!')}</g>"),
    )


@pytest.mark.parametrize("name", tuple(_CASES))
def test_locked_graphical_svg_and_html(name: str) -> None:
    case = _CASES[name]
    kmx: str = case["kmx"]
    source_svg: str = case["svg"]
    if name == "html_diagram_works_on_all_gates":
        with pytest.raises(ParseError):
            fromstring(source_svg)
        source_svg = source_svg.replace("&=!", "&amp;=!")
    rendered = render_reversible_phase_svg(kmx)
    _xml_equal(fromstring(rendered), fromstring(source_svg))
    html = render_reversible_phase_html(kmx)
    stats = re.search(r"\(Circuit stats: (.*?)\)</div>", html)
    assert stats is not None and stats[1] == case["stats"]
    match = _PAYLOAD.search(html)
    assert match is not None
    data = json.loads(match[1])
    for key in ("transitions", "width", "height"):
        assert data[key] == case[key]
    for (x, near, far), (sx, sn, sf) in zip(
        data["objects"], case["objects"], strict=True
    ):
        assert x == sx
        _markup_equal(near, sn)
        _markup_equal(far, sf)
    for row, source_row in zip(data["wires"], case["wires"], strict=True):
        for markup, source_markup in zip(row, source_row, strict=True):
            _markup_equal(markup, source_markup)
    assert render_reversible_phase_svg(decode_kmb(encode_kmb(kmx))) == rendered


def test_labels_remain_data_in_interactive_html() -> None:
    raw = '</script><img onerror="window.injected=1">` ${window.injected=2}'
    escaped = raw.replace('"', r"\Q")
    text = f'INFO_TAG "@enter:{escaped}"\nX q0'
    html = render_reversible_phase_html(text)
    assert raw not in html
    assert html.count("</script>") == 2
    match = _PAYLOAD.search(html)
    assert match is not None
    markup = json.loads(match[1])["wires"][-1][0]
    assert fromstring(f"<g>{markup}</g>")[-1].text == raw
