"""Compare native browser transitions with the frozen, unmodified score script."""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from flagquantum.drawer.register_score_data import MusicalScoreData, SymbolClass
from flagquantum.drawer.register_score_html import export_register_score_html

pytestmark = [pytest.mark.integration, pytest.mark.ftqc_reference]
REFERENCE = Path(__file__).resolve().parents[3] / "ftqc-reference-env"

CLOCK = """
let clock = 1000;
let serial = 0;
const frames = new Map();
Object.defineProperty(performance, "now", {value: () => clock});
window.requestAnimationFrame = callback => {frames.set(++serial, callback); return serial;};
window.cancelAnimationFrame = id => frames.delete(id);
function advance(ms) {
  clock += ms;
  const pending = [...frames.values()];
  frames.clear();
  for (const callback of pending) callback(clock);
  if (typeof d3 !== "undefined") d3.timerFlush();
}
"""

SNAPSHOT = """
function position(node) {
  if (node.tagName === "circle") return [Number(node.getAttribute("cx")), Number(node.getAttribute("cy"))];
  const matrix = node.transform.baseVal.consolidate().matrix;
  return [matrix.e, matrix.f];
}
function snapshot() {
  const groups = native
    ? [...document.querySelectorAll("g[data-port]")]
    : [...DATA_G.node().querySelectorAll("circle.circle,g.modplus,g.TextBox,g.Text,g.LarrowTextBox,g.RarrowTextBox")];
  const glyphs = groups.map(node => {
    const kind = native ? node.getAttribute("class") : node.__data__.symb_cls;
    const identity = native ? node.dataset.port : node.__data__.ident;
    const circle = node.tagName === "circle" ? node : node.querySelector("circle");
    const text = node.querySelector("text");
    return {key: kind + "," + identity, position: position(node),
      opacity: Number(node.getAttribute("opacity") ?? 1),
      fill: kind === "Circle" ? circle.getAttribute("fill") : "",
      text: text ? text.textContent : ""};
  }).sort((a,b) => a.key.localeCompare(b.key));
  function lines(kind) {
    const records = native
      ? [...(kind === "h" ? horizontal : vertical).values()]
      : [...DATA_G.node().querySelectorAll("line." + kind)].map(node => ({node, data: node.__data__}));
    return records.map(({node, data}) => ({key: kind === "h" ? `${data.y},${data.seq_x_start}` : String(data.x),
      coordinates: ["x1", "x2", "y1", "y2"].map(attr => Number(node.getAttribute(attr)))}))
      .sort((a,b) => a.key.localeCompare(b.key));
  }
  return {glyphs, horizontal: lines("h"), vertical: lines("v")};
}
function draw(data) {
  if (native) {
    views.sequential = data;
    selector.value = "sequential";
    selector.dispatchEvent(new Event("change"));
  } else musicalScore(data.soqs, data.vlines, data.hlines, {x: d => d.seq_x, y: d => d.y});
  advance(0);
}
"""


@pytest.fixture(scope="module")
def browser() -> str:
    executable = os.environ.get("FLAGQUANTUM_CHROME") or shutil.which("chromium")
    if executable is None:
        pytest.skip("Set FLAGQUANTUM_CHROME to run real browser animation checks")
    return executable


@pytest.fixture(scope="module")
def source_script() -> str:
    for name, blob in {
        "musical_score.html": "8731aa90a752d4fd252ea931f0527266da0c3934",
        "musical_score.js": "ead9b24fc77b0a772e05f18280a734ddf65ee3a1",
    }.items():
        raw = (REFERENCE / "qualtran/drawing" / name).read_bytes()
        assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob
    library = (REFERENCE / "d3-7.9.0.min.js").read_bytes()
    assert hashlib.sha256(library).hexdigest() == (
        "f2094bbf6141b359722c4fe454eb6c4b0f0e42cc10cc7af921fc158fceb86539"
    )
    return (
        library.decode()
        + "\n</script><script>"
        + (REFERENCE / "qualtran/drawing/musical_score.js").read_text()
    )


def data() -> MusicalScoreData:
    kinds: tuple[SymbolClass, ...] = (
        "Circle",
        "ModPlus",
        "TextBox",
        "LarrowTextBox",
        "RarrowTextBox",
        "Text",
    )
    score: MusicalScoreData = {
        "max_x": 3,
        "max_y": 6,
        "soqs": [
            {
                "symb_cls": kind,
                "symb_attributes": (
                    {"filled": True}
                    if kind == "Circle"
                    else {} if kind == "ModPlus" else {"text": kind[:1]}
                ),
                "ident": str(i),
                "seq_x": 0,
                "topo_gen": 1,
                "y": i,
            }
            for i, kind in enumerate(kinds)
        ],
        "hlines": [
            {"y": i, "seq_x_start": -1, "seq_x_end": 3, "flavor": "HLineFlavor.QUANTUM"}
            for i in range(6)
        ],
        "vlines": [
            {"x": 0, "top_y": 0, "bottom_y": 5, "label": {"text": "", "fontsize": 10}}
        ],
    }
    for side, x, generation in (("l", -1, 0), ("r", 3, 2)):
        score["soqs"].append(
            {
                "symb_cls": "Text",
                "symb_attributes": {"text": "q"},
                "ident": f"dang,{side}",
                "seq_x": x,
                "topo_gen": generation,
                "y": 0,
            }
        )
    return score


def updated(original: MusicalScoreData) -> MusicalScoreData:
    score = deepcopy(original)
    for port in score["soqs"][:6]:
        port["seq_x"] = 2
        port["y"] = 5 - port["y"]
        if port["symb_cls"] == "Circle":
            port["symb_attributes"]["filled"] = False
        elif port["symb_cls"] not in {"ModPlus"}:
            port["symb_attributes"]["text"] = "new"
    score["soqs"] = [port for port in score["soqs"] if port["ident"] != "3"]
    score["soqs"][4]["symb_cls"] = "TextBox"
    score["soqs"].append(
        {
            "symb_cls": "Circle",
            "symb_attributes": {"filled": True},
            "ident": "new",
            "seq_x": 2,
            "topo_gen": 1,
            "y": 6,
        }
    )
    score["hlines"][0]["seq_x_end"] = 4
    score["hlines"].pop(1)
    score["hlines"].append(
        {"y": 6, "seq_x_start": 0, "seq_x_end": 4, "flavor": "HLineFlavor.QUANTUM"}
    )
    score["vlines"][0]["bottom_y"] = 2
    score["vlines"].append(
        {"x": 2, "top_y": 0, "bottom_y": 6, "label": {"text": "", "fontsize": 10}}
    )
    return score


def run_page(
    browser: str,
    tmp_path: Path,
    original: MusicalScoreData,
    probe: str,
    *,
    source: str | None = None,
) -> Any:
    native = source is None
    prelude = "<script>" + CLOCK + "</script>"
    if native:
        page = export_register_score_html(original).replace(
            "<script>", prelude + "<script>", 1
        )
        initialize = "advance(0);"
    else:
        assert source is not None
        page = (
            '<html><body><div id="content"></div>'
            + prelude
            + (
                "<script>const xDomain = [-2,5]; const yDomain = [0,7];</script><script>"
                + source
                + "</script></body></html>"
            )
        )
        # Configure the source's existing mutable scales to the native viewport.
        # The frozen script itself is inserted verbatim above.
        initialize = "xScale.range([0,385]); yScale.range([326,32]);"
    code = f"const native = {json.dumps(native)};\n" + SNAPSHOT + initialize
    code += (
        "\ntry {\n"
        + probe
        + "\n} catch (error) {document.body.dataset.error = error.stack;}"
    )
    script = '<pre id="result"></pre><script>' + code + "</script>"
    path = tmp_path / ("native.html" if native else "source.html")
    path.write_text(page.replace("</html>", script + "</html>"))
    result = subprocess.run(
        [
            browser,
            "--headless",
            "--disable-background-networking",
            "--no-first-run",
            f"--user-data-dir={tmp_path / ('native-profile' if native else 'source-profile')}",
            "--dump-dom",
            path.as_uri(),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    match = re.search(r'<pre id="result">(.*?)</pre>', result.stdout, re.S)
    assert match and match[1], result.stdout[-12000:] + result.stderr
    return json.loads(html.unescape(match[1]))


def assert_snapshots(actual: Any, expected: Any) -> None:
    assert len(actual) == len(expected)
    for native, source in zip(actual, expected, strict=True):
        for group in ("glyphs", "horizontal", "vertical"):
            assert len(native[group]) == len(source[group])
            for row, other in zip(native[group], source[group], strict=True):
                assert row.keys() == other.keys()
                for key in row:
                    if key in {"coordinates", "position", "opacity"}:
                        assert row[key] == pytest.approx(other[key], abs=2e-5)
                    else:
                        assert row[key] == other[key]


def test_all_glyphs_enter_update_exit_and_cubic_samples_match_unmodified_source(
    browser: str,
    source_script: str,
    tmp_path: Path,
) -> None:
    original = data()
    second = updated(original)
    empty: MusicalScoreData = {
        "max_x": 3,
        "max_y": 6,
        "soqs": [],
        "hlines": [],
        "vlines": [],
    }
    probe = "const history = [];\n"
    for score in (original, second, empty):
        probe += "draw(" + json.dumps(score) + ");history.push(snapshot());\n"
        for delta in (93.75, 93.75, 187.5, 187.5, 187.5):
            probe += f"advance({delta});history.push(snapshot());\n"
    probe += 'document.getElementById("result").textContent = JSON.stringify(history);'
    actual = run_page(browser, tmp_path, original, probe)
    expected = run_page(browser, tmp_path, original, probe, source=source_script)
    assert_snapshots(actual, expected)
    assert len(actual[0]["glyphs"]) == 8
    assert actual[0]["glyphs"][0]["opacity"] == 0
    assert actual[1]["glyphs"][0]["opacity"] == pytest.approx(0.0078125)
    assert actual[-1] == {"glyphs": [], "horizontal": [], "vertical": []}


def test_interrupted_exit_reuses_dom_and_resumes_from_current_geometry(
    browser: str,
    source_script: str,
    tmp_path: Path,
) -> None:
    original = data()
    second = updated(original)
    probe = "draw(" + json.dumps(original) + ");advance(750);\n"
    probe += "const node = native ? ports.get('LarrowTextBox,3').node : DATA_G.node().querySelector('g.LarrowTextBox');\n"
    probe += (
        "draw(" + json.dumps(second) + ");advance(187.5);const middle = snapshot();\n"
    )
    probe += (
        "draw(" + json.dumps(original) + ");const resumed = snapshot();advance(750);\n"
    )
    probe += "const retained = native ? ports.get('LarrowTextBox,3').node : DATA_G.node().querySelector('g.LarrowTextBox');\n"
    probe += 'document.getElementById("result").textContent = JSON.stringify({same: node === retained, states: [middle, resumed, snapshot()]});'
    actual = run_page(browser, tmp_path, original, probe)
    expected = run_page(browser, tmp_path, original, probe, source=source_script)
    assert actual["same"] and expected["same"]
    assert_snapshots(actual["states"], expected["states"])
    assert actual["states"][0] != actual["states"][-1]


def test_normal_view_switch_preserves_nodes_and_finishes_geometry(
    browser: str, tmp_path: Path
) -> None:
    original = data()
    original["soqs"][2]["symb_attributes"] = {
        "text": "a long label </script><script>window.injected = true</script>",
        "fontsize": 19,
    }
    original["hlines"][0]["flavor"] = "HLineFlavor.CLASSICAL"
    original["vlines"][0]["label"]["text"] = "caption"
    probe = """
advance(750);
const root = container.querySelector('svg');
const node = ports.get('TextBox,2').node;
const before = snapshot();
selector.value = 'dependency';
selector.dispatchEvent(new Event('change'));
advance(187.5);
const middle = snapshot();
selector.value = 'sequential';
selector.dispatchEvent(new Event('change'));
advance(0);
const resumed = snapshot();
advance(750);
const finished = snapshot();
selector.value = 'dependency';
selector.dispatchEvent(new Event('change'));
advance(750);
const dependency = views.dependency;
const geometry = [...ports.values()].every(({node, data}) => {
  const matrix = node.transform.baseVal.consolidate().matrix;
  const cell = dependency.soqs.reduce((size, port) => Math.max(
    size, (port.symb_attributes.text || '').length * 7 + 14), 55);
  return matrix.e === (data.seq_x + 2) * cell && matrix.f === 32 + data.y * 42;
});
document.getElementById('result').textContent = JSON.stringify({
  same: root === container.querySelector('svg') && node === ports.get('TextBox,2').node,
  before, middle, resumed, finished, geometry,
  count: container.querySelectorAll('svg').length,
  text: node.querySelector('text').textContent,
  fontsize: node.querySelector('text').getAttribute('font-size'),
  width: Number(node.querySelector('rect').getAttribute('width')),
  classical: horizontal.values().next().value.node.getAttribute('stroke'),
  caption: [...captions.values()][0].node.textContent,
  injected: window.injected === true
});
"""
    result = run_page(browser, tmp_path, original, probe)
    assert result["same"] and result["geometry"]
    assert result["count"] == 1
    assert result["middle"] != result["before"]
    assert_snapshots([result["middle"]], [result["resumed"]])
    assert_snapshots([result["before"]], [result["finished"]])
    assert result["text"] == original["soqs"][2]["symb_attributes"]["text"]
    assert result["fontsize"] == "19"
    assert result["width"] > 44
    assert result["classical"] == "blue"
    assert result["caption"] == "caption"
    assert not result["injected"]


def test_empty_offline_page_switches_without_leaving_nodes(
    browser: str, tmp_path: Path
) -> None:
    empty: MusicalScoreData = {
        "max_x": 0,
        "max_y": 0,
        "soqs": [],
        "hlines": [],
        "vlines": [],
    }
    probe = """
advance(750);
selector.value = 'dependency';
selector.dispatchEvent(new Event('change'));
advance(750);
document.getElementById('result').textContent = JSON.stringify(snapshot());
"""
    assert run_page(browser, tmp_path, empty, probe) == {
        "glyphs": [],
        "horizontal": [],
        "vertical": [],
    }
