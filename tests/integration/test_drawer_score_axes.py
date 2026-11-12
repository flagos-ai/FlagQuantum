"""Compare fixed score geometry and axes with unchanged source browser code."""

from __future__ import annotations

import html
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from flagquantum.drawer.register_score_data import MusicalScoreData
from flagquantum.drawer.register_score_html import export_register_score_html
from tests.integration.test_drawer_score_animation import (
    CLOCK,
    SNAPSHOT,
    assert_snapshots,
    data,
    updated,
)
from tests.integration.test_drawer_score_animation import (
    browser as browser,
)
from tests.integration.test_drawer_score_animation import (
    source_script as source_script,
)
from tests.integration.test_drawer_score_loading import run_loading_page
from tests.integration.test_drawer_score_loading import (
    score_server as score_server,
)

pytestmark = [pytest.mark.integration, pytest.mark.ftqc_reference]

STYLE_SNAPSHOT = """
function styles() {
  const root = document.querySelector('svg');
  const datag = native ? root : DATA_G.node();
  const groups = native ? [...datag.querySelectorAll('g[data-port]')]
    : [...datag.querySelectorAll('g.Text,g.TextBox,g.LarrowTextBox,g.RarrowTextBox')];
  const glyphs = groups.filter(node => node.querySelector('text')).map(node => {
    const key = native ? node.dataset.port : node.__data__.ident;
    const text = node.querySelector('text');
    const font = getComputedStyle(text);
    const shape = node.querySelector('rect,polygon');
    return {key, family: font.fontFamily, size: font.fontSize,
      anchor: font.textAnchor, baseline: font.dominantBaseline,
      outline: shape ? ['x','y','width','height','points'].map(attr => shape.getAttribute(attr)) : null};
  }).sort((a,b) => a.key.localeCompare(b.key));
  const axisGroups = native ? root.querySelectorAll('.score-axis') : [...root.children].slice(0,2);
  const axes = [...axisGroups].map(axis => ({transform: axis.getAttribute('transform'),
    domain: !!axis.querySelector('.domain'),
    ticks: [...axis.querySelectorAll('.tick')].map(tick => {
      const line = tick.querySelector('line'), text = tick.querySelector('text');
      const font = getComputedStyle(text);
      return {position: position(tick).map(value => Number(value.toFixed(6))), text: text.textContent,
        end: [line.getAttribute('x2'),line.getAttribute('y2')],
        label: ['x','y','dy'].map(attr => text.getAttribute(attr)),
        style: [font.fontSize,font.fontFamily,font.textAnchor,font.dominantBaseline,font.fill]};
    })}));
  return {size: ['width','height','viewBox'].map(attr => root.getAttribute(attr).replaceAll(',', ' ')),
    glyphs, axes, strokes: [...datag.querySelectorAll('line.h')].map(line => line.getAttribute('stroke'))};
}
"""


def run_fixed_page(
    browser: str,
    tmp_path: Path,
    original: MusicalScoreData,
    probe: str,
    *,
    source: str | None = None,
    pixel_ratio: int = 1,
) -> Any:
    """Use source domains directly, without changing its scale ranges."""
    native = source is None
    prelude = "<script>" + CLOCK + "</script>"
    if native:
        page = export_register_score_html(original).replace(
            '<option value="fixed">', '<option value="fixed" selected>'
        )
        page = page.replace("<script>", prelude + "<script>", 1)
        initialize = "advance(0);"
    else:
        page = (
            '<html><body><div id="content"></div>'
            + prelude
            + f"<script>const xDomain = [-2,{original['max_x'] + 2}];"
            + f"const yDomain = [0,{original['max_y'] + 1}];</script><script>"
            + str(source)
            + "</script></body></html>"
        )
        initialize = f"draw({json.dumps(original)});"
    script = (
        '<pre id="result"></pre><script>'
        + f"const native = {json.dumps(native)};\n"
        + SNAPSHOT
        + STYLE_SNAPSHOT
        + initialize
        + "\ntry {"
        + probe
        + "\n} catch (error) {document.getElementById('result').textContent = JSON.stringify({error: error.stack});}"
        + "</script>"
    )
    path = tmp_path / ("fixed-native.html" if native else "fixed-source.html")
    path.write_text(page.replace("</html>", script + "</html>"))
    result = subprocess.run(
        [
            browser,
            "--headless",
            "--disable-background-networking",
            "--no-first-run",
            f"--force-device-scale-factor={pixel_ratio}",
            f"--user-data-dir={tmp_path / ('fixed-native-profile' if native else 'fixed-source-profile')}",
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
    output = json.loads(html.unescape(match[1]))
    assert "error" not in output, output
    return output


@pytest.mark.parametrize("pixel_ratio", [1, 2])
@pytest.mark.parametrize("max_x,max_y", [(0, 0), (3, 6), (17, 12), (2000, 3)])
def test_fixed_axes_fonts_shapes_and_geometry_match_source(
    browser: str,
    source_script: str,
    tmp_path: Path,
    max_x: int,
    max_y: int,
    pixel_ratio: int,
) -> None:
    original = data()
    original["max_x"], original["max_y"] = max_x, max_y
    if max_y == 0:
        for port in original["soqs"]:
            port["y"] = 0
        original["hlines"] = original["hlines"][:1]
        original["vlines"] = []
    # Source ignores explicit fontsize and long text when sizing fixed boxes.
    for port in original["soqs"]:
        if "text" in port["symb_attributes"]:
            port["symb_attributes"].update(text="long score label", fontsize=27)
    original["hlines"][0]["flavor"] = "HLineFlavor.CLASSICAL"
    probe = "advance(750); document.getElementById('result').textContent = JSON.stringify({geometry: snapshot(), styles: styles()});"
    actual = run_fixed_page(browser, tmp_path, original, probe, pixel_ratio=pixel_ratio)
    expected = run_fixed_page(
        browser,
        tmp_path,
        original,
        probe,
        source=source_script,
        pixel_ratio=pixel_ratio,
    )
    assert_snapshots([actual["geometry"]], [expected["geometry"]])
    assert actual["styles"] == expected["styles"]


def test_fixed_updates_exit_and_style_switch_preserve_shared_nodes(
    browser: str, source_script: str, tmp_path: Path
) -> None:
    original = data()
    original["vlines"][0]["label"]["text"] = "native caption"
    second = updated(original)
    probe = (
        "const samples = [];"
        + "for (const dt of [0,93.75,93.75,187.5,187.5,187.5]) {advance(dt); samples.push(snapshot());}"
        + "const initial = styles();"
        + f"draw({json.dumps(second)});"
        + "for (const dt of [0,93.75,93.75,187.5,187.5,187.5]) {advance(dt); samples.push(snapshot());}"
        + "const final = styles();"
        + "if (native) {const node = ports.get('TextBox,2').node;"
        + "styleSelector.value = 'adaptive'; styleSelector.dispatchEvent(new Event('change')); advance(750);"
        + "if (svg.querySelectorAll('.score-axis').length) throw Error('Stale axes');"
        + "if (getComputedStyle(node.querySelector('text')).fontSize !== '12px') throw Error('Adaptive font');"
        + "styleSelector.value = 'fixed'; styleSelector.dispatchEvent(new Event('change')); advance(750);"
        + "if (ports.get('TextBox,2').node !== node) throw Error('Replaced glyph');"
        + "if (JSON.stringify(styles()) !== JSON.stringify(final)) throw Error('Style roundtrip');}"
        + "document.getElementById('result').textContent = JSON.stringify({initial,final,samples});"
    )
    actual = run_fixed_page(browser, tmp_path, original, probe)
    expected = run_fixed_page(browser, tmp_path, original, probe, source=source_script)
    assert_snapshots(actual["samples"], expected["samples"])
    assert actual["initial"] == expected["initial"]
    assert actual["final"] == expected["final"]


def test_fixed_empty_score_and_dependency_domains(browser: str, tmp_path: Path) -> None:
    original = data()
    probe = """
advance(750);
selector.value = 'dependency'; selector.dispatchEvent(new Event('change')); advance(750);
const dependency = views.dependency;
if (Number(svg.getAttribute('width')) !== (dependency.max_x + 4) * 55 + 82) throw Error('Dependency width');
if (Number(svg.getAttribute('height')) !== (dependency.max_y + 1) * 42 + 62) throw Error('Dependency height');
views.sequential = {max_x:0,max_y:0,soqs:[],hlines:[],vlines:[]};
selector.value = 'sequential'; selector.dispatchEvent(new Event('change')); advance(750);
document.getElementById('result').textContent = JSON.stringify({ports: ports.size,axes: styles().axes,size: styles().size});
"""
    result = run_fixed_page(browser, tmp_path, original, probe)
    assert result["ports"] == 0
    assert len(result["axes"]) == 2
    assert result["size"] == ["302", "104", "0 0 302 104"]


def test_http_import_recomputes_fixed_domains_without_old_dependency(
    browser: str, source_script: str, tmp_path: Path, score_server: str
) -> None:
    score = updated(data())
    score["max_x"], score["max_y"] = 17, 12
    (tmp_path / "fixed.json").write_text(json.dumps(score))
    probe = STYLE_SNAPSHOT + """
styleSelector.value = 'fixed'; styleSelector.dispatchEvent(new Event('change')); advance(750);
await load('fixed.json', null);
if (selector.value !== 'sequential' || !selector.querySelector('option[value="dependency"]').disabled)
  throw Error('Imported view ownership');
if (styleSelector.value !== 'fixed') throw Error('Lost fixed style');
document.getElementById('result').textContent = JSON.stringify({geometry:snapshot(),styles:styles()});
"""
    actual = run_loading_page(browser, tmp_path, score_server, probe)
    expected = run_fixed_page(
        browser,
        tmp_path,
        score,
        "advance(750); document.getElementById('result').textContent = JSON.stringify({geometry:snapshot(),styles:styles()});",
        source=source_script,
    )
    assert_snapshots([actual["geometry"]], [expected["geometry"]])
    assert actual["styles"] == expected["styles"]


def test_original_multi_and_fixed_style_matches_source_without_scale_mutation(
    browser: str, source_script: str, tmp_path: Path
) -> None:
    from qualtran.bloqs.mcmt import MultiAnd
    from qualtran.drawing.musical_score import (
        MusicalScoreEncoder,
        get_musical_score_data,
    )

    score: MusicalScoreData = json.loads(
        json.dumps(
            get_musical_score_data(MultiAnd((1, 1, 0, 1)).decompose_bloq()),
            cls=MusicalScoreEncoder,
        )
    )
    probe = "advance(750);document.getElementById('result').textContent = JSON.stringify({geometry:snapshot(),styles:styles()});"
    actual = run_fixed_page(browser, tmp_path, score, probe)
    expected = run_fixed_page(browser, tmp_path, score, probe, source=source_script)
    assert_snapshots([actual["geometry"]], [expected["geometry"]])
    assert actual["styles"] == expected["styles"]
