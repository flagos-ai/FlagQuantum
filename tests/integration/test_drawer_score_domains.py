"""Exercise source manual domains through the native browser bounds form."""

from __future__ import annotations

import html
import json
import re
import subprocess
from pathlib import Path

import pytest

from tests.integration.test_drawer_score_animation import (
    CLOCK,
    REFERENCE,
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
from tests.integration.test_drawer_score_axes import STYLE_SNAPSHOT, run_fixed_page
from tests.integration.test_drawer_score_loading import run_loading_page
from tests.integration.test_drawer_score_loading import (
    score_server as score_server,
)

pytestmark = [pytest.mark.integration, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "domains",
    [
        ((-3.25, 8.75), (-4.5, 2.75)),
        ((0.11, 0.12), (0.13, 0.14)),
        ((0.13, 0.14), (0.11, 0.12)),
        ((1000.25, 1004.75), (-1004.75, -1000.25)),
        ((0.00000001, 0.00000002), (0.00000002, 0.00000004)),
        ((2, 2), (-3, -3)),
        ((1.25, 1.25), (-3.75, -3.75)),
        ((-0.001, -0.001), (-0.1, -0.1)),
        ((1, 0), (0, 1)),
        ((0, 1), (1, 0)),
        ((0.51, 0.49), (0.11, 0.12)),
    ],
)
def test_manual_domains_match_source_canvas_ticks_glyphs_and_baseline(
    browser: str,
    source_script: str,
    tmp_path: Path,
    domains: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    original = data()
    second = updated(original)
    probe = (
        "const initial = styles(); const history = [snapshot()];"
        + f"draw({json.dumps(second)});"
        + "for (const dt of [0,93.75,93.75,187.5,187.5,187.5]) {advance(dt);history.push(snapshot());}"
        + "document.getElementById('result').textContent = JSON.stringify({initial,final:styles(),history});"
    )
    actual = run_fixed_page(browser, tmp_path, original, probe, domains=domains)
    expected = run_fixed_page(
        browser, tmp_path, original, probe, source=source_script, domains=domains
    )
    assert_snapshots(actual["history"], expected["history"])
    assert actual["initial"] == expected["initial"]
    assert actual["final"] == expected["final"]


@pytest.mark.parametrize(
    "bounds",
    [
        ["", "5", "0", "7"],
        ["NaN", "5", "0", "7"],
        ["-2", "Infinity", "0", "7"],
        ["5", "-2", "0", "7"],
        ["-2", "5", "7", "0"],
        ["-1e308", "1e308", "0", "7"],
        ["-2", "5", "-1e308", "1e308"],
        ["1e16", "10000000000000002", "0", "7"],
        ["0", "100000", "0", "7"],
        ["1e20", "1e20", "0", "7"],
    ],
)
def test_invalid_or_unbounded_manual_domains_keep_graph_and_view(
    browser: str, tmp_path: Path, bounds: list[str]
) -> None:
    probe = (
        "advance(750); const before = snapshot(); const priorStyles = styles();"
        + "const node = ports.get('TextBox,2').node; const selected = selector.value;"
        + f"boundInputs.forEach((input,i) => {{input.value = {json.dumps(bounds)}[i];}});"
        + "document.getElementById('score-bounds').dispatchEvent(new Event('submit',{cancelable:true}));"
        + "advance(750);document.getElementById('result').textContent = JSON.stringify({"
        + "before,after:snapshot(),priorStyles,afterStyles:styles(),same:node === ports.get('TextBox,2').node,"
        + "manual:manualDomains,selected,afterSelected:selector.value,"
        + "errors:[...container.querySelectorAll('[role=alert]')].map(node => node.textContent)});"
    )
    result = run_fixed_page(browser, tmp_path, data(), probe)
    assert result["before"] == result["after"]
    assert result["priorStyles"] == result["afterStyles"]
    assert result["selected"] == result["afterSelected"]
    assert result["same"] and result["manual"] is None
    assert len(result["errors"]) == 1
    assert result["errors"][0].startswith("Invalid score boundsRangeError:")


def test_manual_domains_survive_views_styles_and_http_then_fit_selected_view(
    browser: str, tmp_path: Path, score_server: str
) -> None:
    score = updated(data())
    score["max_x"], score["max_y"] = 17, 12
    (tmp_path / "manual.json").write_text(json.dumps(score))
    probe = STYLE_SNAPSHOT + """
const oldNode = ports.get('TextBox,2').node;
boundInputs.forEach((input,i) => {input.value = [-4,8,-2,10][i];});
document.getElementById('score-bounds').dispatchEvent(new Event('submit',{cancelable:true}));advance(750);
const initialSize = styles().size;
selector.value = 'dependency';selector.dispatchEvent(new Event('change'));advance(750);
if (JSON.stringify(styles().size) !== JSON.stringify(initialSize)) throw Error('View lost manual bounds');
styleSelector.value = 'adaptive';styleSelector.dispatchEvent(new Event('change'));advance(750);
styleSelector.value = 'fixed';styleSelector.dispatchEvent(new Event('change'));advance(750);
if (JSON.stringify(styles().size) !== JSON.stringify(initialSize)) throw Error('Style lost manual bounds');
await load('manual.json', null);
if (JSON.stringify(styles().size) !== JSON.stringify(initialSize)) throw Error('Import lost manual bounds');
if (ports.get('TextBox,2').node !== oldNode) throw Error('Import replaced retained glyph');
const manual = snapshot();const domains = manualDomains;
document.getElementById('score-fit').click();advance(750);
document.getElementById('result').textContent = JSON.stringify({domains,manual,
  reset:manualDomains,values:boundInputs.map(input => input.valueAsNumber),fitted:snapshot(),size:styles().size});
"""
    result = run_loading_page(browser, tmp_path, score_server, probe)
    assert result["domains"] == [[-4, 8], [-2, 10]]
    assert result["reset"] is None
    assert result["values"] == [-2, 19, 0, 13]
    assert result["manual"] != result["fitted"]
    assert result["size"] == ["1237", "608", "0 0 1237 608"]


def test_bounds_submit_selects_fixed_and_failed_import_retains_manual_bounds(
    browser: str, tmp_path: Path, score_server: str
) -> None:
    probe = STYLE_SNAPSHOT + """
if (styleSelector.value !== 'adaptive') throw Error('Initial style');
boundInputs.forEach((input,i) => {input.value = [-3,6,-1,8][i];});
document.getElementById('score-bounds').requestSubmit();advance(750);
if (styleSelector.value !== 'fixed') throw Error('Submit did not select fixed');
const initial = snapshot();const size = styles().size;
await load('missing.json', 1);
document.getElementById('result').textContent = JSON.stringify({initial,after:snapshot(),
  size,afterSize:styles().size,domains:manualDomains,errors:errors()});
"""
    result = run_loading_page(browser, tmp_path, score_server, probe)
    assert result["initial"] == result["after"]
    assert result["size"] == result["afterSize"]
    assert result["domains"] == [[-3, 6], [-1, 8]]
    assert len(result["errors"]) == 1


def test_retry_bounds_preserves_committed_selection_then_recovers(
    browser: str, tmp_path: Path
) -> None:
    probe = """
boundInputs.forEach((input,i) => {input.value = [-3,6,-1,8][i];});
document.getElementById('score-bounds').requestSubmit();advance(750);
selector.value = 'dependency';selector.dispatchEvent(new Event('change'));advance(750);
const initial = snapshot();const domains = manualDomains;
boundInputs[0].value = '';
document.getElementById('score-bounds').dispatchEvent(new Event('submit',{cancelable:true}));advance(750);
if (JSON.stringify(manualDomains) !== JSON.stringify(domains) || selector.value !== 'dependency')
  throw Error('Rejected edit changed committed state');
if (JSON.stringify(initial) !== JSON.stringify(snapshot())) throw Error('Rejected edit changed graph');
boundInputs[0].value = '-4';
document.getElementById('score-bounds').requestSubmit();advance(750);
document.getElementById('result').textContent = JSON.stringify({domains:manualDomains,
  selected:selector.value,errors:container.querySelectorAll('[role=alert]').length,
  initial,after:snapshot()});
"""
    result = run_fixed_page(browser, tmp_path, data(), probe)
    assert result["domains"] == [[-4, 6], [-1, 8]]
    assert result["selected"] == "dependency"
    assert result["errors"] == 1
    assert result["initial"] != result["after"]


def test_original_html_load_chain_matches_native_fixed_score(
    browser: str, source_script: str, tmp_path: Path, score_server: str
) -> None:
    """Relocate the CDN resource and inject only the clock/probe in source HTML."""
    from qualtran.bloqs.mcmt import MultiAnd
    from qualtran.drawing.musical_score import (
        MusicalScoreEncoder,
        get_musical_score_data,
    )

    original = json.loads(
        json.dumps(
            get_musical_score_data(MultiAnd((1, 1, 0, 1)).decompose_bloq()),
            cls=MusicalScoreEncoder,
        )
    )
    assert original["max_x"] == 3 and original["max_y"] == 6
    (tmp_path / "musical_score_example.json").write_text(json.dumps(original))
    raw_js = (REFERENCE / "qualtran/drawing/musical_score.js").read_text()
    assert source_script.endswith(raw_js)
    (tmp_path / "musical_score.js").write_text(raw_js)
    (tmp_path / "d3.js").write_bytes((REFERENCE / "d3-7.9.0.min.js").read_bytes())
    page = (REFERENCE / "qualtran/drawing/musical_score.html").read_text()
    assert page.count('src="https://d3js.org/d3.v7.min.js"') == 1
    page = page.replace('src="https://d3js.org/d3.v7.min.js"', 'src="d3.js"')
    page = page.replace("<head>", "<head><script>" + CLOCK + "</script>", 1)
    probe = "const native = false;" + SNAPSHOT + STYLE_SNAPSHOT + """
(async () => {
  try {
    for (let i = 0; i < 300 && !DATA_G.select('g.Text').node(); i++)
      await new Promise(resolve => setTimeout(resolve,5));
    if (!DATA_G.select('g.Text').node()) throw Error('Source HTML did not load JSON');
    advance(0);advance(750);
    document.getElementById('result').textContent = JSON.stringify({geometry:snapshot(),styles:styles()});
  } catch(error) {document.getElementById('result').textContent = JSON.stringify({error:error.stack});}
})();
"""
    page = page.replace(
        "</html>", '<pre id="result"></pre><script>' + probe + "</script></html>"
    )
    (tmp_path / "original-score.html").write_text(page)
    completed = subprocess.run(
        [
            browser,
            "--headless",
            "--disable-background-networking",
            "--no-first-run",
            "--force-device-scale-factor=1",
            "--virtual-time-budget=5000",
            f'--user-data-dir={tmp_path / "original-html-profile"}',
            "--dump-dom",
            f"{score_server}/original-score.html",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    match = re.search(r'<pre id="result">(.*?)</pre>', completed.stdout, re.S)
    assert match and match[1], completed.stdout[-4000:] + completed.stderr[-1000:]
    expected = json.loads(html.unescape(match[1]))
    assert "error" not in expected, expected
    actual = run_fixed_page(
        browser,
        tmp_path,
        original,
        "advance(750);document.getElementById('result').textContent = JSON.stringify({geometry:snapshot(),styles:styles()});",
    )
    assert_snapshots([actual["geometry"]], [expected["geometry"]])
    assert actual["styles"] == expected["styles"]
