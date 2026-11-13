"""Exercise HTTP score loading and errors against the frozen browser script."""

from __future__ import annotations

import html
import json
import re
import subprocess
import threading
from collections.abc import Iterator
from copy import deepcopy
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

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

pytestmark = [pytest.mark.integration, pytest.mark.ftqc_reference]


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        """Keep expected error responses out of the test runner's stderr."""


@pytest.fixture
def score_server(tmp_path: Path) -> Iterator[str]:
    handler = partial(QuietHandler, directory=str(tmp_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


ASYNC_PROBE = """
async function waitFor(predicate) {
  for (let i = 0; i < 300; i++) {
    if (predicate()) return;
    await new Promise(resolve => setTimeout(resolve, 5));
  }
  throw new Error('Browser probe timed out');
}
function errors() {
  const parent = document.getElementById(native ? 'score-container' : 'content');
  return [...parent.querySelectorAll('h3')].map(heading => ({
    message: heading.textContent, details: heading.nextElementSibling.textContent,
    color: getComputedStyle(heading).color
  }));
}
async function load(filename, expectedErrors, expectedText = 'new') {
  if (native) {
    document.getElementById('score-url').value = filename;
    document.getElementById('score-load').dispatchEvent(new Event('submit', {cancelable: true}));
  } else make_from_data(filename);
  await waitFor(() => expectedErrors === null
    ? snapshot().glyphs.some(port => port.text === expectedText)
    : errors().length === expectedErrors);
  advance(750);
}
"""


def run_loading_page(
    browser: str,
    tmp_path: Path,
    server: str,
    probe: str,
    *,
    source: str | None = None,
    cell: int = 55,
) -> Any:
    native = source is None
    prelude = "<script>" + CLOCK + "</script>"
    if native:
        page = export_register_score_html(data()).replace(
            "<script>", prelude + "<script>", 1
        )
        initialize = "advance(750);"
    else:
        assert source is not None
        page = (
            '<html><body><div id="content"></div>'
            + prelude
            + "<script>const xDomain = [-2,5];const yDomain = [0,7];</script><script>"
            + source
            + "</script></body></html>"
        )
        initialize = (
            # Configure the source's mutable scale for native label spacing.
            f"xScale.range([0,{7 * cell}]);yScale.range([326,32]);draw("
            + json.dumps(data())
            + ");advance(750);"
        )
    code = (
        f"const native = {json.dumps(native)};\n"
        + SNAPSHOT
        + ASYNC_PROBE
        + initialize
        + "\n(async () => {try {\n"
        + probe
        + "\n} catch(error) {document.getElementById('result').textContent = "
        + "JSON.stringify({probeError: error.stack});}})();"
    )
    page = page.replace(
        "</html>", '<pre id="result"></pre><script>' + code + "</script></html>"
    )
    name = "loading-native.html" if native else "loading-source.html"
    (tmp_path / name).write_text(page, encoding="utf-8")
    result = subprocess.run(
        [
            browser,
            "--headless",
            "--disable-background-networking",
            "--no-first-run",
            "--virtual-time-budget=5000",
            f"--user-data-dir={tmp_path / ('native-profile' if native else 'source-profile')}",
            "--dump-dom",
            f"{server}/{name}",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    match = re.search(r'<pre id="result">(.*?)</pre>', result.stdout, re.S)
    assert match and match[1], result.stdout[-4000:] + result.stderr[-1000:]
    record = json.loads(html.unescape(match[1]))
    assert "probeError" not in record, record
    return record


@pytest.mark.parametrize("legacy", [False, True])
def test_http_load_matches_source_and_replaces_dependency_view(
    browser: str, source_script: str, tmp_path: Path, score_server: str, legacy: bool
) -> None:
    score = updated(data())
    if legacy:
        # The frozen browser asset also accepts pre-flavor, string-label exports.
        raw = json.loads(json.dumps(score))
        for line in raw["hlines"]:
            del line["flavor"]
        for line in raw["vlines"]:
            line["label"] = line["label"]["text"]
    else:
        raw = score
    (tmp_path / "score.json").write_text(json.dumps(raw))
    probe = """
if (native) {selector.value = 'dependency';selector.dispatchEvent(new Event('change'));advance(750);}
await load('score.json', null);
const logs = [];
console.log = (event, port) => logs.push({type: event.type, port});
for (const kind of ['TextBox', 'Text', 'RarrowTextBox', 'LarrowTextBox']) {
  const node = document.querySelector('g.' + kind);
  if (node) node.dispatchEvent(new Event('click'));
}
const control = document.querySelector(native ? 'g.Circle' : 'circle.circle');
const target = document.querySelector(native ? 'g.ModPlus' : 'g.modplus');
control.dispatchEvent(new Event('click'));target.dispatchEvent(new Event('click'));
document.getElementById('result').textContent = JSON.stringify({
  state: snapshot(), errors: errors(), logs,
  selector: native ? selector.value : 'sequential',
  disabled: native ? selector.querySelector('[value="dependency"]').disabled : true,
  stale: native ? 'dependency' in views : false
});
"""
    actual = run_loading_page(browser, tmp_path, score_server, probe)
    expected = run_loading_page(
        browser, tmp_path, score_server, probe, source=source_script
    )
    assert_snapshots([actual["state"]], [expected["state"]])
    assert actual["logs"] == expected["logs"]
    assert actual["logs"]
    assert actual["errors"] == expected["errors"] == []
    assert actual["selector"] == "sequential" and actual["disabled"]
    assert not actual["stale"]


@pytest.mark.parametrize("original_asset", [False, True])
def test_original_multi_and_json_loads_without_record_translation(
    browser: str,
    source_script: str,
    tmp_path: Path,
    score_server: str,
    original_asset: bool,
) -> None:
    from qualtran.bloqs.mcmt import MultiAnd
    from qualtran.drawing.musical_score import (
        MusicalScoreEncoder,
        get_musical_score_data,
    )

    original = get_musical_score_data(MultiAnd((1, 1, 0, 1)).decompose_bloq())
    encoded = json.dumps(original, cls=MusicalScoreEncoder)
    if original_asset:
        raw = (
            Path(__file__).resolve().parents[1] / "fixtures/register_score/example.json"
        ).read_bytes()
    else:
        raw = encoded.encode()
    records = json.loads(raw)
    (tmp_path / "original.json").write_bytes(raw)
    text = next(
        port["symb_attributes"]["text"]
        for port in records["soqs"]
        if port["symb_cls"] == "Text"
    )
    probe = "await load('original.json',null," + json.dumps(text) + ");\n"
    probe += """
document.getElementById('result').textContent = JSON.stringify({
  state: snapshot(), errors: errors(),
  records: native ? views.sequential : {
    soqs: [...DATA_G.node().querySelectorAll('circle.circle,g.modplus,g.TextBox,g.Text,g.LarrowTextBox,g.RarrowTextBox')].map(node => node.__data__)
  }
});
"""
    actual = run_loading_page(browser, tmp_path, score_server, probe)
    cell = max(
        55,
        *(
            len(port["symb_attributes"].get("text", "")) * 7 + 14
            for port in records["soqs"]
        ),
    )
    expected = run_loading_page(
        browser, tmp_path, score_server, probe, source=source_script, cell=cell
    )
    assert_snapshots([actual["state"]], [expected["state"]])
    assert actual["records"] == records
    assert actual["errors"] == expected["errors"] == []


@pytest.mark.parametrize("failure", ["missing", "malformed", "cors", "file"])
def test_http_failures_keep_geometry_and_match_source_error_panel(
    browser: str, source_script: str, tmp_path: Path, score_server: str, failure: str
) -> None:
    page_origin = score_server
    if failure == "malformed":
        (tmp_path / "broken.json").write_text('{"soqs":')
        filename = "broken.json"
    elif failure == "cors":
        (tmp_path / "score.json").write_text(json.dumps(updated(data())))
        filename = score_server.replace("127.0.0.1", "localhost") + "/score.json"
    elif failure == "file":
        (tmp_path / "score.json").write_text(json.dumps(updated(data())))
        filename = "score.json"
        page_origin = tmp_path.as_uri()
    else:
        filename = "missing'<img src=x onerror=window.injected=true>.json"
    probe = (
        "const before = snapshot();await load("
        + json.dumps(filename)
        + ",1);await load("
        + json.dumps(filename)
        + ",2);\n"
        + "document.getElementById('result').textContent = JSON.stringify({"
        + "before, after: snapshot(), errors: errors(), injected: window.injected === true});"
    )
    actual = run_loading_page(browser, tmp_path, page_origin, probe)
    expected = run_loading_page(
        browser, tmp_path, page_origin, probe, source=source_script
    )
    assert_snapshots([actual["before"]], [actual["after"]])
    assert actual["errors"] == expected["errors"]
    assert all(row["color"] == "rgb(255, 0, 0)" for row in actual["errors"])
    assert not actual["injected"] and not expected["injected"]


def test_invalid_import_is_refused_before_existing_views_change(
    browser: str, tmp_path: Path, score_server: str
) -> None:
    invalid: list[object] = [None, {"soqs": []}]
    value: object
    attrs: object
    for field, value in (("soqs", [None]), ("hlines", [None]), ("vlines", [None])):
        record = deepcopy(data())
        invalid.append({**record, field: value})
    wrong_text = deepcopy(data())
    wrong_text["soqs"][2]["symb_attributes"]["text"] = 3
    invalid.append(wrong_text)
    duplicate = deepcopy(data())
    duplicate["soqs"].append(deepcopy(duplicate["soqs"][0]))
    invalid.append(duplicate)
    for field, value in (("max_x", -1), ("max_y", "6"), ("hlines", {})):
        invalid.append({**data(), field: value})
    for field, value in (("symb_cls", "Unknown"), ("ident", 3), ("seq_x", "0")):
        raw = json.loads(json.dumps(data()))
        raw["soqs"][0][field] = value
        invalid.append(raw)
    for attrs in (
        None,
        [],
        {"filled": 1},
        {"fontsize": -1},
        {"fontsize": "12"},
        {"text": 1},
    ):
        raw = json.loads(json.dumps(data()))
        raw["soqs"][0]["symb_attributes"] = attrs
        invalid.append(raw)
    for field, value in (("label", None), ("label", {"text": 3}), ("top_y", "0")):
        raw = json.loads(json.dumps(data()))
        raw["vlines"][0][field] = value
        invalid.append(raw)
    for field in ("hlines", "vlines"):
        raw = json.loads(json.dumps(data()))
        raw[field].append(deepcopy(raw[field][0]))
        invalid.append(raw)
    for i, item in enumerate(invalid):
        (tmp_path / f"invalid{i}.json").write_text(json.dumps(item))
    probe = """
selector.value = 'dependency';selector.dispatchEvent(new Event('change'));advance(750);
const before = snapshot();
const previous = views.sequential;
"""
    for i in range(len(invalid)):
        probe += f"await load('invalid{i}.json', {i + 1});\n"
    probe += """
document.getElementById('result').textContent = JSON.stringify({
  before, after: snapshot(), errors: errors(), same: previous === views.sequential,
  selector: selector.value, disabled: selector.querySelector('[value="dependency"]').disabled
});
"""
    result = run_loading_page(browser, tmp_path, score_server, probe)
    assert_snapshots([result["before"]], [result["after"]])
    assert result["same"] and result["selector"] == "dependency"
    assert not result["disabled"]
    assert len(result["errors"]) == len(invalid)
    assert all(
        "TypeError: Invalid musical score" in row["message"] for row in result["errors"]
    )


def test_successful_retry_keeps_source_error_history_and_safe_label_text(
    browser: str, source_script: str, tmp_path: Path, score_server: str
) -> None:
    score = updated(data())
    label = "<img src=x onerror=window.injected=true>"
    score["soqs"][2]["symb_attributes"]["text"] = label
    (tmp_path / "retry.json").write_text(json.dumps(score))
    probe = "await load('missing.json',1);await load('retry.json',null);\n"
    probe += """
document.getElementById('result').textContent = JSON.stringify({
  state: snapshot(), errors: errors(), imageCount: document.querySelectorAll('img').length,
  injected: window.injected === true
});
"""
    actual = run_loading_page(browser, tmp_path, score_server, probe)
    expected = run_loading_page(
        browser, tmp_path, score_server, probe, source=source_script
    )
    # Native long-label spacing is an existing, separately recorded style boundary.
    assert [row["text"] for row in actual["state"]["glyphs"]] == [
        row["text"] for row in expected["state"]["glyphs"]
    ]
    assert label in [row["text"] for row in actual["state"]["glyphs"]]
    assert actual["errors"] == expected["errors"]
    assert actual["imageCount"] == expected["imageCount"] == 0
    assert not actual["injected"] and not expected["injected"]


def test_newest_request_owns_view_and_stale_failure_stays_silent(
    browser: str, tmp_path: Path, score_server: str
) -> None:
    second = updated(data())
    probe = """
const pending = new Map();
window.fetch = filename => new Promise((resolve, reject) => pending.set(filename, {resolve, reject}));
const stale = loadScore('old.json');
const fresh = loadScore('new.json');
"""
    probe += (
        "pending.get('new.json').resolve(new Response("
        + json.dumps(json.dumps(second))
        + "));await fresh;advance(750);const before = snapshot();\n"
    )
    probe += """
pending.get('old.json').reject(new Error('obsolete failure'));await stale;
const first = loadScore('first.json');const last = loadScore('last.json');
"""
    probe += (
        "pending.get('last.json').resolve(new Response("
        + json.dumps(json.dumps(second))
        + "));await last;advance(750);\n"
        + "pending.get('first.json').resolve(new Response("
        + json.dumps(json.dumps(data()))
        + "));await first;advance(750);\n"
    )
    probe += """
document.getElementById('result').textContent = JSON.stringify({before, after: snapshot(), errors: errors()});
"""
    result = run_loading_page(browser, tmp_path, score_server, probe)
    assert_snapshots([result["before"]], [result["after"]])
    assert result["errors"] == []
