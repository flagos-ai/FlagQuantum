"""Run the frozen renderer's SVG controls in a real browser."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from flagquantum.drawer.flame_graph import iter_flame_graph_data, render_flame_graph_svg
from tests.unit.test_drawer_flame_svg import REFERENCE_SCRIPT

pytestmark = [pytest.mark.integration, pytest.mark.ftqc_reference]

PROBE = r"""
<script>
window.addEventListener("load", () => {
    const nodes = [...document.querySelectorAll("#frames > g")];
    const frame = name => nodes.find(node =>
        node.querySelector("title").textContent.startsWith(name + "(T:"));
    const left = frame("Left");
    const right = frame("Right");
    const rect = node => node.querySelector("rect");
    const width = node => Number(rect(node).getAttribute("width"));
    const click = node => node.dispatchEvent(new MouseEvent("click", {bubbles: true}));
    const before = width(left);
    click(rect(left));
    const zoomed = width(left) > before && right.classList.contains("hide");
    click(document.querySelector("#unzoom"));
    const reset = width(left) === before && !right.classList.contains("hide");

    const fill = rect(left).getAttribute("fill");
    const search = document.querySelector("#search");
    const match = document.querySelector("#matched");
    window.prompt = () => "^Left[(]";
    click(search);
    const found = match.textContent === "Matched: 70.0%" &&
        !match.classList.contains("hide") && fill !== rect(left).getAttribute("fill");
    click(search);
    const cleared = fill === rect(left).getAttribute("fill") &&
        match.classList.contains("hide");

    document.body.dataset.flameResult = [zoomed, reset, found, cleared].join(",");
});
</script>
"""


def test_flame_graph_zoom_reset_search_and_clear(tmp_path: Path) -> None:
    chromium = os.environ.get("FLAGQUANTUM_CHROME") or shutil.which("chromium")
    if chromium is None:
        pytest.skip("Set FLAGQUANTUM_CHROME to run the local browser check")

    rows = iter_flame_graph_data(
        "Root",
        {("Root", "Left"): 1, ("Root", "Right"): 1},
        {"Root": 10, "Left": 7, "Right": 3},
    )
    svg = render_flame_graph_svg(rows, executable=str(REFERENCE_SCRIPT)).decode("utf-8")
    # Inline the SVG element so the test script can dispatch clicks into its DOM.
    page = tmp_path / "flame.html"
    page.write_text(
        "<!doctype html><meta charset='utf-8'><body>"
        + svg[svg.index("<svg") :]
        + PROBE
        + "</body>",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            chromium,
            "--headless",
            "--disable-gpu",
            "--disable-background-networking",
            "--no-first-run",
            f"--user-data-dir={tmp_path / 'profile'}",
            "--dump-dom",
            page.as_uri(),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert 'data-flame-result="true,true,true,true"' in result.stdout, result.stderr
