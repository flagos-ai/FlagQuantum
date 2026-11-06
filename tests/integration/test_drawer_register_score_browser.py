"""Open an exported score in Chromium and exercise its offline view control."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from flagquantum.drawer.register_score import register_program_score
from flagquantum.drawer.register_score_data import register_score_data
from flagquantum.drawer.register_score_html import export_register_score_html
from tests.register_program_cases import Registers

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("legacy", [False, True])
def test_offline_register_score_switches_columns(tmp_path: Path, legacy: bool) -> None:
    chromium = os.environ.get("FLAGQUANTUM_CHROME") or shutil.which("chromium")
    if chromium is None:
        pytest.skip("Set FLAGQUANTUM_CHROME to test with a local Chromium browser")

    registers = Registers()
    control = registers.declare("ctrl", 1)
    target = registers.declare("target", 1)
    registers.emit("qreg.allocate", control)
    registers.emit("qreg.allocate", target)
    registers.emit("qreg.pair_gate", control, target, gate="cx")
    data = register_score_data(register_program_score(registers.finish()))
    if legacy:
        # The locked browser fixture predates track flavors and structured labels.
        for horizontal in data["hlines"]:
            del horizontal["flavor"]  # type: ignore[misc]
        for vertical in data["vlines"]:
            vertical["label"] = vertical["label"]["text"]  # type: ignore[typeddict-item]
    page = tmp_path / "score.html"
    probe = """<script>
const cols = () => [...document.querySelectorAll("g.TextBox")].map(
    port => port.dataset.column
);
const initial = cols();
const select = document.getElementById("score-view");
select.value = "dependency";
select.dispatchEvent(new Event("change"));
const projected = cols();
const control = document.querySelector("g.Circle");
const target = document.querySelector("g.ModPlus");
const sameColumn = control.dataset.column === target.dataset.column;
const distinctLanes = control.dataset.lane !== target.dataset.lane;
select.value = "sequential";
select.dispatchEvent(new Event("change"));
if (initial.join() === "0,1" && projected.join() === "1,1" &&
    sameColumn && distinctLanes && cols().join() === initial.join()) {
    document.body.dataset.browserResult = "passed";
}
</script>"""
    page.write_text(
        export_register_score_html(data).replace("</html>", probe + "</html>"),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            chromium,
            "--headless",
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
    assert 'data-browser-result="passed"' in result.stdout, result.stderr
