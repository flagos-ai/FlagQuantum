"""Complete source diagnostic patterns exercised with actual tool output."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.unit.test_development_tools import REFERENCE

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / ".github/problem-matchers"
SOURCE = REFERENCE / ".github/problem-matchers"


def test_complete_source_declarations_and_provenance() -> None:
    for name, blob in {
        "README.md": "ac82dcb67ab3a747bd023f012d074aec62dae253",
        "black.json": "605f10c2706c5ae64fb19142c88f9a91eb9892e6",
        "mypy.json": "f048fce528941f8ad5a5f9d453f1d690c7204115",
        "pylint.json": "5624ca695c4e0a117f65054d653d822c17897b25",
    }.items():
        raw = (SOURCE / name).read_bytes()
        assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob
        if name.endswith(".json"):
            assert (NATIVE / name).read_bytes() == raw


def captures(config: Path, output: str) -> list[dict[str, str | int]]:
    node = shutil.which("node")
    assert node is not None, "JavaScript matcher execution is required"
    code = """
const fs = require('fs');
const rules = JSON.parse(fs.readFileSync(process.argv[1], 'utf8')).problemMatcher;
const lines = fs.readFileSync(0, 'utf8').split(/\\r?\\n/);
const observations = [];
for (const rule of rules) {
  for (const line of lines) {
    const pattern = rule.pattern[0];
    const match = new RegExp(pattern.regexp).exec(line);
    if (!match) continue;
    const result = {owner: rule.owner};
    if (rule.severity) result.severity = rule.severity;
    for (const key of ['file', 'line', 'column', 'severity', 'message', 'code']) {
      if (pattern[key]) result[key] = match[pattern[key]];
    }
    observations.push(result);
  }
}
process.stdout.write(JSON.stringify(observations));
"""
    result = subprocess.run(
        [node, "-e", code, str(config)],
        input=output,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    rows: list[dict[str, str | int]] = json.loads(result.stdout)
    return rows


def tool(root: Path, module: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REFERENCE / "g2_execution_python.py"),
            "-m",
            module,
            *arguments,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=90,
    )


@pytest.mark.parametrize("module", ["black", "mypy", "pylint"])
def test_actual_tool_failure_diagnostics_keep_original_captures(
    module: str, tmp_path: Path
) -> None:
    source = tmp_path / "bad input.py"
    if module == "black":
        source.write_text("value=  [1,2]\n")
        arguments = ["--check", "--no-color", str(source)]
    elif module == "mypy":
        source.write_text('value: int = "wrong"\n')
        arguments = [
            "--no-incremental",
            "--no-pretty",
            "--no-color-output",
            str(source),
        ]
    else:
        source.write_text("import os\nanswer = missing\n")
        arguments = [
            "--reports=n",
            "--score=n",
            "--disable=all",
            "--enable=E0602,W0611,C0114",
            str(source),
        ]
    result = tool(tmp_path, module, *arguments)
    assert result.returncode != 0
    output = result.stdout + result.stderr
    original = captures(SOURCE / (module + ".json"), output)
    native = captures(NATIVE / (module + ".json"), output)
    assert native == original and original
    assert all(str(row["file"]).endswith("bad input.py") for row in native)
    if module == "black":
        assert native == [
            {
                "owner": "black",
                "severity": "error",
                "file": str(source),
                "message": "would reformat",
            }
        ]
    elif module == "mypy":
        assert native[0]["line"] == "1" and native[0]["severity"] == "error"
        assert "Incompatible types" in str(native[0]["message"])
    else:
        assert {row["code"] for row in native} == {"E0602", "W0611", "C0114"}
        assert {row["severity"] for row in native} == {"error", "warning"}


@pytest.mark.parametrize("name", ["black", "mypy", "pylint"])
def test_summary_and_note_lines_are_not_diagnostics(name: str) -> None:
    assert (
        captures(
            NATIVE / (name + ".json"),
            "All done!\nSuccess: no issues found\na.py:2: note: explanation\n",
        )
        == []
    )


def test_original_pylint_all_categories_and_position_fields() -> None:
    output = "\n".join(
        f"source/file.py:12:3: {code}: diagnostic"
        for code in ["E0602", "F0001", "C0114", "R1705", "W0611"]
    )
    original = captures(SOURCE / "pylint.json", output)
    assert captures(NATIVE / "pylint.json", output) == original
    assert len(original) == 5
    for row in original:
        assert row["line"] == "12" and row["column"] == "3"
        assert row["message"] == str(row["code"]) + ": diagnostic"


def test_native_ruff_annotations_and_exit_status(tmp_path: Path) -> None:
    source = tmp_path / "bad.py"
    source.write_text("import os\n")
    result = tool(
        tmp_path,
        "ruff",
        "check",
        "--isolated",
        "--output-format",
        "github",
        str(source),
    )
    assert result.returncode == 1
    assert "::error" in result.stdout
    assert "F401" in result.stdout and "file=" in result.stdout


def test_existing_quality_job_registers_declared_matchers_before_consumers() -> None:
    workflow: dict[str, Any] = yaml.safe_load(
        (ROOT / ".github/workflows/ci.yml").read_text()
    )
    steps = workflow["jobs"]["quality"]["steps"]
    registration = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Register quality diagnostic matchers"
    )
    assert "actions/checkout@" in str(steps[:registration])
    for name in ("black", "mypy", "pylint"):
        assert (
            "::add-matcher::.github/problem-matchers/" + name + ".json"
            in steps[registration]["run"]
        )
    runs = [step.get("run", "") for step in steps[registration + 1 :]]
    assert "black --check flagquantum tests tools" in runs
    assert "ruff check --output-format github flagquantum tests tools" in runs
    assert any("mypy --strict" in run for run in runs)
