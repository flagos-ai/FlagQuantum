"""Run moved synthesis tutorials through their documented public module entries."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
PREMISE_PHRASES = {
    "source_trial": (
        "unreleased pinned Qmod subset",
        "source text and regular UTF-8 files share the checked Program and named reader",
    ),
    "trial": (
        "unreleased CPU synthesis profile",
        "finite checked candidates",
    ),
}


def _run(name: str, *args: str) -> str:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            (
                "tools.synthesis_import_examples.source_trial.main"
                if name == "source_trial"
                else f"examples.synthesis.tutorials.getting_started.{name}.main"
            ),
            *args,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )
    return completed.stdout


def _labelled(output: str, label: str) -> str:
    for line in output.splitlines():
        name, separator, value = line.partition(":")
        if separator and name.strip() == label:
            return value.strip()
    raise AssertionError(f"the example printed no {label!r} line:\n{output}")


def _assert_premise(name: str, output: str) -> None:
    paragraph = _labelled(output, "premise")
    for phrase in PREMISE_PHRASES[name]:
        assert (
            phrase in paragraph
        ), f"{name} lost its premise phrase {phrase!r}: {paragraph}"


def test_synthesis_trial_runs_basis_and_coherent_named_workflows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)
    output = _run("trial")
    basis = ast.literal_eval(_labelled(output, "basis"))
    uniform = ast.literal_eval(_labelled(output, "uniform"))
    assert basis == pytest.approx({(3, 2, 11): 1.0}, abs=1e-10)
    expected = {(x, y, x * x + y): 1 / 16 for x in range(4) for y in range(4)}
    assert uniform == pytest.approx(expected, abs=1e-10)
    resources = _labelled(output, "logical resources")
    assert "n_qubits=16" in resources
    assert "n_operations=81" in resources
    assert "depth=59" in resources
    _assert_premise("trial", output)


@pytest.mark.parametrize("file_input", [False, True])
def test_synthesis_source_trial_runs_text_and_local_file(
    tmp_path: Path, file_input: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)
    if file_input:
        source = tmp_path / "example.qmod"
        source.write_text(
            "qfunc main(output q:qbit) { allocate(q); X(q); }", encoding="utf-8"
        )
        output = _run("source_trial", str(source))
        expected = {(1,): 1.0}
        qubits, operations, depth = 1, 1, 1
    else:
        output = _run("source_trial")
        expected = {(x, x * x + 1): 0.25 for x in range(4)}
        qubits, operations, depth = 6, 8, 5
    observed = ast.literal_eval(_labelled(output, "distribution"))
    assert observed == pytest.approx(expected, abs=1e-12)
    resources = _labelled(output, "resources")
    assert f"n_qubits={qubits}," in resources
    assert f"n_operations={operations}," in resources
    assert f"depth={depth}," in resources
    _assert_premise("source_trial", output)
