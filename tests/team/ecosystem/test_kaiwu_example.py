from __future__ import annotations

import runpy
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]


def test_kaiwu_matrix_boundary_example_runs_without_remote_access(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runpy.run_path(
        str(ROOT / "examples" / "kaiwu_matrix_boundary.py"),
        run_name="__main__",
    )

    output = capsys.readouterr().out
    assert "FlagQuantum Kaiwu matrix boundary check passed" in output
    assert "binary solution: [0, 1]" in output
    assert "QUBO energy: -1.250000" in output
    assert "encoded Ising energy: -1.250000" in output
    assert "remote submission: not performed" in output
