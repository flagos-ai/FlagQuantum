from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_compiler_optimize_user_example_runs_end_to_end() -> None:
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "-m", "examples.compiler_optimize"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "FlagQuantum compiler optimization check passed" in completed.stdout
    assert "instructions: 6 -> 3" in completed.stdout
    assert "execution path: local_statevector" in completed.stdout
