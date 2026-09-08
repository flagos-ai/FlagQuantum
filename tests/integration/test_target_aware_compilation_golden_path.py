from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_target_aware_compilation_user_example_runs_end_to_end() -> None:
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "-m", "examples.target_aware_compilation"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "FlagQuantum target-aware compilation check passed" in completed.stdout
    assert "target edges: valid" in completed.stdout
    assert "mapping restored: True" in completed.stdout
    assert "execution path: local_statevector" in completed.stdout
