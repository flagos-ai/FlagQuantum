"""A failed diagnostic case must not hide the rest of the frozen study."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from benchmarks import compiler_sabre_depth as study

pytestmark = pytest.mark.unit


def test_study_retains_failure_attempts_remaining_cases_and_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "study.json"
    original = study._generic
    attempted: list[tuple[int, str, int]] = []

    def observe(width: int, topology: str, seed: int) -> dict[str, object]:
        attempted.append((width, topology, seed))
        if (width, topology, seed) == (6, "line", 0):
            raise RuntimeError("Injected diagnostic case failure")
        return original(width, topology, seed)

    monkeypatch.setattr(study, "_generic", observe)
    monkeypatch.setattr(
        sys, "argv", ["study", "--suite", "generic", "--output", str(output)]
    )
    with pytest.raises(SystemExit) as error:
        study.main()
    assert error.value.code == 1
    report = json.loads(output.read_text())
    assert len(attempted) == len(set(attempted)) == 80
    assert report["completed"] is True
    assert len(report["rows"]) == 79
    assert report["failures"] == [
        {
            "case": [6, "line", 0],
            "type": "RuntimeError",
            "message": "Injected diagnostic case failure",
        }
    ]
