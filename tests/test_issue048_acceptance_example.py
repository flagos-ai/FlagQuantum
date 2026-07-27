from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_maintained_hybrid_acceptance_source_script() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples/hybrid_model_acceptance.py"),
            "--model",
            "classifier",
            "--steps",
            "2",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["model"] == "HybridQuantumClassifier"
    assert payload["accuracy_is_performance_claim"] is False
    assert set(payload) >= {"correctness", "accuracy", "performance", "deployment"}
