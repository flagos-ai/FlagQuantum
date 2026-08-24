from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools" / "validate_split_real_imag_training_evidence.py"
SPEC = importlib.util.spec_from_file_location("split_p1_evidence", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _payload() -> dict:
    return json.loads(
        (ROOT / "artifacts/split_real_imag_training_a800_20260824.json").read_text()
    )


def test_checked_in_split_p1_a800_evidence_passes() -> None:
    assert MODULE.evidence_errors(_payload()) == ()


def test_split_p1_a800_evidence_rejects_promotion_and_numerical_drift() -> None:
    payload = copy.deepcopy(_payload())
    payload["claims"]["hardware_certification"] = True
    payload["cases"][0]["gradient_relative_error"] = 1.0
    errors = MODULE.evidence_errors(payload)
    assert any("forbidden promotion" in error for error in errors)
    assert any("gradient error" in error for error in errors)
