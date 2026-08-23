from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools" / "validate_flagos_reference_evidence.py"
EVIDENCE_PATH = ROOT / "artifacts" / "flagos_cuda_reference_a100_20260821.json"
SPEC = importlib.util.spec_from_file_location("flagos_reference_evidence", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _evidence() -> dict:
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


def test_checked_in_flagos_reference_evidence_passes_development_gate() -> None:
    assert MODULE.evidence_errors(_evidence()) == ()


@pytest.mark.parametrize(
    ("field", "unsafe_value", "expected_error"),
    (
        ("hardware_certification", True, "hardware_certification_mismatch"),
        ("scalability_claim_allowed", True, "scalability_claim_allowed_mismatch"),
        ("artifact_class", "release_certified", "artifact_class_mismatch"),
    ),
)
def test_reference_gate_rejects_claim_promotion_without_hardware_evidence(
    field: str, unsafe_value: object, expected_error: str
) -> None:
    payload = _evidence()
    payload[field] = unsafe_value
    assert expected_error in MODULE.evidence_errors(payload)


def test_reference_gate_rejects_missing_dtype_depth_or_bad_numerics() -> None:
    payload = _evidence()
    payload["validation_matrix"] = payload["validation_matrix"][:-1]
    assert "validation_matrix_coverage_mismatch" in MODULE.evidence_errors(payload)

    payload = deepcopy(_evidence())
    payload["validation_matrix"][0]["state_metrics"]["max_abs_error"] = 1.0
    assert "max_abs_error_out_of_bounds:complex64:8" in MODULE.evidence_errors(payload)


def test_reference_gate_requires_auditable_source_identity() -> None:
    payload = _evidence()
    payload["source"] = {"revision": "unavailable", "tree_dirty": None}
    assert "source_revision_missing" in MODULE.evidence_errors(payload)
