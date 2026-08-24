from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools" / "validate_flagos_reference_evidence.py"
RUNTIME_VALIDATOR_PATH = ROOT / "tools" / "validate_flagos_cuda_reference.py"
EVIDENCE_PATH = ROOT / "artifacts" / "flagos_cuda_reference_a800_20260824.json"
SPEC = importlib.util.spec_from_file_location("flagos_reference_evidence", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
RUNTIME_SPEC = importlib.util.spec_from_file_location(
    "flagos_cuda_reference", RUNTIME_VALIDATOR_PATH
)
assert RUNTIME_SPEC is not None and RUNTIME_SPEC.loader is not None
RUNTIME_VALIDATOR = importlib.util.module_from_spec(RUNTIME_SPEC)
RUNTIME_SPEC.loader.exec_module(RUNTIME_VALIDATOR)


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

    payload = _evidence()
    payload["source"]["tree_dirty"] = True
    assert "source_tree_dirty" in MODULE.evidence_errors(payload)


@pytest.mark.parametrize(
    "field",
    (
        "container_image",
        "python",
        "torch_package",
        "torch_distribution",
        "torch_cuda_runtime",
        "torch_fl_package",
        "torch_fl_commit",
        "torch_fl_build",
    ),
)
def test_reference_gate_rejects_environment_identity_drift(field: str) -> None:
    payload = _evidence()
    payload["environment"][field] = "drifted"
    assert f"environment_{field}_mismatch" in MODULE.evidence_errors(payload)


def test_strict_runtime_identity_matches_v2_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = json.loads(
        (ROOT / "ci" / "flagos_cuda_reference.lock.json").read_text(encoding="utf-8")
    )
    monkeypatch.setattr(
        RUNTIME_VALIDATOR.platform, "python_version", lambda: lock["python"]
    )
    monkeypatch.setenv("FLAGQUANTUM_FLAGOS_CONTAINER_IMAGE", lock["container_image"])
    monkeypatch.setenv("FLAGQUANTUM_TORCH_FL_COMMIT", lock["torch_fl"]["commit"])
    monkeypatch.setenv(
        "FLAGQUANTUM_TORCH_FL_BUILD", json.dumps(lock["torch_fl"]["build"])
    )
    monkeypatch.setenv("FLAGQUANTUM_ACCELERATOR_MODEL", "reference accelerator")
    payload = {
        "torch_version": lock["torch"]["package"],
        "torch_distribution": lock["torch"]["distribution"],
        "torch_cuda_runtime": lock["torch"]["cuda_runtime"],
        "torch_fl_package": lock["torch_fl"]["package"],
        "device_count": lock["runtime_contract"]["visible_device_count"],
    }
    RUNTIME_VALIDATOR._require_locked_environment(
        environment_lock=lock, payload=payload
    )

    payload["torch_version"] = "2.10.0+cpu"
    with pytest.raises(RuntimeError, match="environment lock mismatch"):
        RUNTIME_VALIDATOR._require_locked_environment(
            environment_lock=lock, payload=payload
        )
