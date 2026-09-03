from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "contracts/ir-phase3-exit-review-candidate.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_phase3_exit_candidate_binds_full_authorization_and_evidence_chain() -> None:
    candidate = _load(CANDIDATE)

    assert candidate["status"] == "ready_for_owner_review"
    for artifact in candidate["authorization_chain"].values():
        assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]
        assert _load(ROOT / artifact["path"])["status"] == "approved"
    for artifact in candidate["performance_validations"].values():
        assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]
        validation = _load(ROOT / artifact["path"])
        assert validation["status"] == "passed"
        assert all(case["passed"] for case in validation["cases"])
    for relative_path, expected_hash in candidate["implementation_artifacts"].items():
        assert _sha256(ROOT / relative_path) == expected_hash
    for relative_path, expected_hash in candidate["review_artifacts"].items():
        assert _sha256(ROOT / relative_path) == expected_hash
    for artifact in candidate["public_contract"].values():
        if isinstance(artifact, dict) and "path" in artifact:
            assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]


def test_phase3_exit_candidate_proves_three_shared_target_families() -> None:
    candidate = _load(CANDIDATE)
    boundary = candidate["three_target_family_boundary"]
    fixture = _load(ROOT / boundary["fixture"]["path"])

    assert _sha256(ROOT / boundary["fixture"]["path"]) == boundary["fixture"]["sha256"]
    assert {item["family"] for item in fixture["cases"]} == {
        "local_runtime_plan",
        "synthetic_qasm_text",
        "synthetic_non_qasm_artifact",
    }
    assert boundary["shared_artifact_runtime_boundary"] is True
    assert boundary["identical_ordered_conformance_suite"] is True
    assert boundary["real_provider_or_qpu_claimed"] is False


def test_phase3_exit_is_private_technical_completion_only() -> None:
    candidate = _load(CANDIDATE)
    decision = candidate["formal_exit_decision"]
    scope = candidate["accepted_scope_if_approved"]
    public = candidate["public_contract"]

    assert decision["technical_evidence_complete"] is True
    assert decision["phase3_complete"] is False
    assert decision["phase4_authorized"] is False
    assert all(
        decision[name] is False
        for name in (
            "api_owner_accepted",
            "compiler_owner_accepted",
            "runtime_owner_accepted",
            "training_owner_accepted",
        )
    )
    assert scope["implementation_visibility"] == "private internal"
    assert scope["public_exports"] == []
    assert scope["default_path_enabled"] is False
    assert scope["provider_or_remote_execution_enabled"] is False
    assert scope["canary_enabled"] is False
    assert scope["legacy_retired"] is False
    assert public["root_exports_added"] == []
    assert public["default_run_plan_path_changed"] is False
    assert public["deployment_package_schema_changed"] is False
    assert candidate["known_limitations"]
    assert candidate["next_review_command"] == "approve IR-PHASE3-EXIT"
