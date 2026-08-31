from __future__ import annotations

import json
from pathlib import Path

import pytest

import flagquantum as fq

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "contracts" / "execution-plan-v1-candidate.json"
PUBLIC_CANDIDATE = ROOT / "contracts" / "public-api-v1-candidate.json"
MANIFEST = ROOT / "docs" / "public_api_v1.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_execution_plan_candidate_is_draft_without_implementation_authority() -> None:
    candidate = _load(CANDIDATE)

    assert candidate["status"] == "draft_pending_approval"
    assert candidate["implementation_authorized"] is False
    assert candidate["root_manifest_authorized"] is False
    assert candidate["rules"]["candidate_is_frozen_contract"] is False


def test_execution_plan_remains_a_planned_root_addition() -> None:
    candidate = _load(CANDIDATE)
    public_candidate = _load(PUBLIC_CANDIDATE)
    manifest = _load(MANIFEST)

    assert candidate["root_addition"] == "ExecutionPlan"
    assert "ExecutionPlan" in public_candidate["stable_core"]["planned_additions"]
    assert "ExecutionPlan" not in manifest["stable_exports"]
    assert "ExecutionPlan" not in fq.__all__


def test_identity_contract_includes_semantics_but_excludes_volatile_state() -> None:
    candidate = _load(CANDIDATE)
    inputs = set(candidate["identity_inputs"])
    exclusions = set(candidate["identity_exclusions"])

    assert {
        "program_fingerprint",
        "resolved_execution_semantics",
        "compiler_fingerprint",
        "required_environment_constraints",
        "selected_execution_decision",
    } <= inputs
    assert {
        "created_at",
        "hostname",
        "pid",
        "rank_id",
        "credentials",
        "python_repr",
    } <= exclusions
    assert inputs.isdisjoint(exclusions)


def test_plan_input_is_closed_to_all_semantic_overrides() -> None:
    rules = _load(CANDIDATE)["plan_input_rules"]

    assert rules == {
        "options_must_be_none": True,
        "measurements_must_be_none": True,
        "noise_model_must_be_none": True,
        "replanning_forbidden": True,
        "silent_fallback_forbidden": True,
        "result_plan_is_input_plan_in_process": True,
    }


def test_serialization_contract_is_json_and_rejects_unverified_payloads() -> None:
    serialization = _load(CANDIDATE)["serialization"]

    assert serialization["format"] == "canonical_json"
    assert serialization["pickle_allowed"] is False
    assert serialization["unknown_top_level_fields_fail"] is True
    assert serialization["identity_recomputed_on_read"] is True
    assert set(serialization["fingerprint_fields"]) == {
        "program",
        "options",
        "environment",
        "compiler",
    }


def test_execution_plan_is_local_executable_not_a_deployment_package() -> None:
    boundaries = _load(CANDIDATE)["boundaries"]

    assert boundaries["local_executable"] is True
    assert boundaries["cross_process_restore"] is True
    assert boundaries["deployment_package"] is False
    assert boundaries["provider_credentials_allowed"] is False
    assert boundaries["live_runtime_objects_allowed"] is False
