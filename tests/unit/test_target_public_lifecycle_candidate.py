from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import flagquantum

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "contracts" / "target-public-lifecycle-v1-candidate.json"


def _candidate() -> dict[str, object]:
    return json.loads(CANDIDATE.read_text(encoding="utf-8"))


def test_candidate_records_exact_unapproved_target_preview() -> None:
    candidate = _candidate()
    assert candidate["status"] == "proposed_awaiting_owner_approval"
    assert candidate["approval_token"] == (
        "approve API_CHANGE_PROPOSAL_027_TARGET_PUBLIC_LIFECYCLE"
    )
    assert candidate["initial_namespace"] == "flagquantum.experimental.targets"
    assert candidate["approved_symbols"] == [
        "TargetSnapshot",
        "RequirementSet",
        "CapabilityMatch",
        "load_target_snapshot",
        "dump_target_snapshot",
        "load_requirement_set",
        "dump_requirement_set",
        "match_capabilities",
    ]


def test_candidate_requires_deterministic_bounded_read_only_matching() -> None:
    candidate = _candidate()
    assert candidate["returns_frozen_role_views"] is True
    assert candidate["core_values_public"] is False
    assert candidate["core_serialization_authoritative"] is True
    assert candidate["explicit_evaluation_time_required"] is True
    assert candidate["maximum_json_utf8_bytes"] == 16 * 1024 * 1024
    assert candidate["duplicate_keys_rejected"] is True
    assert candidate["fact_query_preserves_status_and_exposure"] is True
    assert candidate["match_binds_input_identities_and_time"] is True
    assert candidate["stable_root_exports_changed"] is False
    assert candidate["default_path_changed"] is False
    assert "targets" not in flagquantum.__all__


def test_candidate_is_not_implemented_before_owner_approval() -> None:
    candidate = _candidate()
    assert candidate["implementation"] == {
        "authorized": False,
        "experimental_domain_added": False,
        "role_views_added": False,
        "loaders_and_dumpers_added": False,
        "pure_matcher_added": False,
        "stable_exports_added": False,
    }
    assert importlib.util.find_spec("flagquantum.experimental.targets") is None
