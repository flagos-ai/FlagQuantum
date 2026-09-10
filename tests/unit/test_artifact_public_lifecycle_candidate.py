from __future__ import annotations

import json
from pathlib import Path

import pytest

import flagquantum

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "contracts" / "artifact-public-lifecycle-v1-candidate.json"


def _candidate() -> dict[str, object]:
    return json.loads(CANDIDATE.read_text(encoding="utf-8"))


def test_candidate_records_exact_approved_preview_surface() -> None:
    candidate = _candidate()
    assert candidate["status"] == "approved_experimental_read_only_complete"
    assert candidate["approved_on"] == "2026-09-10"
    assert candidate["approval_token"] == (
        "approve API_CHANGE_PROPOSAL_026_ARTIFACT_PUBLIC_LIFECYCLE"
    )
    assert candidate["initial_namespace"] == "flagquantum.experimental.artifacts"
    assert candidate["initial_lifecycle"] == "experimental_read_only"
    assert candidate["approved_symbols"] == [
        "ProgramArtifact",
        "CompilationEvidence",
        "load_program_artifact",
        "dump_program_artifact",
        "load_compilation_evidence",
        "dump_compilation_evidence",
    ]


def test_candidate_preserves_core_versions_and_stable_api() -> None:
    candidate = _candidate()
    assert candidate["concrete_version_classes_public"] is False
    assert candidate["returns_frozen_role_view"] is True
    assert candidate["core_dispatch_authoritative"] is True
    assert candidate["canonical_bytes_changed"] is False
    assert candidate["stable_root_exports_changed"] is False
    assert candidate["default_path_changed"] is False
    assert "artifacts" not in flagquantum.__all__


def test_candidate_records_completed_bounded_implementation() -> None:
    candidate = _candidate()
    assert candidate["implementation"] == {
        "authorized": True,
        "experimental_domain_added": True,
        "role_views_added": True,
        "loaders_added": True,
        "dumpers_added": True,
        "stable_exports_added": False,
    }
    assert "artifacts" in flagquantum.experimental.__all__
