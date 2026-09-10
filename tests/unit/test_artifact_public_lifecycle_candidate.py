from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import flagquantum

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "contracts" / "artifact-public-lifecycle-v1-candidate.json"


def _candidate() -> dict[str, object]:
    return json.loads(CANDIDATE.read_text(encoding="utf-8"))


def test_candidate_records_exact_unapproved_preview_surface() -> None:
    candidate = _candidate()
    assert candidate["status"] == "proposed_awaiting_owner_approval"
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


def test_candidate_is_not_implemented_before_owner_approval() -> None:
    candidate = _candidate()
    assert candidate["implementation"] == {
        "authorized": False,
        "experimental_domain_added": False,
        "role_views_added": False,
        "loaders_added": False,
        "dumpers_added": False,
        "stable_exports_added": False,
    }
    assert importlib.util.find_spec("flagquantum.experimental.artifacts") is None
