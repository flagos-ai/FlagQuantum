from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs" / "public_api_v1.json"
CANDIDATE = ROOT / "contracts" / "public-api-v1-candidate.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _classified_symbols(candidate: dict[str, object]) -> list[str]:
    stable_core = candidate["stable_core"]
    assert isinstance(stable_core, dict)
    symbols = list(stable_core["retain"])
    for section_name in ("stable_extensions", "experimental"):
        sections = candidate[section_name]
        assert isinstance(sections, list)
        for section in sections:
            assert isinstance(section, dict)
            symbols.extend(section["symbols"])
    removals = candidate["remove_before_public"]
    assert isinstance(removals, list)
    symbols.extend(removals)
    return symbols


def test_candidate_classifies_every_current_stable_export_exactly_once() -> None:
    manifest = _load(MANIFEST)
    candidate = _load(CANDIDATE)
    expected = manifest["stable_exports"]
    assert isinstance(expected, list)

    classified = _classified_symbols(candidate)

    assert len(classified) == len(set(classified))
    assert set(classified) == set(expected)


def test_candidate_stable_core_stays_within_reviewed_root_budget() -> None:
    candidate = _load(CANDIDATE)
    stable_core = candidate["stable_core"]
    rules = candidate["rules"]
    assert isinstance(stable_core, dict)
    assert isinstance(rules, dict)

    final_core = set(stable_core["retain"]) | set(stable_core["planned_additions"])

    assert len(final_core) == 22
    assert len(final_core) <= rules["root_export_budget"]
    assert {"Circuit", "Module", "ExecutionOptions", "ExecutionPlan"} <= final_core
    assert {"plan", "run", "train", "ExecutionResult", "TrainingResult"} <= final_core


def test_candidate_records_approval_but_is_not_yet_frozen() -> None:
    candidate = _load(CANDIDATE)
    rules = candidate["rules"]
    approval = candidate["approval"]
    assert isinstance(rules, dict)
    assert isinstance(approval, dict)

    assert candidate["status"] == "approved"
    assert approval["scope"] == "Stable Core disposition and namespace migration"
    assert rules["breaking_changes_require_approved_proposal"] is True
    assert rules["candidate_is_frozen_contract"] is False
