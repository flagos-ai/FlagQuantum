from __future__ import annotations

import json
from pathlib import Path

import pytest

import flagquantum as fq

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs" / "public_api_v1.json"
CANDIDATE = ROOT / "contracts" / "public-api-v1-candidate.json"
BASELINE = ROOT / "contracts" / "public-api-v0.2-baseline.json"
EXECUTION_OPTIONS = ROOT / "contracts" / "execution-options-v1-candidate.json"


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


def test_candidate_classifies_every_historical_stable_export_exactly_once() -> None:
    baseline = _load(BASELINE)
    candidate = _load(CANDIDATE)
    exports = baseline["exports"]
    assert isinstance(exports, dict)

    classified = _classified_symbols(candidate)

    assert len(classified) == len(set(classified))
    options_contract = _load(EXECUTION_OPTIONS)
    authorized_additions = (
        {options_contract["root_addition"]}
        if options_contract["root_manifest_authorized"] is True
        else set()
    )
    assert set(classified) == set(exports) | authorized_additions


def test_current_manifest_is_the_implemented_stable_core() -> None:
    manifest = _load(MANIFEST)
    candidate = _load(CANDIDATE)
    stable_core = candidate["stable_core"]
    assert isinstance(stable_core, dict)

    assert set(manifest["stable_exports"]) == set(stable_core["retain"])
    assert not set(stable_core["planned_additions"]) & set(manifest["stable_exports"])
    assert set(fq.__all__) == set(manifest["stable_exports"])


def test_migrated_exports_are_not_discoverable_at_root() -> None:
    candidate = _load(CANDIDATE)
    migrated = {
        symbol
        for section_name in ("stable_extensions", "experimental")
        for section in candidate[section_name]
        for symbol in section["symbols"]
    }
    migrated.update(candidate["remove_before_public"])

    assert migrated.isdisjoint(dir(fq))


def test_migrated_and_removed_exports_are_not_accessible_at_root() -> None:
    candidate = _load(CANDIDATE)
    replacements = {
        symbol
        for section_name in ("stable_extensions", "experimental")
        for section in candidate[section_name]
        for symbol in section["symbols"]
    }
    removals = set(candidate["remove_before_public"])

    for name in sorted(replacements):
        with pytest.raises(AttributeError, match="moved before the first public alpha"):
            getattr(fq, name)
    for name in sorted(removals):
        with pytest.raises(
            AttributeError, match="removed before the first public alpha"
        ):
            getattr(fq, name)


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
