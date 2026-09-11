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
EXECUTION_PLAN = ROOT / "contracts" / "execution-plan-v1-candidate.json"
MODULE_TRAINING = ROOT / "contracts" / "module-training-v1-candidate.json"
ERRORS_MODULE = ROOT / "contracts" / "errors-module-boundary-v1-candidate.json"
EXTENSION_PROTOCOL = ROOT / "contracts" / "extension-protocol-v1-candidate.json"
REMOTE_JOBS = ROOT / "contracts" / "remote-jobs-v1-candidate.json"
OBSERVABLE_OUTPUTS = ROOT / "contracts" / "observable-outputs-v1-candidate.json"


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
    plan_contract = _load(EXECUTION_PLAN)
    if plan_contract["root_manifest_authorized"] is True:
        authorized_additions.add(plan_contract["root_addition"])
    module_training_contract = _load(MODULE_TRAINING)
    stable_extension = module_training_contract["stable_extension"]
    assert isinstance(stable_extension, dict)
    if module_training_contract["implementation_authorized"] is True:
        authorized_additions.update(stable_extension["additions"])
    errors_module_contract = _load(ERRORS_MODULE)
    errors_extension = errors_module_contract["stable_extension"]
    assert isinstance(errors_extension, dict)
    if errors_module_contract["implementation_authorized"] is True:
        authorized_additions.update(errors_extension["additions"])
    pre_public_renames = candidate["pre_public_renames"]
    assert isinstance(pre_public_renames, dict)
    authorized_additions.update(
        replacement.rsplit(".", 1)[-1] for replacement in pre_public_renames.values()
    )
    extension_contract = _load(EXTENSION_PROTOCOL)
    if extension_contract["implementation_authorized"] is True:
        if extension_contract["root_manifest_change"] is True:
            authorized_additions.update(extension_contract["root_additions"])
        for extension in extension_contract["stable_extensions"]:
            authorized_additions.update(extension["additions"])
    observable_outputs = _load(OBSERVABLE_OUTPUTS)
    authorized_additions.update(observable_outputs["root_additions"])
    jobs_contract = _load(REMOTE_JOBS)
    if jobs_contract["implementation_authorized"] is True:
        authorized_additions.update(jobs_contract["root_additions"])
    authorized_additions.update(candidate.get("approved_namespace_additions", ()))
    assert set(classified) == set(exports) | authorized_additions


def test_pre_public_renames_have_one_stable_destination() -> None:
    candidate = _load(CANDIDATE)
    pre_public_renames = candidate["pre_public_renames"]
    assert isinstance(pre_public_renames, dict)
    stable_extensions = {
        f"{section['namespace']}.{symbol}"
        for section in candidate["stable_extensions"]
        for symbol in section["symbols"]
    }

    assert set(pre_public_renames) <= set(candidate["remove_before_public"])
    root_exports = {
        f"flagquantum.{symbol}" for symbol in candidate["stable_core"]["retain"]
    }
    assert set(pre_public_renames.values()) <= stable_extensions | root_exports


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


def test_non_root_exports_are_not_accessible_at_root() -> None:
    candidate = _load(CANDIDATE)
    replacements = {
        symbol
        for section_name in ("stable_extensions", "experimental")
        for section in candidate[section_name]
        for symbol in section["symbols"]
    }
    removals = set(candidate["remove_before_public"])
    module_training_contract = _load(MODULE_TRAINING)
    stable_extension = module_training_contract["stable_extension"]
    assert isinstance(stable_extension, dict)
    new_namespace_only = set(stable_extension["new_namespace_only"])
    errors_module_contract = _load(ERRORS_MODULE)
    errors_extension = errors_module_contract["stable_extension"]
    assert isinstance(errors_extension, dict)
    new_namespace_only.update(errors_extension["new_namespace_only"])
    extension_contract = _load(EXTENSION_PROTOCOL)
    for extension in extension_contract["stable_extensions"]:
        new_namespace_only.update(extension["new_namespace_only"])
    pre_public_renames = candidate["pre_public_renames"]
    assert isinstance(pre_public_renames, dict)
    new_namespace_only.update(
        replacement.rsplit(".", 1)[-1] for replacement in pre_public_renames.values()
    )
    new_namespace_only.difference_update(candidate["stable_core"]["retain"])

    for name in sorted(replacements | removals | new_namespace_only):
        with pytest.raises(AttributeError):
            getattr(fq, name)


def test_candidate_stable_core_stays_within_reviewed_root_budget() -> None:
    candidate = _load(CANDIDATE)
    stable_core = candidate["stable_core"]
    rules = candidate["rules"]
    assert isinstance(stable_core, dict)
    assert isinstance(rules, dict)

    final_core = set(stable_core["retain"]) | set(stable_core["planned_additions"])

    jobs_contract = _load(REMOTE_JOBS)
    assert jobs_contract["implementation_authorized"] is True
    assert set(jobs_contract["root_additions"]) == {"submit", "restore_job"}
    assert set(jobs_contract["root_additions"]) <= final_core
    assert len(final_core) == 33
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
