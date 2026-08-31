from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import public_api_snapshot

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_public_api_matches_reviewed_retained_baseline() -> None:
    assert public_api_snapshot.validate() == ()


def test_baseline_covers_current_stable_export_manifest() -> None:
    manifest = json.loads((ROOT / "docs/public_api_v1.json").read_text())
    baseline = json.loads(
        (ROOT / "contracts/public-api-v0.2-baseline.json").read_text()
    )
    options = json.loads(
        (ROOT / "contracts/execution-options-v1-candidate.json").read_text()
    )
    plan = json.loads((ROOT / "contracts/execution-plan-v1-candidate.json").read_text())
    authorized_additions = (
        {options["root_addition"]}
        if options["root_manifest_authorized"] is True
        else set()
    )
    if plan["root_manifest_authorized"] is True:
        authorized_additions.add(plan["root_addition"])

    assert (
        set(manifest["stable_exports"])
        <= set(baseline["exports"]) | authorized_additions
    )
    assert baseline["status"] == "pre_open_source_migration_baseline"


def test_snapshot_diagnostic_forbids_unreviewed_regeneration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = public_api_snapshot.generate()
    changed["exports"]["Circuit"] = {"changed": True}
    monkeypatch.setattr(public_api_snapshot, "generate", lambda: changed)

    errors = public_api_snapshot.validate()

    assert len(errors) == 1
    assert "do not regenerate" in errors[0]
