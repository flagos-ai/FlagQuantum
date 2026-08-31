from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import public_api_snapshot

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_public_api_matches_pre_open_source_migration_baseline() -> None:
    assert public_api_snapshot.validate() == ()


def test_baseline_covers_exact_stable_export_manifest() -> None:
    manifest = json.loads((ROOT / "docs/public_api_v1.json").read_text())
    baseline = json.loads(
        (ROOT / "contracts/public-api-v0.2-baseline.json").read_text()
    )

    assert set(baseline["exports"]) == set(manifest["stable_exports"])
    assert baseline["status"] == "pre_open_source_migration_baseline"


def test_snapshot_diagnostic_forbids_unreviewed_regeneration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(public_api_snapshot, "generate", lambda: {"changed": True})

    errors = public_api_snapshot.validate()

    assert len(errors) == 1
    assert "do not regenerate" in errors[0]
