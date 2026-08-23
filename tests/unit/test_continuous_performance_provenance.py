"""Fail-closed provenance coverage for the continuous benchmark runner."""

from __future__ import annotations

import pytest

from benchmarks.continuous_performance import _commit

pytestmark = pytest.mark.unit


def test_explicit_source_commit_supports_archived_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "a" * 40
    monkeypatch.setenv("FLAGQUANTUM_SOURCE_COMMIT", commit.upper())

    assert _commit() == commit


@pytest.mark.parametrize("commit", ["", "unknown", "g" * 40, "a" * 39])
def test_explicit_source_commit_rejects_invalid_provenance(
    monkeypatch: pytest.MonkeyPatch, commit: str
) -> None:
    monkeypatch.setenv("FLAGQUANTUM_SOURCE_COMMIT", commit)

    with pytest.raises(ValueError, match="40-character SHA"):
        _commit()
