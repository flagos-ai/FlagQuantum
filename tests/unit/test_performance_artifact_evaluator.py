"""Tests for strict and optional performance baseline handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.evaluate_performance_artifact import resolve_optional_baseline

pytestmark = pytest.mark.unit


def test_optional_baseline_is_used_when_present(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}")

    assert resolve_optional_baseline(baseline) == baseline


def test_missing_optional_baseline_falls_back_to_intrinsic_thresholds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = tmp_path / "missing.json"

    assert resolve_optional_baseline(baseline) is None
    assert "evaluating intrinsic thresholds only" in capsys.readouterr().out
