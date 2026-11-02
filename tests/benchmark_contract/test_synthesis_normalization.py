"""AO-01 comparisons preserve the original measured artifact and failed scopes."""

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pytest

from benchmarks.synthesis.normalization import differences

pytestmark = pytest.mark.benchmark_contract
BASELINE = (
    Path(__file__).resolve().parents[2] / "benchmarks/synthesis/baselines/ao00.json"
)


def test_frozen_identity_comparison_keeps_the_timeout_incomplete() -> None:
    rows = differences(BASELINE, BASELINE)
    assert len(rows) == 90
    assert sum(row.classification == "unchanged" for row in rows) == 89
    assert sum(row.classification == "incomplete" for row in rows) == 1


def test_comparison_refuses_even_a_whitespace_rewrite_of_original(
    tmp_path: Path,
) -> None:
    rewritten = tmp_path / "changed.json"
    rewritten.write_bytes(BASELINE.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="historical"):
        differences(rewritten, BASELINE)


def test_comparison_detects_a_resource_regression_instead_of_claiming_improvement(
    tmp_path: Path,
) -> None:
    synthetic = json.loads(BASELINE.read_text())
    resource = synthetic["rows"][0]["resources"]
    resource["operations"] += 1
    resource["operation_counts"]["x"] = resource["operation_counts"].get("x", 0) + 1
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(synthetic))
    row = differences(BASELINE, path)[0]
    assert row.classification == "regressed"
    assert row.deltas["operations"] == 1


def test_changed_budget_is_not_a_comparable_resource_result(tmp_path: Path) -> None:
    altered = json.loads(BASELINE.read_text())
    altered["configuration"]["objective"] = "depth"
    path = tmp_path / "different_objective.json"
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="configuration"):
        differences(BASELINE, path)


def test_checked_measurements_preserve_oracles_and_reproduce_comparison() -> None:
    current = BASELINE.with_name("ao01.json")
    rows = differences(BASELINE, current)
    assert not any(row.classification in {"regressed", "tradeoff"} for row in rows)
    assert sum(row.classification == "incomplete" for row in rows) == 1
    before, after = json.loads(BASELINE.read_text()), json.loads(current.read_text())
    assert all(
        after["source_hashes"][name] == digest
        for name, digest in before["source_hashes"].items()
    )
    comparison = json.loads(BASELINE.with_name("ao01-comparison.json").read_text())
    assert comparison["summary"] == dict(Counter(r.classification for r in rows))
    assert comparison["rows"] == [asdict(row) for row in rows]
