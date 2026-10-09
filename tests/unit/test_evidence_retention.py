from __future__ import annotations

from pathlib import Path

import pytest

from tools.check_evidence_retention import SCHEMA, retention_errors

pytestmark = pytest.mark.unit


def manifest(entries: dict[str, object]) -> dict[str, object]:
    return {"schema": SCHEMA, "entries": entries}


def add_file(root: Path, path: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("evidence\n", encoding="utf-8")


def test_existing_evidence_is_grandfathered() -> None:
    assert retention_errors((), manifest=manifest({})) == ()


def test_new_evidence_needs_a_retention_entry(tmp_path: Path) -> None:
    path = "benchmarks/results/smoke/new-result.json"
    add_file(tmp_path, path)

    errors = retention_errors((path,), root=tmp_path, manifest=manifest({}))

    assert errors == (
        f"{path}: new benchmark evidence needs an entry in "
        "benchmarks/evidence-retention.json",
    )


def test_temporary_evidence_records_a_review_date(tmp_path: Path) -> None:
    path = "benchmarks/results/smoke/investigation.txt"
    add_file(tmp_path, path)
    entry = {
        "class": "temporary",
        "reason": "Needed while the optimizer regression is investigated.",
        "review_after": "2026-12-01",
    }

    assert (
        retention_errors((path,), root=tmp_path, manifest=manifest({path: entry})) == ()
    )


@pytest.mark.parametrize("retention_class", ["current_claim", "regression_baseline"])
def test_durable_evidence_names_an_existing_consumer(
    tmp_path: Path, retention_class: str
) -> None:
    path = "benchmarks/results/comparison/result.csv"
    consumer = "docs/reference/PERFORMANCE.md"
    add_file(tmp_path, path)
    add_file(tmp_path, consumer)
    entry = {
        "class": retention_class,
        "reason": "Supports the documented backend comparison.",
        "consumers": [consumer],
    }

    assert (
        retention_errors((path,), root=tmp_path, manifest=manifest({path: entry})) == ()
    )


def test_missing_consumer_and_stale_entries_fail(tmp_path: Path) -> None:
    path = "benchmarks/results/local/missing.json"
    entry = {
        "class": "current_claim",
        "reason": "Supports a performance statement in the documentation.",
        "consumers": ["docs/reference/MISSING.md"],
    }

    errors = retention_errors((), root=tmp_path, manifest=manifest({path: entry}))

    assert f"{path}: retention entry points to a missing file" in errors
    assert f"{path}: consumer does not exist: docs/reference/MISSING.md" in errors


def test_invalid_temporary_lifecycle_fails(tmp_path: Path) -> None:
    path = "benchmarks/results/smoke/result.log"
    add_file(tmp_path, path)
    entry = {
        "class": "temporary",
        "reason": "   ",
        "review_after": "next quarter",
    }

    errors = retention_errors((path,), root=tmp_path, manifest=manifest({path: entry}))

    assert f"{path}: reason must explain why the artifact is retained" in errors
    assert f"{path}: temporary evidence needs review_after as YYYY-MM-DD" in errors


def test_readmes_are_not_generated_evidence() -> None:
    assert (
        retention_errors(("benchmarks/results/smoke/README.md",), manifest=manifest({}))
        == ()
    )


def test_supersedes_stays_within_benchmark_results(tmp_path: Path) -> None:
    path = "benchmarks/results/smoke/replacement.json"
    add_file(tmp_path, path)
    entry = {
        "class": "temporary",
        "reason": "Replaces evidence from the previous investigation.",
        "review_after": "2026-12-01",
        "supersedes": ["artifacts/old-result.json"],
    }

    errors = retention_errors((path,), root=tmp_path, manifest=manifest({path: entry}))

    assert errors == (f"{path}: supersedes must name a path under benchmarks/results/",)
