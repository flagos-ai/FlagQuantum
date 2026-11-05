"""Artifact retries stay bounded, preserve inputs, and retain job ownership."""

from __future__ import annotations

import re
from itertools import pairwise, product
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
ACTION = ROOT / ".github/actions/upload-artifact/action.yml"


def _steps() -> list[dict[str, object]]:
    return yaml.safe_load(ACTION.read_text())["runs"]["steps"]


@pytest.mark.parametrize(
    "failures,expected_calls,passed",
    [(0, 1, True), (1, 2, True), (2, 3, True), (3, 3, False)],
)
def test_retry_graph_stops_on_success_and_propagates_persistent_failure(
    failures: int,
    expected_calls: int,
    passed: bool,
) -> None:
    outcomes: dict[str, str] = {}
    calls = 0
    rejected = False
    for step in _steps():
        condition = step["if"]
        assert "!cancelled()" in condition
        dependency = re.search(r"steps\.([a-z]+)\.outcome == 'failure'", condition)
        if dependency and outcomes.get(dependency[1]) != "failure":
            continue
        calls += 1
        outcome = "failure" if calls <= failures else "success"
        outcomes[step["id"]] = outcome
        rejected |= outcome == "failure" and not step.get("continue-on-error", False)
    assert calls == expected_calls
    assert (not rejected) == passed


def test_only_uploads_are_retried_and_the_last_attempt_is_required() -> None:
    steps = _steps()
    assert len(steps) == 3
    assert all(step["uses"] == "actions/upload-artifact@v4" for step in steps)
    assert steps[0]["if"] == "${{ !cancelled() }}"
    for previous, following in pairwise(steps):
        assert (
            following["if"]
            == "${{ !cancelled() && steps."
            + previous["id"]
            + ".outcome == 'failure' }}"
        )
    assert [step.get("continue-on-error", False) for step in steps] == [
        True,
        True,
        False,
    ]
    for index, step in enumerate(steps):
        for key in (
            "name",
            "path",
            "retention-days",
            "compression-level",
            "include-hidden-files",
            "if-no-files-found",
        ):
            assert step["with"][key] == "${{ inputs." + key + " }}"
        assert step["with"]["overwrite"] == (
            "true" if index else "${{ inputs.overwrite }}"
        )
    assert (
        yaml.safe_load(ACTION.read_text())["inputs"]["include-hidden-files"]["default"]
        == "false"
    )


def test_workflow_uploads_have_budgets_and_unique_names_for_every_matrix_leg() -> None:
    count = 0
    for workflow in ("ci.yml", "local-gpu.yml"):
        names: set[str] = set()
        jobs = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())[
            "jobs"
        ]
        for job_name, job in jobs.items():
            matrix = job.get("strategy", {}).get("matrix", {})
            rows = matrix.get("include") or [
                dict(zip(matrix, row, strict=True)) for row in product(*matrix.values())
            ]
            for index, row in enumerate(rows):
                for step in job["steps"]:
                    assert step.get("uses") != "actions/upload-artifact@v4"
                    if step.get("uses") != "./.github/actions/upload-artifact":
                        continue
                    assert step["timeout-minutes"] == 6
                    assert not step.get("continue-on-error", False)
                    context = {
                        "github.job": job_name,
                        "github.sha": "commit",
                        "github.run_id": "run",
                        "github.run_attempt": "attempt",
                        "strategy.job-index || '0'": str(index),
                    }
                    context.update(
                        {"matrix." + key: str(value) for key, value in row.items()}
                    )
                    name = re.sub(
                        r"\$\{\{\s*(.*?)\s*\}\}",
                        lambda match, values=context: values[match[1]],
                        step["with"]["name"],
                    )
                    assert (
                        name not in names
                    ), f"{workflow}: two jobs own artifact {name}"
                    names.add(name)
                    count += 1
    assert count > 64
