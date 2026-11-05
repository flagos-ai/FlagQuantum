"""Artifact transport retries do not reuse partial data or hide failures."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
ACTION = ROOT / ".github/actions/download-artifact/action.yml"


def test_retry_keeps_digest_checks_and_uses_only_the_successful_attempt() -> None:
    action = yaml.safe_load(ACTION.read_text())
    stage, first, retry = action["runs"]["steps"]
    assert stage["id"] == "stage"
    assert first["continue-on-error"] is True
    assert not retry.get("continue-on-error", False)
    assert retry["if"] == "${{ !cancelled() && steps.first.outcome == 'failure' }}"
    assert first["uses"] == retry["uses"] == "actions/download-artifact@v8.0.2"
    for key in ("name", "pattern", "merge-multiple", "digest-mismatch"):
        assert first["with"][key] == retry["with"][key]
    assert first["with"]["digest-mismatch"] == "error"
    assert first["with"]["path"] != retry["with"]["path"]
    output = action["outputs"]["path"]["value"]
    assert "steps.first.outcome == 'success'" in output
    assert "{0}/first" in output and "{0}/retry" in output


@pytest.mark.skipif(os.name == "nt", reason="Coverage transport stages on Linux bash")
@pytest.mark.parametrize(
    "name,pattern,valid",
    [
        ("plan", "", True),
        ("", "shard-*", True),
        ("", "", False),
        ("plan", "shard-*", False),
    ],
)
def test_stage_refuses_ambiguous_selectors_and_keeps_retry_directory_clean(
    tmp_path: Path, name: str, pattern: str, valid: bool
) -> None:
    action = yaml.safe_load(ACTION.read_text())
    stage = action["runs"]["steps"][0]["run"]
    output = tmp_path / "output"
    result = subprocess.run(
        ["bash", "-e", "-c", stage],
        env={
            **os.environ,
            "RUNNER_TEMP": str(tmp_path),
            "GITHUB_OUTPUT": str(output),
            "FQ_ARTIFACT_NAME": name,
            "FQ_ARTIFACT_PATTERN": pattern,
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    if not valid:
        assert result.returncode != 0 and not output.exists()
        return
    assert result.returncode == 0, result.stderr
    root = Path(output.read_text().strip().split("=", 1)[1])
    assert root.parent == tmp_path
    (root / "first" / "partial").write_text("abandoned first transfer")
    assert (root / "retry").is_dir()
    assert not list((root / "retry").iterdir())


def test_coverage_workflow_consumes_only_isolated_download_outputs() -> None:
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
    for job, artifact_id, environment in [
        ("coverage-shard", "coverage-plan-download", "FQ_COVERAGE_PLAN"),
        ("coverage", "coverage-data", "FQ_COVERAGE_INPUTS"),
    ]:
        steps = jobs[job]["steps"]
        index, step = next(
            (i, step) for i, step in enumerate(steps) if step.get("id") == artifact_id
        )
        assert step["uses"] == "./.github/actions/download-artifact"
        assert step["timeout-minutes"] == 6
        assert not step.get("continue-on-error", False)
        consumers = [
            step for step in steps[index + 1 :] if environment in step.get("env", {})
        ]
        assert len(consumers) == 2
        assert all(
            step["env"][environment] == "${{ steps." + artifact_id + ".outputs.path }}"
            for step in consumers
        )
