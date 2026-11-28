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
        ("coverage-shard", "source-incremental-download", "FQ_COVERAGE_PLAN"),
        ("coverage", "coverage-data", "FQ_COVERAGE_INPUTS"),
    ]:
        steps = jobs[job]["steps"]
        index, step = next(
            (i, step) for i, step in enumerate(steps) if step.get("id") == artifact_id
        )
        expected_path = "${{ steps." + artifact_id + ".outputs.path }}"
        if job == "coverage":
            assert step["timeout-minutes"] == 4
            assert not step.get("continue-on-error", False)
            assert "python3 -I tools/ci/ci_coverage_artifacts.py" in step["run"]
            assert "timeout --kill-after=10s 240s" in step["run"]
            assert not any(s.get("id") == "coverage-data-retry" for s in steps)
        else:
            retry, promotion, preserved_plan = steps[index + 1 : index + 4]
            assert step["uses"] == retry["uses"] == "actions/download-artifact@v8.0.2"
            assert step["continue-on-error"] is True
            assert step["timeout-minutes"] == 1
            assert retry["timeout-minutes"] == 2
            assert not retry.get("continue-on-error", False)
            assert (
                retry["if"]
                == "${{ !cancelled() && steps.source-incremental-download.outcome == 'failure' }}"
            )
            assert (
                step["with"]["name"]
                == retry["with"]["name"]
                == "coverage-plan-${{ github.sha }}"
            )
            assert (
                step["with"]["digest-mismatch"]
                == retry["with"]["digest-mismatch"]
                == "error"
            )
            assert step["with"]["path"] != retry["with"]["path"]
            assert (
                promotion["if"]
                == "${{ steps.source-incremental-retry.outcome == 'success' }}"
            )
            assert promotion["env"]["FQ_CI_SOURCE_DIR"] == step["with"]["path"]
            assert (
                preserved_plan["name"]
                == "Preserve the verified coverage plan before source fallback"
            )
            assert not preserved_plan.get("continue-on-error", False)
            assert preserved_plan["id"] == "coverage-plan-stage"
            expected_path = "${{ steps.coverage-plan-stage.outputs.path }}"
        consumers = [
            step for step in steps[index + 1 :] if environment in step.get("env", {})
        ]
        assert len(consumers) == 2
        assert all(step["env"][environment] == expected_path for step in consumers)
