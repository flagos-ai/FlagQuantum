"""The security workflow preserves pinned actions and the exact SARIF handoff."""

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def _workflow() -> dict[str, Any]:
    document = yaml.load(
        (ROOT / ".github/workflows/security-scorecard.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    assert isinstance(document, dict)
    return document


def test_scorecard_is_independent_and_has_only_required_write_scopes() -> None:
    document = _workflow()
    assert set(document["on"]) == {"schedule", "workflow_dispatch"}
    assert document["on"]["schedule"] == [{"cron": "19 20 * * 6"}]
    assert document["permissions"] == {"contents": "read"}
    assert document["concurrency"]["cancel-in-progress"] == "false"
    assert document["concurrency"]["queue"] == "max"
    assert set(document["jobs"]) == {"scorecard"}
    job = document["jobs"]["scorecard"]
    assert job["runs-on"] == "ubuntu-24.04"
    assert job["timeout-minutes"] == "10"
    assert job["permissions"] == {
        "contents": "read",
        "security-events": "write",
        "id-token": "write",
    }
    assert job["concurrency"] == {"group": "flagquantum-gh-2", "queue": "max"}


def test_pinned_scorecard_output_is_the_uploaded_sarif() -> None:
    steps = _workflow()["jobs"]["scorecard"]["steps"]
    assert len(steps) == 3
    checkout, producer, upload = steps
    assert checkout["uses"] == "actions/checkout@v5"
    assert checkout["with"]["persist-credentials"] == "false"
    assert (
        producer["uses"]
        == "ossf/scorecard-action@f49aabe0b5af0936a0987cfb85d86b75731b0186"
    )
    assert producer["with"] == {
        "results_file": "results.sarif",
        "results_format": "sarif",
        "publish_results": "true",
    }
    assert (
        upload["uses"]
        == "github/codeql-action/upload-sarif@b56ba49b26e50535fa1e7f7db0f4f7b4bf65d80d"
    )
    assert upload["with"] == {"sarif_file": producer["with"]["results_file"]}
    # Engine defaults stop at a failed producer, rather than uploading stale data.
    assert not any("if" in step or "continue-on-error" in step for step in steps)
    assert not any("env" in step or "run" in step for step in steps)
