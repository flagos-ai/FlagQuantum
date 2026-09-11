import json
import sys
import time
from pathlib import Path

import pytest

from tools.run_multinode_watchdog import run_jobs


def test_peer_failure_terminates_survivor_and_retains_diagnostics(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "survivor-finished"
    jobs = [
        {
            "name": "failed-node",
            "command": [sys.executable, "-c", "import sys; sys.exit(137)"],
        },
        {
            "name": "surviving-node",
            "command": [
                sys.executable,
                "-c",
                f"import time; time.sleep(30); open({str(marker)!r}, 'w').close()",
            ],
        },
    ]
    started = time.monotonic()
    payload = run_jobs(jobs, output_dir=tmp_path / "logs", timeout_seconds=10)

    assert time.monotonic() - started < 3
    assert payload["status"] == "failed_closed"
    assert payload["reason"] == "peer_launcher_failure"
    assert payload["first_failure_job"] == "failed-node"
    assert payload["failure_detected_seconds"] is not None
    assert payload["cleanup_elapsed_seconds"] < 3
    assert payload["all_launchers_exited"] is True
    assert payload["cleanup_verified"] is False
    assert not marker.exists()
    retained = json.loads((tmp_path / "logs" / "watchdog.json").read_text())
    assert retained["jobs"] == payload["jobs"]


def test_all_jobs_must_complete_for_success(tmp_path: Path) -> None:
    jobs = [
        {"name": "node-0", "command": [sys.executable, "-c", "pass"]},
        {"name": "node-1", "command": [sys.executable, "-c", "pass"]},
    ]
    payload = run_jobs(jobs, output_dir=tmp_path, timeout_seconds=3)
    assert payload["status"] == "passed"
    assert payload["reason"] == "completed"
    assert payload["cleanup_verified"] is True
    assert payload["failure_detected_seconds"] is None
    assert payload["cleanup_elapsed_seconds"] == 0.0


def test_cleanup_commands_are_required_and_audited_on_failure(tmp_path: Path) -> None:
    jobs = [
        {
            "name": "failed-node",
            "command": [sys.executable, "-c", "raise SystemExit(1)"],
            "cleanup_command": [sys.executable, "-c", "pass"],
        },
        {
            "name": "peer-node",
            "command": [sys.executable, "-c", "import time; time.sleep(30)"],
            "cleanup_command": [sys.executable, "-c", "pass"],
        },
    ]
    payload = run_jobs(jobs, output_dir=tmp_path, timeout_seconds=5)
    assert payload["cleanup_verified"] is True
    assert payload["cleanup_failures"] == []
    assert payload["cleanup_results"] == {"failed-node": 0, "peer-node": 0}


@pytest.mark.parametrize("jobs", [[], [{"name": "only", "command": ["true"]}]])
def test_requires_multiple_jobs(jobs: list[dict], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least two jobs"):
        run_jobs(jobs, output_dir=tmp_path, timeout_seconds=1)
