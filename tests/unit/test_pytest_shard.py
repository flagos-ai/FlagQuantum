from __future__ import annotations

import json

import pytest

from tools.ci.pytest_shard import (
    DURATION_HINTS,
    balanced_assignments,
    load_duration_hints,
    shard_for,
)

pytestmark = pytest.mark.unit


def test_a_node_id_has_one_stable_shard() -> None:
    nodeid = "tests/unit/test_example.py::test_case[value]"
    assert shard_for(nodeid, 7) == shard_for(nodeid, 7)
    assert 0 <= shard_for(nodeid, 7) < 7


def test_every_node_id_is_assigned_exactly_once() -> None:
    nodeids = tuple(f"tests/unit/test_{index}.py::test_case" for index in range(100))
    shard_by_nodeid = balanced_assignments(
        nodeids,
        4,
        default_seconds=0.1,
        durations={},
    )
    assignments = {
        nodeid: [index for index in range(4) if shard_by_nodeid[nodeid] == index]
        for nodeid in nodeids
    }
    assert all(len(indices) == 1 for indices in assignments.values())
    assert {indices[0] for indices in assignments.values()} == {0, 1, 2, 3}


def test_a_non_positive_shard_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        shard_for("tests/unit/test_example.py::test_case", 0)


def test_duration_balancing_separates_the_longest_tests() -> None:
    nodeids = tuple(f"test_{index}" for index in range(8))
    durations = {nodeid: float(8 - index) for index, nodeid in enumerate(nodeids)}

    assignments = balanced_assignments(
        nodeids,
        4,
        default_seconds=0.1,
        durations=durations,
    )
    loads = [0.0] * 4
    for nodeid, shard in assignments.items():
        loads[shard] += durations[nodeid]

    assert loads == [9.0, 9.0, 9.0, 9.0]


def test_checked_in_duration_hints_are_positive_and_cited() -> None:
    default_seconds, durations = load_duration_hints(DURATION_HINTS)
    payload = json.loads(DURATION_HINTS.read_text(encoding="utf-8"))

    assert default_seconds > 0
    assert len(durations) >= 100
    assert all(nodeid.startswith("tests/") for nodeid in durations)
    assert all(seconds > 0 for seconds in durations.values())
    assert payload["source"]["run_id"] == 37877763072
    assert len(payload["source"]["commit"]) == 40
    assert all(
        (DURATION_HINTS.parents[1] / nodeid.split("::", 1)[0]).is_file()
        for nodeid in durations
    )


def test_checked_in_slow_tests_are_balanced_across_eight_shards() -> None:
    default_seconds, durations = load_duration_hints(DURATION_HINTS)
    assignments = balanced_assignments(
        tuple(durations),
        8,
        default_seconds=default_seconds,
        durations=durations,
    )
    loads = [0.0] * 8
    for nodeid, shard in assignments.items():
        loads[shard] += durations[nodeid]

    assert max(loads) - min(loads) < 5.0


def test_shard_helper_does_not_require_posix_resource_on_windows(monkeypatch):
    import sys
    from types import SimpleNamespace

    from tools.ci import pytest_shard

    monkeypatch.setattr(pytest_shard, "os", SimpleNamespace(name="nt"))
    monkeypatch.setitem(sys.modules, "resource", None)
    assert pytest_shard._cpu_seconds() >= 0


@pytest.mark.parametrize("workers", [0, 2])
def test_complete_per_node_timing_keeps_serial_parallel_and_failure_outcomes(
    tmp_path, workers
):
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_duration_sample.py").write_text(
        "import pytest\n"
        "@pytest.mark.parametrize('value', range(6))\n"
        "def test_value(value):\n"
        "    if value == 4:\n        pytest.skip('deliberate boundary')\n"
        "    assert value != 5\n"
    )
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    report = tmp_path / "reports"
    report.mkdir()
    (report / "resources.json").write_text(
        json.dumps({"effective_cpus": 2, "shared_registrations_budgeted": 1})
    )
    environment = dict(
        os.environ,
        PYTHONPATH=str(root),
        PYTEST_ADDOPTS="",
        FQ_CI_REPORT_DIR=str(report),
        GITHUB_SHA="d" * 40,
    )
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "tools.ci.pytest_shard",
        "-q",
        "--tb=short",
    ]
    if workers:
        command += ["-n", str(workers)]
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 1
    assert "1 failed" in result.stdout
    assert "4 passed" in result.stdout
    assert "1 skipped" in result.stdout
    timing = json.loads((report / "test-durations.json").read_text())
    assert timing["schema"] == "flagquantum_pytest_durations_v1"
    assert timing["source_sha"] == "d" * 40
    assert timing["exit_status"] == 1
    assert timing["complete"] is True
    assert timing["selected_nodes"] == timing["recorded_nodes"] == 6
    expected = {f"tests/test_duration_sample.py::test_value[{i}]" for i in range(6)}
    assert set(timing["tests"]) == expected
    for phases in timing["tests"].values():
        assert set(phases) == {"setup", "call", "teardown"}
        assert all(seconds >= 0 for seconds in phases.values())
    usage = json.loads((report / "test-usage.json").read_text())
    assert usage["failed_tests"] == 1
    assert usage["exit_status"] == 1
