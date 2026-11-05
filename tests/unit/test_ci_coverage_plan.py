"""Coverage fan-out preserves the complete selection and source identity."""

import json

import pytest

from tools.ci_coverage_plan import verify_plan, write_plan

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("requested", [1, 16, 192])
def test_plan_partitions_real_tests_once_without_empty_shards(
    tmp_path, monkeypatch, requested
):
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    nodes = tuple(
        f"tests/unit/test_{i}.py::test_case[{j}]" for i in range(211) for j in range(3)
    )
    directory = tmp_path / "plan"
    count = write_plan(nodes, directory, requested)
    shards = [verify_plan(directory, i).read_text().splitlines() for i in range(count)]
    assert all(shards)
    assert sorted(node for shard in shards for node in shard) == sorted(nodes)
    owners = {}
    for i, shard in enumerate(shards):
        for node in shard:
            filename = node.split("::")[0]
            assert owners.setdefault(filename, i) == i
    matrix = json.loads((tmp_path / "output").read_text().split("matrix=", 1)[1])
    assert [row["shard"] for row in matrix["include"]] == list(range(count))
    assert {row["hosted-bucket"] for row in matrix["include"]} <= set(range(8))


def test_small_selection_reduces_count_and_refuses_tampered_or_stale_plan(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    assert write_plan(("tests/test_a.py::test_a",), tmp_path / "plan", 192) == 1
    verify_plan(tmp_path / "plan", 0)
    with pytest.raises(ValueError, match="index or count"):
        verify_plan(tmp_path / "plan", 1)
    monkeypatch.setenv("GITHUB_SHA", "b" * 40)
    with pytest.raises(ValueError, match="source commit"):
        verify_plan(tmp_path / "plan", 0)
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    (tmp_path / "plan/shard-0.txt").write_text("tests/test_a.py::test_b\n")
    with pytest.raises(ValueError, match="contents differ"):
        verify_plan(tmp_path / "plan", 0)


@pytest.mark.parametrize(
    "nodes,count",
    [
        ((), 16),
        (("tests/a.py::test_a",) * 2, 16),
        (("tests/a.py::test_a",), 0),
        (("tests/a.py::test_a",), 193),
    ],
)
def test_invalid_collection_and_unbounded_fanout_fail(tmp_path, nodes, count):
    with pytest.raises(ValueError):
        write_plan(nodes, tmp_path / "plan", count)


def test_collected_plan_executes_the_same_selection_and_records_cpu(
    tmp_path, monkeypatch
):
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    tests = tmp_path / "tests"
    tests.mkdir()
    for index in range(3):
        (tests / f"test_{index}.py").write_text(
            "import pytest\npytestmark = pytest.mark.unit\n"
            "@pytest.mark.parametrize('value', [0, 1, 2])\n"
            "def test_value(value):\n    assert value >= 0\n"
        )
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers = unit: selected\n")
    report = tmp_path / "resources"
    report.mkdir()
    (report / "resources.json").write_text(
        json.dumps(
            {
                "effective_cpus": 2,
                "shared_registrations_budgeted": 1,
            }
        )
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(root),
        "GITHUB_SHA": "c" * 40,
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "FQ_CI_REPORT_DIR": str(report),
    }
    plan = tmp_path / "plan"
    collect = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-m",
            "unit",
            "-p",
            "tools.pytest_shard",
            "--fq-write-coverage-plan",
            str(plan),
            "--fq-coverage-plan-count",
            "2",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert collect.returncode == 0, collect.stdout + collect.stderr
    total = 0
    for index in range(2):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "@" + str(plan / f"shard-{index}.txt"),
                "-q",
                "-p",
                "tools.pytest_shard",
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        usage = json.loads((report / "test-usage.json").read_text())
        total += usage["collected_tests"]
        assert usage["failed_tests"] == 0
        assert usage["process_cpu_seconds"] > 0
        assert usage["wall_seconds"] > 0
        assert usage["assigned_ci_cpus"] == 2
    assert total == 9


def test_large_untimed_file_is_spread_instead_of_forming_one_long_tail(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GITHUB_SHA", "d" * 40)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    nodes = tuple(f"tests/test_new_sweep.py::test_case[{i}]" for i in range(160))
    assert write_plan(nodes, tmp_path / "plan", 192) == 10
    shards = [
        (tmp_path / "plan" / f"shard-{i}.txt").read_text().splitlines()
        for i in range(10)
    ]
    assert all(len(shard) == 16 for shard in shards)
    assert sorted(n for shard in shards for n in shard) == sorted(nodes)
