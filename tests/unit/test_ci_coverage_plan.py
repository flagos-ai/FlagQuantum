"""Coverage fan-out preserves the complete selection and source identity."""

import json
from pathlib import Path

import pytest

from tools.ci.ci_coverage_plan import verify_plan, write_plan

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("requested", [1, 16, 192, 256])
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
        (("tests/a.py::test_a",), 257),
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
            "tools.ci.pytest_shard",
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
                "tools.ci.pytest_shard",
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


def test_measured_heavy_cases_in_one_file_can_use_separate_slots(tmp_path, monkeypatch):
    from tools.ci import ci_coverage_plan

    monkeypatch.setenv("GITHUB_SHA", "e" * 40)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    nodes = tuple(f"tests/test_full_state.py::test_case[{i}]" for i in range(16))
    hints = {
        node: seconds for node, seconds in zip(nodes[:3], (603, 479, 468), strict=True)
    }
    monkeypatch.setattr(ci_coverage_plan, "load_duration_hints", lambda: (0.05, hints))
    directory = tmp_path / "plan"
    assert write_plan(nodes, directory, 4) == 4
    shards = [verify_plan(directory, i).read_text().splitlines() for i in range(4)]
    assert sorted(n for shard in shards for n in shard) == sorted(nodes)
    assert all(sum(node in shard for node in nodes[:3]) <= 1 for shard in shards)


@pytest.mark.parametrize("requested", [1, 4])
def test_original_optimization_workloads_receive_only_their_finite_budgets(
    tmp_path, monkeypatch, requested
):
    from tools.ci.ci_coverage_plan import LONG_TEST_TIMEOUT_MINUTES

    monkeypatch.setenv("GITHUB_SHA", "e" * 40)
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    long_nodes = tuple(LONG_TEST_TIMEOUT_MINUTES)
    neighbors = tuple(
        node.rsplit("::", 1)[0] + "::test_neighbor" for node in long_nodes
    )
    ordinary = "tests/unit/test_small.py::test_small"
    nodes = (*long_nodes, *neighbors, ordinary)
    count = write_plan(nodes, tmp_path / "plan", requested)
    rows = json.loads((tmp_path / "output").read_text().split("matrix=", 1)[1])[
        "include"
    ]
    shards = [
        verify_plan(tmp_path / "plan", i).read_text().splitlines() for i in range(count)
    ]
    assert sorted(node for shard in shards for node in shard) == sorted(nodes)
    assert len(rows) == count
    for row, shard in zip(rows, shards, strict=True):
        expected = sum(LONG_TEST_TIMEOUT_MINUTES.get(node, 0) for node in shard)
        assert row["timeout-minutes"] == max(60, expected)
    if requested == 1:
        assert rows[0]["timeout-minutes"] == 240
    else:
        owners = [
            i for i, shard in enumerate(shards) if any(n in long_nodes for n in shard)
        ]
        assert len(owners) == 2
        assert all(rows[i]["timeout-minutes"] == 120 for i in owners)
        assert all(
            rows[i]["timeout-minutes"] == 60 for i in range(count) if i not in owners
        )


@pytest.mark.parametrize("reverse", [False, True])
def test_measured_full_state_cases_do_not_share_a_partition(
    tmp_path, monkeypatch, reverse
):
    monkeypatch.setenv("GITHUB_SHA", "f" * 40)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    full_state = (
        "tests/integration/test_synthesis_qmod_vlasov.py::"
        "test_original_vlasov_preserves_full_state_and_cleans_workspace",
        "tests/integration/test_synthesis_glued_trees_notebook.py::"
        "test_wide_full_state_at_original_first_time",
    )
    ordinary = tuple(f"tests/test_ordinary.py::test_case[{i}]" for i in range(32))
    nodes = (*full_state, *ordinary)
    if reverse:
        nodes = tuple(reversed(nodes))
    directory = tmp_path / "plan"
    assert write_plan(nodes, directory, 4) == 4
    shards = [verify_plan(directory, i).read_text().splitlines() for i in range(4)]
    assert sorted(node for shard in shards for node in shard) == sorted(nodes)
    for node in full_state:
        assert [shard for shard in shards if node in shard] == [[node]]


@pytest.mark.parametrize("requested", [1, 2, 4, 16])
def test_exclusive_cases_and_bounded_fanout_preserve_every_original_node(
    tmp_path, monkeypatch, requested
):
    from tools.ci import ci_coverage_plan

    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    nodes = tuple(f"tests/test_large.py::test_case[{i}]" for i in range(8))
    hints = dict.fromkeys(nodes, 300.0)
    monkeypatch.setattr(ci_coverage_plan, "load_duration_hints", lambda: (0.05, hints))
    count = write_plan(nodes, tmp_path / "plan", requested)
    rows = json.loads((tmp_path / "output").read_text().split("matrix=", 1)[1])[
        "include"
    ]
    shards = [
        verify_plan(tmp_path / "plan", i).read_text().splitlines() for i in range(count)
    ]
    assert sorted(node for shard in shards for node in shard) == sorted(nodes)
    assert count == min(requested, len(nodes))
    for row, shard in zip(rows, shards, strict=True):
        assert row["exclusive"] == (len(shard) == 1)
        assert row["timeout-minutes"] == 60


def test_measured_full_case_has_no_untimed_neighbors(tmp_path, monkeypatch):
    from tools.ci import ci_coverage_plan

    monkeypatch.setenv("GITHUB_SHA", "b" * 40)
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    heavy = "tests/test_full.py::test_complete_original_workflow"
    ordinary = tuple(f"tests/test_full.py::test_neighbor[{i}]" for i in range(32))
    nodes = (heavy, *ordinary)
    monkeypatch.setattr(
        ci_coverage_plan, "load_duration_hints", lambda: (0.05, {heavy: 1200.0})
    )
    count = write_plan(nodes, tmp_path / "plan", 256)
    rows = json.loads((tmp_path / "output").read_text().split("matrix=", 1)[1])[
        "include"
    ]
    shards = [
        verify_plan(tmp_path / "plan", i).read_text().splitlines() for i in range(count)
    ]
    owner = next(i for i, shard in enumerate(shards) if heavy in shard)
    assert shards[owner] == [heavy]
    assert rows[owner]["exclusive"] is True
    assert sorted(node for shard in shards for node in shard) == sorted(nodes)


def test_full_256_way_plan_remains_nonempty_and_complete(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "c" * 40)
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    nodes = tuple(f"tests/test_{i}.py::test_case" for i in range(300))
    assert write_plan(nodes, tmp_path / "plan", 256) == 256
    shards = [
        verify_plan(tmp_path / "plan", i).read_text().splitlines() for i in range(256)
    ]
    assert all(shards)
    assert sorted(node for shard in shards for node in shard) == sorted(nodes)
    rows = json.loads((tmp_path / "output").read_text().split("matrix=", 1)[1])[
        "include"
    ]
    assert len(rows) == 256
    assert not any(row["exclusive"] for row in rows)
    assert {row["runner"] for row in rows} == {
        "flagquantum-bw1100-cpu",
        "flagquantum-mc550-cpu",
    }


@pytest.mark.parametrize("seconds,expected", [(299.9, False), (300.0, True)])
def test_only_indivisible_measured_heavy_cases_use_qualified_cpu_pool(
    tmp_path, monkeypatch, seconds, expected
):
    from tools.ci import ci_coverage_plan

    monkeypatch.setenv("GITHUB_SHA", "b" * 40)
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    heavy = "tests/test_full.py::test_original_workload"
    ordinary = "tests/test_small.py::test_original_workload"
    monkeypatch.setattr(
        ci_coverage_plan, "load_duration_hints", lambda: (0.05, {heavy: seconds})
    )
    assert write_plan((heavy, ordinary), tmp_path / "plan", 2) == 2
    rows = json.loads((tmp_path / "output").read_text().split("matrix=", 1)[1])[
        "include"
    ]
    for row in rows:
        nodes = verify_plan(tmp_path / "plan", row["shard"]).read_text().splitlines()
        assert row["a800"] is (expected and nodes == [heavy])
        assert row["timeout-minutes"] == 60

    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "mixed-output"))
    assert write_plan((heavy, ordinary), tmp_path / "mixed-plan", 1) == 1
    mixed = json.loads((tmp_path / "mixed-output").read_text().split("matrix=", 1)[1])
    assert mixed["include"][0]["a800"] is False


def test_serial_embedding_cases_keep_complete_sole_workloads_on_qualified_slots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools.ci import ci_coverage_plan

    targets = ci_coverage_plan._SERIAL_BW_RUNNERS
    maxwell = (
        "tests/integration/test_synthesis_maxwell_notebook.py::"
        "test_fresh_original_degree_and_original_size_workflow"
    )
    nodes = (*targets, maxwell, "tests/test_small.py::test_original")
    monkeypatch.setenv("GITHUB_SHA", "b" * 40)
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    monkeypatch.setattr(
        ci_coverage_plan,
        "load_duration_hints",
        lambda: (0.05, dict.fromkeys((*targets, maxwell), 800.0)),
    )
    directory = tmp_path / "plan"
    assert write_plan(nodes, directory, len(nodes)) == len(nodes)
    rows = json.loads((tmp_path / "output").read_text().split("matrix=", 1)[1])[
        "include"
    ]
    selected = {}
    for row in rows:
        shard = verify_plan(directory, row["shard"]).read_text().splitlines()
        assert len(shard) == 1
        selected[shard[0]] = row
    assert set(selected) == set(nodes)
    qualified = json.loads(ci_coverage_plan._CPU_QUALIFICATION.read_text())
    assert set(targets.values()) <= {
        r["selector"] for r in qualified["qualified_slots"]
    }
    assert len({runner.split("-cpu")[0] for runner in targets.values()}) == len(targets)
    for node, runner in targets.items():
        assert selected[node]["runner"] == runner
        assert selected[node]["exclusive"] is True
        assert selected[node]["a800"] is False
        assert selected[node]["timeout-minutes"] == 60
    assert selected[maxwell]["a800"] is True
    assert selected[maxwell]["runner"] == ""


@pytest.mark.parametrize("missing", ["profile", "slot"])
def test_serial_coverage_rejects_unqualified_resource_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    from tools.ci import ci_coverage_plan

    node = next(iter(ci_coverage_plan._SERIAL_BW_RUNNERS))
    data = json.loads(ci_coverage_plan._CPU_QUALIFICATION.read_text())
    if missing == "profile":
        data["profiles"].remove("coverage-py3.12")
    else:
        data["qualified_slots"] = []
    proof = tmp_path / "qualification.json"
    proof.write_text(json.dumps(data))
    monkeypatch.setattr(ci_coverage_plan, "_CPU_QUALIFICATION", proof)
    monkeypatch.setenv("GITHUB_SHA", "b" * 40)
    monkeypatch.setattr(
        ci_coverage_plan, "load_duration_hints", lambda: (0.05, {node: 800.0})
    )
    with pytest.raises(ValueError, match="verified locked coverage profile"):
        write_plan((node,), tmp_path / "plan", 1)


def test_mixed_partition_retains_original_routing_without_unused_qualification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools.ci import ci_coverage_plan

    nodes = tuple(ci_coverage_plan._SERIAL_BW_RUNNERS)
    monkeypatch.setenv("GITHUB_SHA", "b" * 40)
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    monkeypatch.setattr(
        ci_coverage_plan, "_CPU_QUALIFICATION", tmp_path / "absent.json"
    )
    monkeypatch.setattr(
        ci_coverage_plan,
        "load_duration_hints",
        lambda: (0.05, dict.fromkeys(nodes, 800.0)),
    )
    directory = tmp_path / "plan"
    assert write_plan(nodes, directory, 1) == 1
    assert set(verify_plan(directory, 0).read_text().splitlines()) == set(nodes)
    row = json.loads((tmp_path / "output").read_text().split("matrix=", 1)[1])[
        "include"
    ][0]
    assert row["runner"] in {"flagquantum-bw1100-cpu", "flagquantum-mc550-cpu"}
    assert row["exclusive"] is False
    assert row["a800"] is False


def test_wide_state_oracle_keeps_complete_selection_on_qualified_bw_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools.ci import ci_coverage_plan

    node = (
        "tests/integration/test_synthesis_glued_trees_notebook.py::"
        "test_wide_full_state_at_original_first_time"
    )
    ordinary = "tests/test_small.py::test_original"
    monkeypatch.setenv("GITHUB_SHA", "c" * 40)
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    monkeypatch.setattr(
        ci_coverage_plan,
        "load_duration_hints",
        lambda: (0.05, {node: 166.0}),
    )
    directory = tmp_path / "plan"
    assert write_plan((node, ordinary), directory, 2) == 2
    rows = json.loads((tmp_path / "output").read_text().split("matrix=", 1)[1])[
        "include"
    ]
    selected = {
        tuple(verify_plan(directory, row["shard"]).read_text().splitlines()): row
        for row in rows
    }
    assert set(selected) == {(node,), (ordinary,)}
    row = selected[(node,)]
    assert row["runner"] == "fq-bw1100-17-cpu16-validation"
    assert row["exclusive"] is True
    assert row["a800"] is False
    assert row["timeout-minutes"] == 60
    assert selected[(ordinary,)]["runner"] in {
        "flagquantum-bw1100-cpu",
        "flagquantum-mc550-cpu",
    }
