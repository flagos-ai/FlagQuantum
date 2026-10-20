"""Verify that CI respects affinity and nested cgroup limits before fan-out."""

from pathlib import Path

import pytest

from tools.ci_cpu_budget import GIB, controller_directories, read_limits, select_workers

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("cpus", "memory_gib", "shared", "expected"),
    [
        (64, 128, True, 8),
        (12, 128, True, 6),
        (3.5, 128, True, 1),
        (0.5, 128, True, 1),
        (64, 12, True, 3),
        (64, 128, False, 2),
        (1, 128, False, 1),
    ],
)
def test_worker_budget_shares_cpu_and_memory_between_registrations(
    cpus: float,
    memory_gib: int,
    shared: bool,
    expected: int,
) -> None:
    assert select_workers(cpus, memory_gib * GIB, shared) == expected


def test_nested_v2_mount_retains_stricter_parent_quota(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    mount = tmp_path / "cgroup"
    job = mount / "runner" / "job"
    job.mkdir(parents=True)
    (proc / "self/cgroup").write_text("0::/tenant/runner/job\n")
    (proc / "self/mountinfo").write_text(
        f"27 23 0:24 /tenant {mount} rw - cgroup2 cgroup rw\n"
    )
    (job / "cpu.max").write_text("max 100000")
    (job.parent / "cpu.max").write_text("350000 100000")
    (mount / "cpu.max").write_text("800000 100000")
    (job / "memory.max").write_text("max")
    (job.parent / "memory.max").write_text(str(12 * GIB))
    paths = controller_directories(proc)
    assert paths["cpu"] == [job, job.parent, mount]
    cpu, memory = read_limits(paths)
    assert min(cpu) == 3.5
    assert min(memory) == 12 * GIB
    assert select_workers(min(cpu), min(memory), shared=True) == 1


def test_v1_cpu_and_memory_controllers_are_resolved_separately(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    cpu_mount, memory_mount = tmp_path / "cpu", tmp_path / "memory"
    cpu, memory = cpu_mount / "job", memory_mount / "job"
    cpu.mkdir(parents=True)
    memory.mkdir(parents=True)
    (proc / "self/cgroup").write_text("2:cpu,cpuacct:/job\n3:memory:/job\n")
    (proc / "self/mountinfo").write_text(
        f"27 23 0:24 / {cpu_mount} rw - cgroup cgroup rw,cpu,cpuacct\n"
        f"28 23 0:25 / {memory_mount} rw - cgroup cgroup rw,memory\n"
    )
    (cpu / "cpu.cfs_quota_us").write_text("600000")
    (cpu / "cpu.cfs_period_us").write_text("100000")
    (memory / "memory.limit_in_bytes").write_text(str(32 * GIB))
    quotas, limits = read_limits(controller_directories(proc))
    assert quotas == [6]
    assert limits == [32 * GIB]


def test_unreadable_cpu_quota_fails_instead_of_assuming_unlimited(
    tmp_path: Path,
) -> None:
    (tmp_path / "cpu.max").write_text("invalid quota")
    with pytest.raises(RuntimeError, match="Cannot determine CPU quota"):
        read_limits({"cpu": [tmp_path], "memory": []})


def test_namespaced_root_still_reads_visible_mount_limit(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    mount = tmp_path / "cgroup"
    mount.mkdir()
    (proc / "self/cgroup").write_text("0::/\n")
    (proc / "self/mountinfo").write_text(
        f"27 23 0:24 /tenant/runner {mount} rw - cgroup2 cgroup rw\n"
    )
    (mount / "cpu.max").write_text("200000 100000")
    quotas, _ = read_limits(controller_directories(proc))
    assert quotas == [2]


def test_probe_uses_affinity_quota_memory_and_exports_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    from tools import ci_cpu_budget as budget

    monkeypatch.setattr(
        budget.os, "sched_getaffinity", lambda _: set(range(32)), raising=False
    )
    monkeypatch.setattr(
        budget.os,
        "sysconf",
        lambda key: 4096 if key == "SC_PAGE_SIZE" else 32 * GIB // 4096,
    )
    monkeypatch.setattr(
        budget,
        "controller_directories",
        lambda: {"cpu": [tmp_path], "memory": [tmp_path]},
    )
    (tmp_path / "cpu.max").write_text("1200000 100000")
    (tmp_path / "memory.max").write_text(str(16 * GIB))
    env = tmp_path / "environment"
    monkeypatch.setenv("GITHUB_ENV", str(env))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary"))
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setenv("RUNNER_ENVIRONMENT", "self-hosted")
    budget.main()
    report = json.loads(
        next(tmp_path.glob("fq-ci-report-*/resources.json")).read_text()
    )
    assert report["effective_cpus"] == 12
    assert report["workers"] == 4  # The smaller memory budget wins.
    assert "FQ_PYTEST_WORKERS=4\n" in env.read_text()
    assert "OMP_NUM_THREADS=1\n" in env.read_text()


@pytest.mark.parametrize(
    ("cpus", "memory_gib", "shared", "expected"),
    [(31, 125, True, 2), (8, 125, True, 1), (31, 16, True, 1), (24, 64, False, 1)],
)
def test_launcher_budget_accounts_for_nested_eight_rank_processes(
    cpus: float,
    memory_gib: int,
    shared: bool,
    expected: int,
) -> None:
    from tools.ci_cpu_budget import select_distributed_workers

    assert select_distributed_workers(cpus, memory_gib * GIB, shared) == expected
