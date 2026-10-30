"""The GitHub-hosted share one FlagQuantum workflow run is allowed to hold.

flagos-ai runs every repository from one pool of 20 concurrent GitHub-hosted
jobs. A job that joins no concurrency group holds a slot of that pool for as
long as it runs, so a workflow that is unconstrained here is a workflow that
queues every other repository in the organisation behind it.

The ceiling is expressed the way GitHub expresses it. A concurrency group
admits one job at a time, so the number of distinct groups one workflow run
uses *is* the number of GitHub-hosted runners that run can hold at once.
`ci.yml` and `pre-commit.yaml` partition their jobs across the sixteen scoped
buckets below, and this module holds that partition still. The pull-request or
branch scope prevents an unrelated older run from serializing a newer run;
the organisation's 20-runner pool remains the aggregate ceiling.

Six invariants:

1. Every GitHub-hosted Ubuntu job joins one of the buckets, and every bucket
   is used: an unused bucket is a ceiling lower than the comment claims, and
   an undeclared one is a ceiling higher than it.
2. The buckets carry `queue: max`. The default keeps only the newest waiting
   job in a group and cancels the ones behind it, so without it a bucket drops
   work rather than delaying it.
3. The bucket count is the ceiling, so it is asserted rather than counted. The
   `cpucore` bucket carries the matrix value in its name, which means a fourth
   interpreter would be a fourth bucket: the set is spelled out so that is a
   decision about the ceiling and not a side effect of a matrix edit.
4. Both workflows cancel a superseded run of the same pull request, and neither
   cancels a superseded run of `main`. A pull request's older tree is a tree
   nobody will merge; a revision of `main` has already been merged, so cancelling
   its run leaves a merge commit that no gate ever read. On 2026-10-08 two merges
   landed 17 seconds apart and all 23 check runs attached to the first of them
   were `cancelled` (issue #579), so the distinction is asserted here. Both halves
   of the expression are asserted, because a group keyed to the commit for every
   event satisfies "a push is never superseded" while cancelling nothing at all.
5. `publish-dev-container.yml` caps its own matrix instead of joining a bucket,
   because its legs build container images and serializing them all would cost
   more than the slots are worth. It keeps cancelling a superseded run of `main`:
   a superseded container build is superseded work, and no merge gate reads it.
6. `cd.yml` runs only for a release, but its cross-platform wheel matrix is
   still capped at four concurrent builders. Together with the one sdist build,
   a release can hold at most five runners after validation completes.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"

# Workflows whose GitHub-hosted jobs share the bounded runner pool.
BUCKETED_WORKFLOWS = ("ci.yml", "pre-commit.yaml")

# One concurrency group admits one job, so these sixteen scoped names are a
# ceiling of sixteen concurrent GitHub-hosted runners for one run.
#
# The scope is the pull request number where there is one and the branch name
# otherwise, so an unrelated branch's older run cannot serialize a newer one.
# Two pushes to `main` do share these groups, and deliberately so: a push run of
# `main` is no longer cancelled when the next push arrives, which means its jobs
# have to wait somewhere instead. `queue: max` turns that group into a queue
# rather than a drop, so the later revision is read after the earlier one instead
# of deleting it.
RUN_SCOPE = "${{ github.event.pull_request.number || github.ref_name }}"
BUCKETS = frozenset(
    {
        f"flagquantum-gh-{RUN_SCOPE}-cpucore-3.10",
        f"flagquantum-gh-{RUN_SCOPE}-cpucore-3.11",
        f"flagquantum-gh-{RUN_SCOPE}-runtime",
        f"flagquantum-gh-{RUN_SCOPE}-cpucore-3.12",
        f"flagquantum-gh-{RUN_SCOPE}-coverage-0",
        f"flagquantum-gh-{RUN_SCOPE}-coverage-1",
        f"flagquantum-gh-{RUN_SCOPE}-coverage-2",
        f"flagquantum-gh-{RUN_SCOPE}-coverage-3",
        f"flagquantum-gh-{RUN_SCOPE}-coverage-4",
        f"flagquantum-gh-{RUN_SCOPE}-coverage-5",
        f"flagquantum-gh-{RUN_SCOPE}-coverage-6",
        f"flagquantum-gh-{RUN_SCOPE}-coverage-7",
        f"flagquantum-gh-{RUN_SCOPE}-distributed",
        f"flagquantum-gh-{RUN_SCOPE}-light-a",
        f"flagquantum-gh-{RUN_SCOPE}-light-b",
        f"flagquantum-gh-{RUN_SCOPE}-light-c",
    }
)

CEILING = 16
TRUSTED_CPU_ROUTE = "${{ (github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository) && 'flagquantum-cpu-general' || 'ubuntu-latest' }}"
A800_CPU_ROUTE = '${{ fromJSON((github.event_name != \'pull_request\' || github.event.pull_request.head.repo.full_name == github.repository) && \'["self-hosted","linux","x64","flagquantum-cpu"]\' || \'["ubuntu-latest"]\') }}'
BOUNDED_PYTEST_ADDOPTS = "-n 16 --dist=worksteal --durations=50"
GITHUB_HOSTED_UBUNTU_RUNNERS = frozenset({"ubuntu-latest", "ubuntu-22.04"})

_MATRIX_REFERENCE = re.compile(r"\$\{\{\s*matrix\.([A-Za-z0-9_-]+)\s*\}\}")


def _document(workflow: str) -> dict[str, object]:
    document = yaml.safe_load((WORKFLOWS / workflow).read_text(encoding="utf-8"))
    assert isinstance(document, dict), f"{workflow}: not a workflow document"
    return document


def _github_hosted_jobs(workflow: str) -> dict[str, dict[str, object]]:
    """Every job of the workflow that asks GitHub for a runner."""

    jobs = _document(workflow).get("jobs")
    assert isinstance(jobs, dict), f"{workflow}: no jobs mapping"
    found: dict[str, dict[str, object]] = {}
    for name, job in jobs.items():
        assert isinstance(job, dict), f"{workflow}:{name}: not a job mapping"
        if isinstance(job.get("runs-on"), str) and (
            job["runs-on"] in GITHUB_HOSTED_UBUNTU_RUNNERS
            or "'ubuntu-latest'" in job["runs-on"]
            or '["ubuntu-latest"]' in job["runs-on"]
        ):
            found[name] = job
    return found


def _groups(job: dict[str, object]) -> set[str]:
    """The group names a job can produce, with its matrix reference resolved.

    The cpucore group is one bucket per interpreter, so the name in the
    workflow is not the name GitHub sees. The matrix reference is resolved
    against the values the same job declares, which is also what makes a
    widened matrix show up here as a widened ceiling. The run scope is retained
    verbatim because it is the boundary this policy is asserting.
    """

    concurrency = job.get("concurrency")
    if not isinstance(concurrency, dict):
        return set()
    group = concurrency.get("group")
    if not isinstance(group, str):
        return set()
    # Evaluate the hosted fallback of both trusted-pool selectors. Their CPU
    # arms have independent groups and do not consume hosted capacity.
    group = re.sub(
        r"\$\{\{ \(github\.event_name != 'pull_request' \|\| "
        r"github\.event\.pull_request\.head\.repo\.full_name == github\.repository\) "
        r"&& format\('(?:a100-cpu|self-hosted)-\{0\}-[a-z0-9-]+-\{1\}', "
        r"github\.run_id, strategy\.job-index \|\| 0\) \|\| 'gh' \}\}",
        "gh",
        group,
    )
    reference = _MATRIX_REFERENCE.search(group)
    if reference is None:
        return {group}
    strategy = job.get("strategy")
    matrix = strategy.get("matrix") if isinstance(strategy, dict) else None
    assert isinstance(matrix, dict), f"{group}: job declares no matrix"
    values = matrix.get(reference.group(1))
    if values is None and isinstance(matrix.get("include"), list):
        values = [row[reference.group(1)] for row in matrix["include"]]
    assert isinstance(
        values, list
    ), f"{group}: matrix {reference.group(1)} is not a list"
    return {_MATRIX_REFERENCE.sub(str(value), group) for value in values}


def _bucketed_jobs() -> dict[str, dict[str, object]]:
    found: dict[str, dict[str, object]] = {}
    for workflow in BUCKETED_WORKFLOWS:
        for name, job in _github_hosted_jobs(workflow).items():
            found[f"{workflow}:{name}"] = job
    return found


def _steps(workflow: str, job_name: str) -> list[dict[str, object]]:
    job = _document(workflow)["jobs"][job_name]
    steps = job.get("steps")
    assert isinstance(steps, list), f"{workflow}:{job_name}: no steps"
    assert all(isinstance(step, dict) for step in steps)
    return steps


def test_the_scan_finds_the_github_hosted_jobs_it_is_meant_to_police() -> None:
    # A policy scan that finds nothing passes every rule below while checking
    # nothing, so the discovery itself is asserted.
    found = set(_bucketed_jobs())

    assert {
        "ci.yml:quality",
        "ci.yml:cpu-core-3-12",
        "ci.yml:coverage",
        "ci.yml:package",
        "pre-commit.yaml:pre-commit",
    } <= found


def test_every_github_hosted_job_joins_a_bucket() -> None:
    offenders: list[str] = []
    for key, job in sorted(_bucketed_jobs().items()):
        groups = _groups(job)
        if not groups:
            offenders.append(f"{key} declares no concurrency group")
        for group in sorted(groups - BUCKETS):
            offenders.append(f"{key} joins {group!r}")
    assert not offenders, (
        "every GitHub-hosted job must join one of the scoped buckets, so that "
        "one run's share of the organisation's runner pool stays bounded:\n"
        + "\n".join(offenders)
    )


def test_a_bucket_delays_a_waiting_job_instead_of_dropping_it() -> None:
    offenders: list[str] = []
    for key, job in sorted(_bucketed_jobs().items()):
        concurrency = job.get("concurrency")
        if not isinstance(concurrency, dict) or concurrency.get("queue") != "max":
            offenders.append(key)
    assert not offenders, (
        "a bucket without `queue: max` keeps only the newest waiting job and "
        "cancels the ones behind it, which reports nothing at all for the job "
        "that was dropped:\n" + "\n".join(offenders)
    )


def test_the_buckets_in_use_are_exactly_the_declared_ceiling() -> None:
    used: set[str] = set()
    for job in _bucketed_jobs().values():
        used |= _groups(job)

    assert len(BUCKETS) == CEILING, (
        f"the declared buckets are {len(BUCKETS)}, not the {CEILING} the "
        "capacity note in ci.yml states"
    )
    assert used == BUCKETS, (
        f"unused buckets {sorted(BUCKETS - used)}; "
        f"undeclared buckets {sorted(used - BUCKETS)}"
    )


def test_superseded_prs_cancel_without_cancelling_manual_or_main_validation() -> None:
    ci = _document("ci.yml")
    assert ci["concurrency"] == {
        "group": "flagquantum-ci-${{ github.event.pull_request.number || github.run_id }}",
        "cancel-in-progress": "${{ github.event_name == 'pull_request' }}",
    }
    smoke = _document("main-smoke.yml")
    assert "concurrency" not in smoke
    lock = smoke["jobs"]["installed-wheel-smoke"]["concurrency"]
    assert lock["queue"] == "max"
    assert lock.get("cancel-in-progress", False) is False


def test_cpu_pytest_tiers_use_sixteen_bounded_work_stealing_workers() -> None:
    assert [
        name
        for name, definition in _document("ci.yml")["jobs"].items()
        if not definition.get("needs")
    ] == ["pre-commit", "prepare-source"]
    tier_steps = []
    for job in ("cpu-core", "cpu-core-3-12-tests", "cpu-runtime"):
        definition = _document("ci.yml")["jobs"][job]
        assert '"flagquantum-cpu"' in definition["runs-on"]
        assert (
            "github.event.pull_request.head.repo.full_name == github.repository"
            in definition["runs-on"]
        )
        assert '["ubuntu-latest"]' in definition["runs-on"]
        assert definition["timeout-minutes"] == 30
        setup_python = next(
            step
            for step in _steps("ci.yml", job)
            if step.get("uses") == "actions/setup-python@v5"
        )
        assert "cache" not in setup_python["with"]
        assert definition["needs"] == "prepare-source"
        download = next(
            step
            for step in _steps("ci.yml", job)
            if step.get("uses") == "actions/download-artifact@v8.0.2" and step.get("if")
        )
        assert download["with"]["digest-mismatch"] == "error"
        assert (
            download["with"]["name"]
            == "ci-source-${{ needs.prepare-source.outputs.artifact-key }}"
        )
        assert not any(
            step.get("uses", "").startswith("actions/checkout@")
            for step in _steps("ci.yml", job)
        )
        assert definition["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert definition["env"]["OMP_NUM_THREADS"] == "1"
        assert definition["env"]["PIP_INDEX_URL"] == (
            "${{ (github.event_name != 'pull_request' || "
            "github.event.pull_request.head.repo.full_name == github.repository) && "
            "'https://pypi.tuna.tsinghua.edu.cn/simple' || 'https://pypi.org/simple' }}"
        )
        commands = "\n".join(str(step.get("run", "")) for step in _steps("ci.yml", job))
        assert "assert torch.version.cuda is None" in commands
        assert "python -m pip check" in commands
        steps = _steps("ci.yml", job)
        installer = next(
            step
            for step in steps
            if step.get("uses") == "./.github/actions/install-cpu-torch"
        )
        assert installer["with"]["torch-spec"] == "torch>=2.13,<2.14"
        project_install = next(
            step
            for step in steps
            if "python -m pip install --no-build-isolation" in step.get("run", "")
        )
        assert steps.index(installer) < steps.index(project_install)
        assert 'python -m venv "${RUNNER_TEMP}/venv-' in commands
        assert '>> "$GITHUB_PATH"' in commands
        assert 'test "$sha" = "$GITHUB_SHA"' in commands
        assert 'git checkout --force --detach "$sha"' in commands
        assert 'test "$(git rev-parse HEAD)" = "$GITHUB_SHA"' in commands
        tier_steps.extend(
            step
            for step in _steps("ci.yml", job)
            if "pytest -m smoke" in str(step.get("run", ""))
            or "ci_tier.py pr-" in str(step.get("run", ""))
        )
    assert len(tier_steps) == 4
    for step in tier_steps:
        env = step.get("env")
        assert isinstance(env, dict)
        assert env.get("PYTEST_ADDOPTS") == (
            "-n ${{ env.FQ_PYTEST_WORKERS }} --dist=worksteal --durations=50 "
            "--junitxml=${{ env.FQ_CI_REPORT_DIR }}/pytest.xml"
        )
    from tools.ci_cpu_budget import GIB, select_workers

    assert select_workers(24, 64 * GIB, shared=True, dedicated=True) == 16
    assert select_workers(24, 64 * GIB, shared=False, dedicated=True) == 2


@pytest.mark.parametrize(
    "job",
    [pytest.param("distributed-cpu-shard", id="distributed-cpu"), "coverage-shard"],
)
def test_migrated_cpu_jobs_keep_frameworks_and_processes_isolated(job: str) -> None:
    definition = _document("ci.yml")["jobs"][job]
    assert definition["needs"] == "prepare-source"
    assert definition["env"]["CUDA_VISIBLE_DEVICES"] == ""
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert definition["env"][variable] == "1"
    steps = _steps("ci.yml", job)
    commands = "\n".join(str(step.get("run", "")) for step in steps)
    assert "python -m pip check" in commands
    assert "torch.version.cuda is None" in commands
    assert "jax.default_backend() == 'cpu'" in commands
    assert definition["runs-on"] == A800_CPU_ROUTE
    assert definition["timeout-minutes"] == (
        60 if job == "distributed-cpu-shard" else 30
    )
    assert any(
        step.get("uses") == "./.github/actions/isolated-python" for step in steps
    )
    if job == "coverage-shard":
        assert [
            row["shard"] for row in definition["strategy"]["matrix"]["include"]
        ] == list(range(16))
    else:
        assert "python tools/ci_tier.py pr-distributed" in commands


def test_dependency_bounds_retain_the_full_matrix_on_isolated_cpu_runners() -> None:
    definition = _document("ci.yml")["jobs"]["dependency-bounds"]
    assert definition["strategy"]["fail-fast"] is False
    assert len(definition["strategy"]["matrix"]["include"]) == 7
    assert definition["env"]["CUDA_VISIBLE_DEVICES"] == ""
    steps = _steps("ci.yml", "dependency-bounds")
    commands = "\n".join(str(step.get("run", "")) for step in steps)
    assert "torch.version.cuda is None" in commands
    assert "python -m pytest tests/smoke tests/test_native_circuit.py -q" in commands
    assert "--print-floor" in commands
    assert "mypy --strict" in commands
    assert definition["runs-on"] == TRUSTED_CPU_ROUTE
    assert "max-parallel" not in definition["strategy"]
    isolated = next(
        step
        for step in steps
        if step.get("uses") == "./.github/actions/isolated-python"
    )
    assert isolated["with"]["torch-spec"] == "${{ matrix.torch }}"


def test_pull_requests_use_compatibility_smoke_on_older_python() -> None:
    document = _document("ci.yml")
    jobs = document.get("jobs")
    assert isinstance(jobs, dict)
    core = jobs["cpu-core"]
    assert isinstance(core, dict)
    assert core["strategy"]["matrix"] == {"python-version": ["3.10", "3.11"]}

    commands = {
        str(step.get("if", "")): str(step.get("run", ""))
        for step in _steps("ci.yml", "cpu-core")
        if "pytest -m smoke" in str(step.get("run", ""))
        or "ci_tier.py pr-default" in str(step.get("run", ""))
    }
    assert commands == {
        "github.event_name == 'pull_request'": "python -m pytest -m smoke -q",
        "github.event_name != 'pull_request'": "python tools/ci_tier.py pr-default",
    }


def test_python_312_required_check_aggregates_parallel_suites() -> None:
    document = _document("ci.yml")
    jobs = document.get("jobs")
    assert isinstance(jobs, dict)
    final = jobs["cpu-core-3-12"]
    assert isinstance(final, dict)
    assert final.get("name") == "cpu-core (3.12)"
    assert set(final.get("needs", ())) == {"cpu-core-3-12-tests", "cpu-runtime"}
    command = "\n".join(
        str(step.get("run", "")) for step in _steps("ci.yml", "cpu-core-3-12")
    )
    assert "needs.cpu-core-3-12-tests.result" in command
    assert "needs.cpu-runtime.result" in command


def test_expensive_cpu_runtime_proofs_run_once_on_the_newest_python() -> None:
    expensive_steps = [
        step
        for step in _steps("ci.yml", "cpu-runtime")
        if "ci_tier.py pr-runtime" in str(step.get("run", ""))
        or "torchrun" in str(step.get("run", ""))
    ]
    assert len(expensive_steps) == 3

    document = _document("ci.yml")
    jobs = document.get("jobs")
    assert isinstance(jobs, dict)
    runtime = jobs["cpu-runtime"]
    assert isinstance(runtime, dict)
    setup = next(
        step
        for step in runtime["steps"]
        if isinstance(step, dict) and step.get("uses") == "actions/setup-python@v5"
    )
    assert setup["with"]["python-version"] == "3.12"


def test_launched_distributed_steps_never_inherit_xdist_workers() -> None:
    launched = [
        step
        for step in _steps("ci.yml", "cpu-runtime")
        if "torchrun" in str(step.get("run", ""))
    ]
    assert launched
    for step in launched:
        assert "-n 2" not in str(step.get("run", ""))
        env = step.get("env")
        assert not isinstance(env, dict) or "PYTEST_ADDOPTS" not in env


def test_coverage_uses_the_same_bounded_work_stealing_policy() -> None:
    steps = [
        step
        for step in _steps("ci.yml", "coverage-shard")
        if "python -m pytest" in str(step.get("run", ""))
    ]
    assert len(steps) == 2
    for step in steps:
        command = str(step.get("run", ""))
        for token in (
            "-n ${{ env.FQ_PYTEST_WORKERS }}",
            "--dist=worksteal",
            "--durations=50",
        ):
            assert token in command
        for token in (
            "-p tools.pytest_shard",
            "--fq-shard-count 16",
            "--fq-shard-index ${{ matrix.shard }}",
        ):
            assert token in command


def test_pull_request_coverage_deduplicates_full_slow_conformance() -> None:
    steps = {
        str(step.get("if", "")): str(step.get("run", ""))
        for step in _steps("ci.yml", "coverage-shard")
        if "python -m pytest" in str(step.get("run", ""))
    }
    assert set(steps) == {
        "github.event_name == 'pull_request'",
        "github.event_name != 'pull_request'",
    }
    assert "and not slow" in steps["github.event_name == 'pull_request'"]
    assert "and not slow" not in steps["github.event_name != 'pull_request'"]


def test_coverage_shards_are_combined_before_the_policy_is_enforced() -> None:
    shard = _document("ci.yml")["jobs"]["coverage-shard"]
    strategy = shard.get("strategy")
    assert isinstance(strategy, dict)
    assert strategy.get("matrix") == {
        "include": [{"shard": i, "hosted-bucket": i % 8} for i in range(16)]
    }

    final = _github_hosted_jobs("ci.yml")["coverage"]
    assert final.get("needs") == "coverage-shard"
    commands = "\n".join(
        str(step.get("run", "")) for step in _steps("ci.yml", "coverage")
    )
    assert "needs.coverage-shard.result" in commands
    download = next(
        step
        for step in _steps("ci.yml", "coverage")
        if step.get("uses") == "actions/download-artifact@v8.0.2"
    )
    assert download["with"]["digest-mismatch"] == "error"
    assert commands.index("rm -rf coverage-data") < commands.index("integrity_check")
    assert commands.index("integrity_check") < commands.index(
        "coverage combine coverage-data"
    )
    assert "coverage combine coverage-data" in commands
    assert commands.index("coverage combine coverage-data") < commands.index(
        "tools/check_coverage.py"
    )


@pytest.mark.parametrize("case", ["complete", "missing", "corrupt", "unexpected"])
def test_coverage_download_validation_refuses_partial_or_corrupt_inputs(tmp_path, case):
    directory = tmp_path / "coverage-data"
    directory.mkdir()
    for i in range(16):
        with sqlite3.connect(directory / f".coverage.{i}") as connection:
            connection.execute("CREATE TABLE sample (value INTEGER)")
            connection.execute("INSERT INTO sample VALUES (1)")
    if case == "missing":
        (directory / ".coverage.2").unlink()
    elif case == "corrupt":
        (directory / ".coverage.2").write_bytes(b"truncated sqlite database")
    elif case == "unexpected":
        (directory / ".coverage.16").write_bytes(b"stale prior shard")
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    step = next(
        step
        for step in _steps("ci.yml", "coverage")
        if step.get("name") == "Validate complete coverage inputs"
    )
    script = step["run"].split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if case == "complete":
        assert result.returncode == 0, result.stderr
        assert "Validated all 16" in result.stdout
    else:
        assert result.returncode != 0
        assert (
            "invalid coverage input .coverage.2"
            if case == "corrupt"
            else "coverage inputs differ"
        ) in result.stderr
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


def test_each_optional_compatibility_matrix_reuses_one_runner() -> None:
    jobs = _github_hosted_jobs("ci.yml")
    expected = {
        "cirq-optional": "1.6.1 1.7.0",
        "braket-optional": "1.117.0 1.127.1",
        "cudaq-optional": "0.15.1 0.16.0.post1",
        "pennylane-optional": "0.44.1 0.45.1",
    }
    for name, versions in expected.items():
        job = jobs[name]
        assert "strategy" not in job
        env = job.get("env")
        assert isinstance(env, dict)
        assert env.get("OPTIONAL_VERSIONS") == versions
        commands = "\n".join(
            str(step.get("run", "")) for step in _steps("ci.yml", name)
        )
        assert "for version in ${OPTIONAL_VERSIONS}" in commands


def test_no_github_hosted_lane_uses_unbounded_auto_workers() -> None:
    for workflow in BUCKETED_WORKFLOWS:
        text = (WORKFLOWS / workflow).read_text(encoding="utf-8")
        assert "-n auto" not in text


def test_the_container_matrix_does_not_land_all_at_once() -> None:
    jobs = _document("publish-dev-container.yml").get("jobs")
    assert isinstance(jobs, dict), "publish-dev-container.yml: no jobs mapping"
    matrices = [
        job["strategy"]
        for job in jobs.values()
        if isinstance(job, dict)
        and isinstance(job.get("strategy"), dict)
        and job["strategy"].get("matrix")
    ]
    assert matrices, "publish-dev-container.yml declares no matrix to police"
    for strategy in matrices:
        assert strategy.get("max-parallel") == 2, (
            "the container matrix is not capped, so one push to main holds a "
            "runner per variant"
        )


def test_the_release_matrix_leaves_capacity_for_other_repositories() -> None:
    jobs = _document("cd.yml")["jobs"]
    wheel_strategy = jobs["build-wheels"]["strategy"]
    wheel_parallelism = wheel_strategy["max-parallel"]

    assert wheel_parallelism == 4
    assert wheel_parallelism + 1 <= 5  # one concurrent sdist build


def test_full_ci_runs_once_per_pr_and_remains_manually_available() -> None:
    ci = _document("ci.yml")
    events = ci.get("on", ci.get(True))
    assert set(events) == {"pull_request", "workflow_dispatch"}
    assert events["pull_request"] == {
        "branches": ["main", "refactor/qec-restructure-*"]
    }
    manual = _document("pre-commit.yaml")
    assert set(manual.get("on", manual.get(True))) == {"workflow_dispatch"}
    gpu = _document("local-gpu.yml")
    assert set(gpu.get("on", gpu.get(True))) == {"pull_request", "workflow_dispatch"}
    assert _document("main-smoke.yml").get(True)["push"] == {"branches": ["main"]}


@pytest.mark.parametrize(
    "step_name,tool,extra",
    [
        (
            "Require every changed path to have an owner",
            "tools/check_team_scope.py",
            ["--require-classified"],
        ),
        (
            "Require retention metadata for new benchmark evidence",
            "tools/check_evidence_retention.py",
            [],
        ),
    ],
)
@pytest.mark.parametrize(
    "event,base,expected_code",
    [
        ("merge_group", "a" * 40, 0),
        ("merge_group", "", 1),
        ("merge_group", "0" * 40, 1),
        ("pull_request", "a" * 40, 0),
        ("push", "a" * 40, 0),
        ("push", "0" * 40, 0),
    ],
)
def test_change_gates_use_the_queue_base_and_refuse_a_missing_queue_base(
    tmp_path: Path,
    step_name: str,
    tool: str,
    extra: list[str],
    event: str,
    base: str,
    expected_code: int,
) -> None:
    step = next(s for s in _steps("ci.yml", "quality") if s.get("name") == step_name)
    assert step["env"]["BASE_SHA"] == (
        "${{ github.event.merge_group.base_sha || "
        "github.event.pull_request.base.sha || github.event.before }}"
    )
    assert step["env"]["EVENT_NAME"] == "${{ github.event_name }}"
    executable = tmp_path / "python"
    executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$FQ_CI_ARGS"\n')
    executable.chmod(0o755)
    calls = tmp_path / "calls"
    completed = subprocess.run(
        ["/bin/bash", "-eu", "-c", step["run"]],
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ.get("PATH", ""),
            "BASE_SHA": base,
            "EVENT_NAME": event,
            "FQ_CI_ARGS": str(calls),
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == expected_code, completed.stderr
    if base and base != "0" * 40:
        assert calls.read_text().splitlines() == [tool, *extra, "--base", base]
    else:
        assert not calls.exists()


def test_queue_uses_full_integration_suites_and_preserves_required_check_names() -> (
    None
):
    jobs = _document("ci.yml")["jobs"]
    assert all(job.get("if") in (None, "${{ always() }}") for job in jobs.values())
    core_steps = _steps("ci.yml", "cpu-core")
    full = next(
        s for s in core_steps if s.get("run") == "python tools/ci_tier.py pr-default"
    )
    assert full["if"] == "github.event_name != 'pull_request'"
    coverage = next(
        s
        for s in _steps("ci.yml", "coverage-shard")
        if s.get("if") == "github.event_name != 'pull_request'"
    )
    assert "and not slow" not in coverage["run"]
    contexts = {job.get("name", key) for key, job in jobs.items()}
    contexts |= {f"cpu-core ({version})" for version in ("3.10", "3.11")}
    contexts.add("pre-commit")
    assert {
        "quality",
        "cpu-core (3.10)",
        "cpu-core (3.11)",
        "cpu-core (3.12)",
        "package",
        "distributed-cpu",
        "coverage",
        "pre-commit",
    } <= contexts


def test_pr_gate_template_preserves_strict_checks_without_a_merge_queue() -> None:
    config = json.loads((ROOT / "ci/pr-gate-main.json").read_text())
    assert config["enforcement"] == "active"
    assert config["target"] == "branch"
    assert config["conditions"] == {
        "ref_name": {"include": ["refs/heads/main"], "exclude": []}
    }
    assert config["bypass_actors"] == []
    assert config["rules"] == [
        {
            "type": "required_status_checks",
            "parameters": {
                "required_status_checks": [
                    {"context": "CI complete", "integration_id": 15368}
                ],
                "strict_required_status_checks_policy": True,
            },
        }
    ]


@pytest.mark.parametrize("result", ["success", "failure", "cancelled", "skipped"])
@pytest.mark.parametrize(
    "upstream", ["cpu-core-3-12-tests", "cpu-runtime", "coverage-shard"]
)
def test_integration_aggregation_never_accepts_an_unsuccessful_suite(
    result: str, upstream: str
) -> None:
    job = "coverage" if upstream == "coverage-shard" else "cpu-core-3-12"
    command = _steps("ci.yml", job)[0]["run"]
    for name in ("cpu-core-3-12-tests", "cpu-runtime", "coverage-shard"):
        command = command.replace(
            "${{ needs." + name + ".result }}",
            result if name == upstream else "success",
        )
    completed = subprocess.run(
        ["/bin/bash", "-eu", "-c", command],
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert (completed.returncode == 0) == (result == "success")


@pytest.mark.parametrize("result", ["success", "failure", "cancelled", "skipped"])
def test_merge_gate_waits_for_every_ci_job_including_source_hygiene(
    result: str,
) -> None:
    jobs = _document("ci.yml")["jobs"]
    complete = jobs["ci-complete"]
    assert complete["name"] == "CI complete"
    assert complete["if"] == "${{ always() }}"
    assert set(complete["needs"]) == set(jobs) - {"ci-complete"}
    step = complete["steps"][0]
    assert step["env"]["NEEDS_JSON"] == "${{ toJSON(needs) }}"
    # Real shell execution catches failures in any optional, matrix or package
    # dependency instead of merely finding a success token in the YAML.
    for changed in complete["needs"]:
        outcomes = {name: {"result": "success"} for name in complete["needs"]}
        outcomes[changed]["result"] = result
        completed = subprocess.run(
            ["/bin/bash", "-eu", "-c", step["run"]],
            env={**os.environ, "NEEDS_JSON": json.dumps(outcomes)},
            capture_output=True,
            timeout=10,
            check=False,
        )
        assert (completed.returncode == 0) == (result == "success"), changed
    completed = subprocess.run(
        ["/bin/bash", "-eu", "-c", step["run"]],
        env={**os.environ, "NEEDS_JSON": "{}"},
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode != 0


def test_distributed_matrix_runs_both_complete_selections_and_one_rank_probe() -> None:
    job = _document("ci.yml")["jobs"]["distributed-cpu-shard"]
    rows = job["strategy"]["matrix"]["include"]
    assert rows == [
        {"command": command, "shard": shard, "shards": count}
        for command, count in [(0, 4), (1, 2)]
        for shard in range(count)
    ]
    assert job["strategy"]["fail-fast"] is False
    commands = "\n".join(str(s.get("run", "")) for s in job["steps"])
    assert "--command-index ${{ matrix.command }}" in commands
    assert "--shard-count ${{ matrix.shards }}" in commands
    assert "--shard-index ${{ matrix.shard }}" in commands
    probe = next(
        s for s in job["steps"] if str(s.get("run", "")).startswith("torchrun")
    )
    assert probe["if"] == "matrix.command == 0 && matrix.shard == 0"
    assert sum(row["command"] == 0 and row["shard"] == 0 for row in rows) == 1
    assert "--nproc-per-node=2" in probe["run"]


def test_distributed_cpu_retains_its_stable_check_and_rejects_a_failed_partition() -> (
    None
):
    job = _document("ci.yml")["jobs"]["distributed-cpu"]
    assert job["if"] == "${{ always() }}"
    assert job["needs"] == "distributed-cpu-shard"
    assert job["steps"] == [
        {
            "name": "Require every partition and the two-rank probe to pass",
            "run": "test '${{ needs.distributed-cpu-shard.result }}' = success",
        }
    ]
