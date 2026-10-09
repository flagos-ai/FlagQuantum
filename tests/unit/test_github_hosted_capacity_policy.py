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

Five invariants:

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
   `cd.yml` is out of scope: it publishes on a release and holds one job, so it
   cannot occupy more than one runner.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"

# The per-push workflows: every job in them runs on a GitHub-hosted runner.
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
        f"flagquantum-gh-{RUN_SCOPE}-runtime",
        f"flagquantum-gh-{RUN_SCOPE}-light-a",
        f"flagquantum-gh-{RUN_SCOPE}-light-b",
        f"flagquantum-gh-{RUN_SCOPE}-light-c",
    }
)

CEILING = 16
BOUNDED_PYTEST_ADDOPTS = "-n 2 --dist=worksteal --durations=50"
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
        if job.get("runs-on") in GITHUB_HOSTED_UBUNTU_RUNNERS:
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
    reference = _MATRIX_REFERENCE.search(group)
    if reference is None:
        return {group}
    strategy = job.get("strategy")
    matrix = strategy.get("matrix") if isinstance(strategy, dict) else None
    assert isinstance(matrix, dict), f"{group}: job declares no matrix"
    values = matrix.get(reference.group(1))
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
    job = _github_hosted_jobs(workflow)[job_name]
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
        "ci.yml:cpu-core",
        "ci.yml:cpu-core-3-12-tests",
        "ci.yml:cpu-runtime",
        "ci.yml:cpu-core-3-12",
        "ci.yml:coverage-shard",
        "ci.yml:coverage",
        "ci.yml:package",
        "ci.yml:distributed-cpu",
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


def test_a_superseded_pull_request_run_is_cancelled_and_a_main_push_is_not() -> None:
    offenders: list[str] = []
    for workflow in BUCKETED_WORKFLOWS:
        concurrency = _document(workflow).get("concurrency")
        if not isinstance(concurrency, dict):
            offenders.append(f"{workflow} declares no workflow-level concurrency")
            continue
        group = str(concurrency.get("group", ""))
        # The two halves have to be asserted together. A group keyed to
        # `github.sha` for every event would satisfy "a push is not superseded"
        # while also giving every push to a branch under review its own group, so
        # `cancel-in-progress` would never cancel anything and each push would
        # queue a full matrix behind the last.
        if "github.sha" not in group:
            offenders.append(
                f"{workflow} keys a push run to its branch, so the next push to "
                "main supersedes a revision that is already merged"
            )
        if "github.event_name == 'push'" not in group:
            offenders.append(
                f"{workflow} does not keep pull request runs in one group, so a "
                "push to a branch under review queues a matrix that is never read"
            )
        if concurrency.get("cancel-in-progress") != (
            "${{ github.event_name == 'pull_request' }}"
        ):
            offenders.append(
                f"{workflow} does not scope cancellation to the pull request"
            )
    assert not offenders, (
        "a push to a branch under review must cancel that branch's superseded "
        "run, and a push to `main` must not cancel the run of a revision that is "
        "already merged, because a cancelled merge-commit reading is "
        "indistinguishable from a passing one:\n" + "\n".join(offenders)
    )


def test_cpu_pytest_tiers_use_two_bounded_work_stealing_workers() -> None:
    tier_steps = []
    for job in ("cpu-core", "cpu-core-3-12-tests", "cpu-runtime"):
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
        assert env.get("PYTEST_ADDOPTS") == BOUNDED_PYTEST_ADDOPTS


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
        for token in ("-n 2", "--dist=worksteal", "--durations=50"):
            assert token in command
        for token in (
            "-p tools.pytest_shard",
            "--fq-shard-count 8",
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
    shard = _github_hosted_jobs("ci.yml")["coverage-shard"]
    strategy = shard.get("strategy")
    assert isinstance(strategy, dict)
    assert strategy.get("matrix") == {"shard": list(range(8))}

    final = _github_hosted_jobs("ci.yml")["coverage"]
    assert final.get("needs") == "coverage-shard"
    commands = "\n".join(
        str(step.get("run", "")) for step in _steps("ci.yml", "coverage")
    )
    assert "needs.coverage-shard.result" in commands
    assert "coverage combine coverage-data" in commands
    assert commands.index("coverage combine coverage-data") < commands.index(
        "tools/check_coverage.py"
    )


def test_each_optional_compatibility_matrix_reuses_one_runner() -> None:
    jobs = _github_hosted_jobs("ci.yml")
    expected = {
        "qiskit-optional": "2.0.* 2.5.*",
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
