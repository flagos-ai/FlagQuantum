"""The GitHub-hosted share one FlagQuantum workflow run is allowed to hold.

flagos-ai runs every repository from one pool of 20 concurrent GitHub-hosted
jobs. A job that joins no concurrency group holds a slot of that pool for as
long as it runs, so a workflow that is unconstrained here is a workflow that
queues every other repository in the organisation behind it.

The ceiling is expressed the way GitHub expresses it. A concurrency group
admits one job at a time, so the number of distinct groups one workflow run
uses *is* the number of GitHub-hosted runners that run can hold at once.
`ci.yml` and `pre-commit.yaml` partition their jobs across the eight scoped
buckets below, and this module holds that partition still. The pull-request or
branch scope prevents an unrelated older run from serializing a newer run;
the organisation's 20-runner pool remains the aggregate ceiling.

Five invariants:

1. Every `ubuntu-latest` job joins one of the buckets, and every bucket is
   used: an unused bucket is a ceiling lower than the comment claims, and an
   undeclared one is a ceiling higher than it.
2. The buckets carry `queue: max`. The default keeps only the newest waiting
   job in a group and cancels the ones behind it, so without it a bucket drops
   work rather than delaying it.
3. The bucket count is the ceiling, so it is asserted rather than counted. The
   `cpucore` bucket carries the matrix value in its name, which means a fourth
   interpreter would be a fourth bucket: the set is spelled out so that is a
   decision about the ceiling and not a side effect of a matrix edit.
4. Both workflows cancel a superseded run of the same branch, so a push that
   lands during an earlier run does not queue a second full matrix behind it.
5. `publish-dev-container.yml` caps its own matrix instead of joining a bucket,
   because its legs build container images and serializing them all would cost
   more than the slots are worth. `cd.yml` is out of scope: it publishes on a
   release and holds one job, so it cannot occupy more than one runner.
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

# One concurrency group admits one job, so these eight scoped names are a
# ceiling of eight concurrent GitHub-hosted runners for one run.
RUN_SCOPE = "${{ github.event.pull_request.number || github.ref_name }}"
BUCKETS = frozenset(
    {
        f"flagquantum-gh-{RUN_SCOPE}-cpucore-3.10",
        f"flagquantum-gh-{RUN_SCOPE}-cpucore-3.11",
        f"flagquantum-gh-{RUN_SCOPE}-cpucore-3.12",
        f"flagquantum-gh-{RUN_SCOPE}-coverage",
        f"flagquantum-gh-{RUN_SCOPE}-distributed",
        f"flagquantum-gh-{RUN_SCOPE}-light-a",
        f"flagquantum-gh-{RUN_SCOPE}-light-b",
        f"flagquantum-gh-{RUN_SCOPE}-light-c",
    }
)

CEILING = 8

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
        if job.get("runs-on") == "ubuntu-latest":
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


def test_the_scan_finds_the_github_hosted_jobs_it_is_meant_to_police() -> None:
    # A policy scan that finds nothing passes every rule below while checking
    # nothing, so the discovery itself is asserted.
    found = set(_bucketed_jobs())

    assert {
        "ci.yml:quality",
        "ci.yml:cpu-core",
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


def test_a_superseded_run_is_cancelled_rather_than_queued() -> None:
    offenders: list[str] = []
    for workflow in BUCKETED_WORKFLOWS:
        concurrency = _document(workflow).get("concurrency")
        if not isinstance(concurrency, dict):
            offenders.append(f"{workflow} declares no workflow-level concurrency")
            continue
        if concurrency.get("cancel-in-progress") is not True:
            offenders.append(f"{workflow} does not cancel a superseded run")
    assert not offenders, (
        "a push that lands during an earlier run must cancel that run, not "
        "queue a second full matrix behind it:\n" + "\n".join(offenders)
    )


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
