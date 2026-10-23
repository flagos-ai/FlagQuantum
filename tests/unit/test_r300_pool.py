"""Admission evidence and deterministic routing keep unqualified sets out of CI."""

import copy
from pathlib import Path

import pytest

from tools.ci.r300_pool import load_pool, plan, qualification_targets

pytestmark = pytest.mark.unit


def test_every_route_is_qualified_and_jobs_use_distinct_targets() -> None:
    pool = load_pool()
    qualified = {m["label"] for m in pool["members"] if m["state"] == "qualified"}
    observed: set[str] = set()
    for seed in range(100):
        routes = plan(pool, seed)
        assert routes == plan(pool, seed)
        assert set(routes) == set(pool["jobs"])
        assert len(set(routes.values())) == len(routes)
        assert set(routes.values()) <= qualified
        observed.update(routes.values())
    assert observed == qualified


def test_candidates_and_quarantined_sets_never_receive_production_jobs() -> None:
    pool = load_pool()
    blocked = {m["label"] for m in pool["members"] if m["state"] != "qualified"}
    assert blocked
    for seed in range(len(pool["members"])):
        assert not blocked.intersection(plan(pool, seed).values())
    candidates = qualification_targets(pool, 4)
    assert len(candidates) == min(
        4,
        sum(
            m["state"] == "candidate" and m["label"] in pool["qualification_batch"]
            for m in pool["members"]
        ),
    )
    assert all(
        next(m for m in pool["members"] if m["label"] == label)["state"] == "candidate"
        for label in candidates
    )


def test_quarantine_immediately_removes_a_member_from_new_plans() -> None:
    pool = copy.deepcopy(load_pool())
    qualified = [m for m in pool["members"] if m["state"] == "qualified"]
    removed = qualified[0]
    removed["state"] = "quarantined"
    pool["jobs"] = pool["jobs"][: len(qualified) - 1]
    for seed in range(20):
        assert removed["label"] not in plan(pool, seed).values()


def test_insufficient_qualified_members_fail_without_routing_to_candidates() -> None:
    pool = copy.deepcopy(load_pool())
    pool["jobs"] += [f"extra-job-{i}" for i in range(len(pool["members"]))]
    with pytest.raises(ValueError, match="distinct qualified"):
        plan(pool, 1)


@pytest.mark.parametrize("limit", [0, 5, -1])
def test_qualification_fanout_is_bounded(limit: int) -> None:
    with pytest.raises(ValueError, match="at most four"):
        qualification_targets(load_pool(), limit)


def test_registry_rejects_duplicate_targets_and_missing_evidence(
    tmp_path: Path,
) -> None:
    import json

    pool = load_pool()
    pool["members"].append(pool["members"][0])
    path = tmp_path / "pool.json"
    path.write_text(json.dumps(pool))
    with pytest.raises(ValueError, match="duplicate"):
        load_pool(path)
    pool = load_pool()
    next(m for m in pool["members"] if m["state"] == "qualified").pop("evidence")
    path.write_text(json.dumps(pool))
    with pytest.raises(ValueError, match="Missing qualification"):
        load_pool(path)


def test_ci_consumes_one_shared_plan_and_qualification_excludes_fork_prs() -> None:
    import yaml

    workflows = Path(__file__).resolve().parents[2] / ".github/workflows"
    jobs = yaml.safe_load((workflows / "ci.yml").read_text())["jobs"]
    assert jobs["prepare-source"]["outputs"]["r300-routes"] == (
        "${{ steps.r300-pool.outputs.r300-routes }}"
    )
    producer = next(
        s for s in jobs["prepare-source"]["steps"] if s.get("id") == "r300-pool"
    )
    assert "python3 tools/ci/r300_pool.py" in producer["run"]
    assert {
        name
        for name, job in jobs.items()
        if "needs.prepare-source.outputs.r300-routes" in str(job.get("runs-on", ""))
    } == set(load_pool()["jobs"])
    for name in load_pool()["jobs"]:
        assert (
            "fromJSON(needs.prepare-source.outputs.r300-routes)['" + name + "']"
            in jobs[name]["runs-on"]
        )
        assert (
            "github.event.pull_request.head.repo.full_name == github.repository"
            in jobs[name]["runs-on"]
        )
    validation = yaml.safe_load((workflows / "r300-validation.yml").read_text())
    qualification = validation["jobs"]["pool-qualification"]
    assert (
        "github.event.pull_request.head.repo.full_name == github.repository"
        in qualification["if"]
    )
    assert qualification["strategy"]["max-parallel"] == 4
    assert qualification["strategy"]["matrix"]["scale_set"] == (
        "${{ fromJSON(needs.prepare-source.outputs.qualification-targets) }}"
    )
    for job in ("resources", "cpu-workload"):
        assert (
            "github.event_name == 'workflow_dispatch'" in validation["jobs"][job]["if"]
        )
    events = validation.get("on") or validation[True]
    assert "pull_request_target" not in events


def test_cli_outputs_valid_workflow_json(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys

    output = tmp_path / "outputs"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.ci.r300_pool",
            "--seed",
            "42",
            "--github-output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert json.loads(values["r300-routes"]) == plan(load_pool(), 42)
    assert json.loads(values["qualification-targets"]) == qualification_targets(
        load_pool(), 4
    )


def test_completed_batch_does_not_start_an_unselected_batch() -> None:
    pool = copy.deepcopy(load_pool())
    for member in pool["members"]:
        if member["label"] in pool["qualification_batch"]:
            member["state"] = "qualified"
    assert any(member["state"] == "candidate" for member in pool["members"])
    assert qualification_targets(pool, 4) == []
