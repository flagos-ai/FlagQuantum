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
    assert len(candidates) == 4
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
