from __future__ import annotations

import json

import pytest

from tools.pytest_shard import (
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
