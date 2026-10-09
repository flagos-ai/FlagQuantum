from __future__ import annotations

import pytest

from tools.pytest_shard import shard_for

pytestmark = pytest.mark.unit


def test_a_node_id_has_one_stable_shard() -> None:
    nodeid = "tests/unit/test_example.py::test_case[value]"
    assert shard_for(nodeid, 7) == shard_for(nodeid, 7)
    assert 0 <= shard_for(nodeid, 7) < 7


def test_every_node_id_is_assigned_exactly_once() -> None:
    nodeids = tuple(f"tests/unit/test_{index}.py::test_case" for index in range(100))
    assignments = {
        nodeid: [index for index in range(4) if shard_for(nodeid, 4) == index]
        for nodeid in nodeids
    }
    assert all(len(indices) == 1 for indices in assignments.values())
    assert {indices[0] for indices in assignments.values()} == {0, 1, 2, 3}


def test_a_non_positive_shard_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        shard_for("tests/unit/test_example.py::test_case", 0)
