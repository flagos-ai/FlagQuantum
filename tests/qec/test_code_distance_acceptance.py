"""The acceptance list stays identical to the requested distance inventory."""

import pytest

from examples.qec.code_distance_acceptance import (
    BATCH_SIZE,
    REPETITION_DISTANCES,
    ROUNDS,
    SHOTS,
    SURFACE_DISTANCES,
    accept_configuration,
    configurations,
)

pytestmark = pytest.mark.unit


def test_complete_distance_inventory_and_execution_budget() -> None:
    assert SURFACE_DISTANCES == (3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 30)
    assert REPETITION_DISTANCES == (3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29)
    assert len(configurations()) == len(set(configurations())) == 42
    assert sum(basis == "z" for _, _, basis in configurations()) == 28
    assert (ROUNDS, SHOTS, BATCH_SIZE) == (3, 64, 16)


def test_outside_inventory_is_refused_before_construction() -> None:
    with pytest.raises(ValueError, match="outside the declared"):
        accept_configuration("surface", 29, "z")
