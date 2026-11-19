"""The executable colour workflow preserves the complete requested inventory."""

import pytest

from examples.qec.colour_distance_acceptance import (
    BATCH_SIZE,
    DISTANCES,
    ROUNDS,
    SHOTS,
    accept_configuration,
)

pytestmark = pytest.mark.unit


def test_complete_inventory_and_fixed_budget() -> None:
    assert DISTANCES == (3, 5, 7, 9, 11, 13, 15)
    assert (ROUNDS, SHOTS, BATCH_SIZE) == (3, 64, 16)


def test_outside_inventory_is_refused_before_sdk_import() -> None:
    with pytest.raises(ValueError, match="outside the declared"):
        accept_configuration(17, "z")
