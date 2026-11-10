"""Measured pool lifetimes preserve clean/borrowed ownership and release order."""

from typing import Any

import pytest

from flagquantum.compiler.registers.layout.workspace import RegisterWorkspace

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("reset", [False, True])
def test_partial_retirement_preserves_remainder_and_old_lease_identity(
    reset: bool,
) -> None:
    pool = RegisterWorkspace((7, 3, 5))
    lease = pool.allocate(3)
    pool.retire_measured_wires((3,), reset=reset)
    if reset:
        successor = pool.allocate(1)
        assert successor.wires == (3,)
    else:
        with pytest.raises(ValueError, match="Insufficient"):
            pool.allocate(1)
    with pytest.raises(ValueError, match="still live"):
        pool.free(lease, order=(7, 3, 5))
    pool.free(lease, order=(5, 7))
    assert pool.allocate(2).wires == (5, 7)
    with pytest.raises(ValueError, match="live lease"):
        pool.free(lease)
    assert pool.used_wires == (3, 5, 7)


@pytest.mark.parametrize("reset", [False, True])
def test_retirement_of_mixed_owners_rejects_atomically(reset: bool) -> None:
    pool = RegisterWorkspace((1,), borrowed=(2,))
    clean = pool.allocate(1)
    borrowed = pool.allocate(1, dirty=True)
    with pytest.raises(ValueError, match="borrowed"):
        pool.retire_measured_wires((1, 2), reset=reset)
    pool.free(clean)
    pool.free(borrowed)
    assert pool.allocate(1).wires == (1,)
    assert pool.allocate(1, dirty=True).wires == (2,)


@pytest.mark.parametrize("wires", [(1, 1), (1, True), (1, 99)])
def test_invalid_retirement_preserves_pool(wires: tuple[int, ...]) -> None:
    pool = RegisterWorkspace((1, 2))
    lease = pool.allocate(2)
    with pytest.raises(ValueError, match="distinct|live wires"):
        pool.retire_measured_wires(wires, reset=False)
    pool.free(lease)
    assert pool.allocate(2).wires == (1, 2)


@pytest.mark.parametrize("reset", [False, True])
def test_adopted_owners_and_stale_retirement(reset: bool) -> None:
    pool = RegisterWorkspace(())
    lease = pool.adopt((4, 8))
    pool.retire_measured_wires((8, 4), reset=reset)
    with pytest.raises(ValueError, match="live wires"):
        pool.retire_measured_wires((4,), reset=reset)
    with pytest.raises(ValueError, match="live lease"):
        pool.free(lease)
    if reset:
        assert pool.allocate(2).wires == (8, 4)
    else:
        with pytest.raises(ValueError, match="Insufficient"):
            pool.allocate(1)


def test_empty_retirement_does_not_consume_zero_width_lease() -> None:
    pool = RegisterWorkspace(())
    lease = pool.allocate(0)
    pool.retire_measured_wires((), reset=False)
    pool.free(lease)
    invalid: Any = 1
    with pytest.raises(ValueError, match="boolean"):
        pool.retire_measured_wires((), reset=invalid)
