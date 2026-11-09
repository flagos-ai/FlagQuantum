"""Atomic adoption and original-pool release of pre-owned physical inputs."""

import pytest

from flagquantum.compiler.registers.layout.workspace import RegisterWorkspace

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dirty", [False, True])
def test_adoption_keeps_available_order_and_releases_current_wire_order(dirty):
    pool = RegisterWorkspace((8,), borrowed=(9,))
    lease = pool.adopt((3, 1, 4), dirty=dirty)
    assert lease.wires == (3, 1, 4) and lease.dirty is dirty
    assert pool.allocate(1).wires == (8,)
    assert pool.allocate(1, dirty=True).wires == (9,)
    with pytest.raises(ValueError, match="Insufficient"):
        pool.allocate(1, dirty=dirty)
    pool.free_wires((4,))
    pool.free(lease, order=(1, 3))
    assert pool.allocate(3, dirty=dirty).wires == (1, 3, 4)
    assert pool.used_wires == (1, 3, 4, 8, 9)


@pytest.mark.parametrize("dirty", [False, True])
@pytest.mark.parametrize("occupied_dirty", [False, True])
@pytest.mark.parametrize("active", [False, True])
def test_adoption_collision_is_atomic_and_retryable(dirty, occupied_dirty, active):
    pool = (
        RegisterWorkspace((), borrowed=(2,))
        if occupied_dirty
        else RegisterWorkspace((2,))
    )
    if active:
        pool.allocate(1, dirty=occupied_dirty)
    with pytest.raises(ValueError, match="overlap"):
        pool.adopt((1, 2), dirty=dirty)
    assert pool.used_wires == ((2,) if active else ())
    adopted = pool.adopt((1,), dirty=dirty)
    pool.free(adopted)
    assert pool.allocate(1, dirty=dirty).wires == (1,)
    with pytest.raises(ValueError, match="overlap"):
        pool.adopt((1,), dirty=dirty)


@pytest.mark.parametrize("wires", [[1], (1, 1), (1, True), (1, -1), (1, 2.0), None])
def test_invalid_adoption_preserves_all_state(wires):
    pool = RegisterWorkspace((7,))
    with pytest.raises(ValueError, match="Adoption"):
        pool.adopt(wires)
    assert pool.used_wires == ()
    lease = pool.adopt((1, 2))
    pool.free(lease)
    assert pool.allocate(3).wires == (1, 2, 7)


@pytest.mark.parametrize("dirty", [0, 1, None, "false"])
def test_adoption_requires_boolean_origin(dirty):
    pool = RegisterWorkspace(())
    with pytest.raises(ValueError, match="boolean"):
        pool.adopt((1,), dirty=dirty)
    assert pool.adopt((1,)).wires == (1,)


def test_partial_release_across_adopted_and_allocated_pools_is_atomic():
    pool = RegisterWorkspace((7,))
    clean = pool.adopt((1, 2))
    pool.adopt((3, 4), dirty=True)
    pool.allocate(1)
    with pytest.raises(ValueError, match="live wires"):
        pool.free_wires((1, 7, 99))
    with pytest.raises(ValueError, match="Insufficient"):
        pool.allocate(1)
    pool.free_wires((4, 7, 2, 3))
    assert pool.allocate(2).wires == (7, 2)
    assert pool.allocate(2, dirty=True).wires == (4, 3)
    pool.free(clean)
    assert pool.allocate(1).wires == (1,)


def test_empty_adopted_lifetimes_keep_separate_lease_identity():
    pool = RegisterWorkspace(())
    first = pool.adopt(())
    second = pool.adopt((), dirty=True)
    pool.free_wires(())
    pool.free(first)
    pool.free(second)
    with pytest.raises(ValueError, match="live lease"):
        pool.free(first)
