import pytest

from flagquantum.runtime.executors.mps.reverse import _planned_canonicalization_bonds

pytestmark = pytest.mark.unit


def test_certified_clean_state_skips_full_sweep():
    assert _planned_canonicalization_bonds(64, (), "dirty") == ()
    assert len(_planned_canonicalization_bonds(64, (), "full")) == 63


def test_dirty_interval_propagates_only_to_right_edge():
    assert _planned_canonicalization_bonds(8, (5,), "dirty") == (5, 6)
    assert _planned_canonicalization_bonds(8, (2, 5), "dirty") == tuple(range(2, 7))
    with pytest.raises(ValueError, match="outside"):
        _planned_canonicalization_bonds(8, (7,), "dirty")
