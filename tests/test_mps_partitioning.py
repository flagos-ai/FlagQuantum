import pytest

from flagquantum.runtime.backends.mps.state import (
    cost_aware_mps_ownership,
    gate_aligned_cost_aware_mps_ownership,
    mps_factorization_site_costs,
    validate_mps_ownership,
)


def test_cost_aware_mps_ownership_is_contiguous_and_reduces_peak_proxy():
    # Open boundaries with a broad chi=8 plateau emulate a variable-bond MPS.
    bonds = (1, 2, 4, 8, 8, 8, 8, 8, 4, 2, 1)
    ownership = cost_aware_mps_ownership(bonds, world_size=4)
    assert validate_mps_ownership(ownership, n_wires=10, world_size=4) == ownership

    costs = mps_factorization_site_costs(bonds)
    balanced_peak = max(sum(costs[wire] for wire in shard) for shard in ownership)
    equal = (range(0, 2), range(2, 5), range(5, 7), range(7, 10))
    equal_peak = max(sum(costs[wire] for wire in shard) for shard in equal)
    assert balanced_peak < equal_peak


def test_gate_aligned_cost_aware_ownership_keeps_all_internal_cuts_even():
    bonds = (1, 2, 4, 8, 8, 8, 8, 8, 8, 8, 4, 2, 1)
    ownership = gate_aligned_cost_aware_mps_ownership(bonds, world_size=4)
    assert validate_mps_ownership(ownership, n_wires=12, world_size=4) == ownership
    assert all((shard[-1] + 1) % 2 == 0 for shard in ownership[:-1])


def test_gate_aligned_cost_aware_ownership_fails_when_cuts_do_not_fit():
    with pytest.raises(ValueError, match="not enough aligned cuts"):
        gate_aligned_cost_aware_mps_ownership((1,) * 7, world_size=5)


@pytest.mark.parametrize(
    "ownership",
    (
        ((0, 1), (3, 4)),  # gap
        ((0, 1), (1, 2)),  # overlap
        ((0, 2), (1, 3)),  # non-contiguous
        ((0, 1, 2, 3), ()),  # empty rank
    ),
)
def test_validate_mps_ownership_rejects_invalid_partitions(ownership):
    with pytest.raises(ValueError):
        validate_mps_ownership(ownership, n_wires=4, world_size=2)
