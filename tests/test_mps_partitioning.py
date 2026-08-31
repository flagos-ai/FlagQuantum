import pytest

from flagquantum.runtime.backends.mps.state import (
    communication_aware_mps_ownership,
    cost_aware_mps_ownership,
    gate_aligned_cost_aware_mps_ownership,
    mps_factorization_site_costs,
    topology_aware_mps_ownership,
    validate_mps_ownership,
)


def test_communication_aware_ownership_avoids_hot_cuts_with_bounded_load():
    bonds = (1, *([64] * 63), 1)
    penalties = [0] * 63
    for bond in (7, 15, 23, 31, 39, 47, 55):
        penalties[bond] = 100

    ownership = communication_aware_mps_ownership(
        bonds,
        world_size=8,
        boundary_penalties=penalties,
        maximum_load_ratio=1.25,
    )
    cuts = {shard[-1] for shard in ownership[:-1]}

    assert cuts.isdisjoint({7, 15, 23, 31, 39, 47, 55})
    assert tuple(wire for shard in ownership for wire in shard) == tuple(range(64))


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


def test_topology_aware_ownership_moves_hot_cut_off_node_boundary():
    bonds = (1, *([4] * 11), 1)
    penalties = [1] * 11
    penalties[5] = 100
    ownership = topology_aware_mps_ownership(
        bonds,
        world_size=4,
        local_world_size=2,
        boundary_penalties=penalties,
        inter_node_multiplier=16,
        maximum_load_ratio=1.6,
    )
    inter_node_cut = ownership[1][-1]
    assert inter_node_cut != 5
    assert tuple(wire for shard in ownership for wire in shard) == tuple(range(12))


@pytest.mark.parametrize(("local_world_size", "multiplier"), ((0, 8), (3, 8), (2, 0)))
def test_topology_aware_ownership_rejects_invalid_topology(
    local_world_size, multiplier
):
    with pytest.raises(ValueError):
        topology_aware_mps_ownership(
            (1,) * 9,
            world_size=4,
            local_world_size=local_world_size,
            boundary_penalties=(1,) * 7,
            inter_node_multiplier=multiplier,
        )


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
