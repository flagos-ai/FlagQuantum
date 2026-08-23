import pytest

from flagquantum.runtime.backends.tensor_network import plan_tn_parameter_owners


def test_tn_parameter_owner_plan_is_balanced_and_complete():
    owners = plan_tn_parameter_owners(36, 16)

    assert len(owners) == 36
    assert set(owners) == set(range(16))
    counts = tuple(owners.count(rank) for rank in range(16))
    assert max(counts) - min(counts) <= 1


@pytest.mark.parametrize(("parameter_count", "world_size"), ((0, 2), (2, 0)))
def test_tn_parameter_owner_plan_rejects_nonpositive_inputs(
    parameter_count,
    world_size,
):
    with pytest.raises(ValueError):
        plan_tn_parameter_owners(parameter_count, world_size)
