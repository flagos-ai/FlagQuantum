import pytest

from flagquantum.runtime.executors.tensor_network.distributed_optimizer import (
    GRADIENT_REDUCTION_MODES,
    DistributedTNOptimizerStepResult,
    execute_rank_owned_tn_sgd_step,
    plan_tn_parameter_owners,
    plan_tn_parameter_ownership,
)

pytestmark = pytest.mark.unit


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


def test_tn_parameter_ownership_map_covers_every_parameter_once():
    ownership = plan_tn_parameter_ownership(5, 4)

    assert len(ownership) == 4
    assert sum(item["owned_parameter_count"] for item in ownership) == 5
    assert [index for item in ownership for index in item["parameter_indices"]] == list(
        range(5)
    )
    for item in ownership:
        assert item["owned_parameter_count"] == len(item["parameter_indices"])
        assert item["writeback_route"] == "owner_update_then_packed_all_gather"


def _step(gradient_reduction: str) -> DistributedTNOptimizerStepResult:
    return DistributedTNOptimizerStepResult(
        rank=0,
        world_size=2,
        local_world_size=1,
        learning_rate=0.05,
        owned_parameter_indices=(0,),
        ownership=plan_tn_parameter_ownership(2, 2),
        gradient_reduction=gradient_reduction,
        collective_count=1,
        collective_payload_bytes_per_rank=32,
        execution_seconds=0.0,
    )


def test_an_owner_scoped_reduction_reports_owned_gradients():
    summary = _step("owner_reduce").summary()

    assert summary["gradient_ownership_semantics"] == "sharded_across_ranks"
    assert summary["gradient_distribution_semantics"] == "sharded_across_ranks"
    assert summary["gradient_ownership"] == summary["optimizer_update_ownership"]


def test_a_replicated_reduction_refuses_to_claim_gradient_ownership():
    """A full all-reduce leaves the whole reduced gradient on every rank."""

    summary = _step("all_reduce").summary()

    assert summary["gradient_distribution_semantics"] == "replicated_after_all_reduce"
    assert summary["gradient_ownership_semantics"] == "replicated_after_all_reduce"
    assert summary["gradient_ownership"] == ()
    # The update is owner-applied either way, so only the gradient claim moves.
    assert summary["optimizer_update_ownership_semantics"] == "sharded_across_ranks"
    assert summary["optimizer_update_ownership"]


@pytest.mark.parametrize("mode", sorted(GRADIENT_REDUCTION_MODES))
def test_every_supported_reduction_has_a_distribution_statement(mode):
    assert _step(mode).gradient_distribution_semantics in {
        "sharded_across_ranks",
        "replicated_after_all_reduce",
    }


def test_the_lower_level_reduction_default_is_not_an_ownership_claim():
    """The reverse executor's own default is a replicated gradient."""

    from flagquantum.runtime.executors.tensor_network.distributed_sliced_reverse import (
        execute_distributed_sliced_tn_explicit_reverse,
    )

    default = execute_distributed_sliced_tn_explicit_reverse.__kwdefaults__[
        "gradient_reduction"
    ]

    assert default == "all_reduce"


def test_an_unknown_reduction_fails_before_any_resource_is_required():
    import torch

    with pytest.raises(ValueError, match="gradient_reduction must be one of"):
        execute_rank_owned_tn_sgd_step(
            (torch.zeros(1, dtype=torch.float64),),
            (torch.zeros(1, dtype=torch.float64),),
            learning_rate=0.05,
            gradient_reduction="reduce_scatter",
        )


def test_the_training_default_is_owner_reduce():
    """The training entry point is the lane whose gradients are owner-scoped."""

    from flagquantum.runtime.executors.tensor_network.training import (
        train_distributed_tensor_network,
    )

    default = train_distributed_tensor_network.__kwdefaults__["gradient_reduction"]

    assert default == "owner_reduce"
