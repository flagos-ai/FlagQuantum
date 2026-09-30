"""Contracts for distributed tensor-network placement and semantics reporting.

Repository rule 5 requires every distributed result to expose `world_size`,
`local_world_size`, `node_count`, rank ownership, memory, communication,
`distribution_semantics`, `scalability_claim_allowed`, and blockers. These tests
fail if the tensor-network entry points report a placement they did not run at,
report rank-local memory as rank-wide evidence, or describe simulated
single-process work as sharded execution.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

import flagquantum as fq
from flagquantum.runtime.executors.tensor_network import execution

pytestmark = pytest.mark.unit


class _FakeContext:
    """The subset of `TorchDistributedContext` the executor reads."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        local_world_size: int,
        backend: str = "gloo",
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.local_world_size = local_world_size
        self.local_rank = rank % local_world_size
        self.node_rank = rank // local_world_size
        self.node_count = world_size // local_world_size
        self.hostname = "fake-host"
        self.backend = backend
        self.device = torch.device("cpu")
        self.identity = None
        self.initialized = True


def _install_single_process_group(
    monkeypatch: pytest.MonkeyPatch,
    *,
    world_size: int,
    local_world_size: int,
    rank: int = 0,
    backend: str = "gloo",
) -> dict[str, int]:
    """Simulate a process group inside one process and count collectives."""

    counters = {"all_reduce": 0, "all_gather": 0, "broadcast": 0}
    context = _FakeContext(
        rank=rank,
        world_size=world_size,
        local_world_size=local_world_size,
        backend=backend,
    )

    def all_reduce(tensor: torch.Tensor, *, op: object) -> None:
        counters["all_reduce"] += 1

    def all_gather(outputs: list[torch.Tensor], tensor: torch.Tensor) -> None:
        counters["all_gather"] += 1
        for output in outputs:
            output.copy_(tensor)

    def broadcast_object_list(payload: list[Any], *, src: int, device: Any) -> None:
        counters["broadcast"] += 1
        assert src == 0
        assert payload[0] is not None

    monkeypatch.setattr(execution.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(execution.dist, "all_gather", all_gather)
    monkeypatch.setattr(
        execution.dist, "get_world_size", lambda *args, **kwargs: world_size
    )
    monkeypatch.setattr(execution.dist, "get_rank", lambda *args, **kwargs: rank)
    monkeypatch.setattr(execution.dist, "broadcast_object_list", broadcast_object_list)
    monkeypatch.setattr(execution, "init_torch_distributed", lambda **_: context)
    return counters


def _circuit() -> fq.Circuit:
    circuit = fq.Circuit(5)
    circuit.h(0).cx(0, 4).ry(2, theta=0.2).rzz(1, 3, theta=-0.4)
    return circuit


def _sliced_labels(*, count: int = 3) -> tuple[int, ...]:
    """Pick contractable internal labels so the plan has many slices."""

    from collections import Counter

    from flagquantum.simulation.tensor_network.entrypoints import (
        build_tensor_network_expectation,
    )

    plan = build_tensor_network_expectation(_circuit(), z=(1, 3))
    counts = Counter(label for node in plan.nodes for label in node.labels)
    return tuple(
        label
        for label, occurrences in counts.items()
        if occurrences >= 2 and label not in plan.output_labels
    )[:count]


@pytest.mark.parametrize(
    ("world_size", "local_world_size", "node_count"),
    [(2, 1, 2), (2, 2, 1), (4, 2, 2), (4, 4, 1), (8, 1, 8)],
)
def test_tn_placement_resolves_nodes_from_the_declared_local_world_size(
    world_size: int, local_world_size: int, node_count: int
) -> None:
    placement = execution._resolve_rank_placement(
        world_size=world_size,
        local_world_size=local_world_size,
        context=None,
    )

    assert placement["world_size"] == world_size
    assert placement["local_world_size"] == local_world_size
    assert placement["node_count"] == node_count
    assert placement["local_world_size_source"] == "caller"


def test_tn_placement_prefers_the_process_group_over_the_argument() -> None:
    context = _FakeContext(rank=3, world_size=4, local_world_size=1)
    placement = execution._resolve_rank_placement(
        world_size=4, local_world_size=None, context=context
    )

    assert placement["local_world_size"] == 1
    assert placement["local_world_size_source"] == "process_group_environment"
    assert placement["node_count"] == 4
    assert placement["rank"] == 3
    assert placement["node_rank"] == 3


@pytest.mark.parametrize(
    ("world_size", "local_world_size", "message"),
    [
        (2, 0, "positive rank count"),
        (2, -1, "positive rank count"),
        (4, 3, "must divide world_size"),
    ],
)
def test_tn_placement_rejects_undeployable_topologies(
    world_size: int, local_world_size: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        execution._resolve_rank_placement(
            world_size=world_size,
            local_world_size=local_world_size,
            context=None,
        )


def test_tn_task_plan_places_slices_at_the_reported_local_world_size() -> None:
    slicing = execution._build_slicing_plan(
        *execution._amplitude_projection(
            execution.build_tensor_network(_circuit()), "10001"
        ),
        sliced_labels=_sliced_labels(),
    )
    tasks = execution._execution_slice_tasks(slicing, world_size=4, local_world_size=2)

    assert slicing.n_slices >= 4
    assert {task.owner_node for task in tasks} == {0, 1}
    assert {task.owner_local_rank for task in tasks} == {0, 1}
    assert {task.owner_rank for task in tasks} == {0, 1, 2, 3}
    # A wider world size with the same local world size spreads across nodes
    # rather than piling every slice onto rank 0.
    wide = execution._execution_slice_tasks(slicing, world_size=8, local_world_size=2)
    assert {task.owner_node for task in wide} == {0, 1, 2, 3}


def test_tn_amplitude_reports_sharded_semantics_and_every_rank_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counters = _install_single_process_group(
        monkeypatch, world_size=2, local_world_size=1
    )
    result = execution.distributed_tensor_network_amplitude(
        _circuit(),
        "10001",
        world_size=2,
        distributed_executor="torch",
        backend="gloo",
        sliced_labels=_sliced_labels(),
    )
    summary = result.summary()

    assert summary["distribution_semantics"] == "sharded_across_ranks"
    assert summary["scalability_claim_allowed"] is False
    assert summary["local_world_size"] == 1
    assert summary["node_count"] == 2
    assert summary["rank_placement"]["node_count"] == 2
    assert summary["full_state_materialized"] is False
    assert counters["all_reduce"] == 1
    # Per-rank memory is gathered evidence, so it is not just the local rank.
    assert counters["all_gather"] == 1
    assert summary["local_memory_bytes_by_rank"] == tuple(
        summary["rank_partial_bytes_by_rank"][rank] for rank in (0, 1)
    )
    assert summary["local_memory_bytes_by_rank"][0] > 0
    assert summary["communication_tiers"]["collective"] == "all_reduce_sum"
    assert (
        summary["communication_tiers"]["inter_node_collective_bytes"]
        == (summary["reduction_payload_bytes"])
    )
    assert summary["communication_tiers"]["intra_node_collective_bytes"] == 0
    assert summary["communication_tiers"]["unattributed_collective_bytes"] == 0


def test_tn_expectation_over_single_node_attributes_reduction_intra_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_single_process_group(monkeypatch, world_size=4, local_world_size=4)
    result = execution.distributed_tensor_network_expectation(
        _circuit(),
        z=(1, 3),
        world_size=4,
        distributed_executor="torch",
        backend="gloo",
        sliced_labels=_sliced_labels(),
    )
    summary = result.summary()

    assert summary["distribution_semantics"] == "sharded_across_ranks"
    assert summary["node_count"] == 1
    assert (
        summary["communication_tiers"]["intra_node_collective_bytes"]
        == (summary["reduction_payload_bytes"])
    )
    assert summary["communication_tiers"]["inter_node_collective_bytes"] == 0
    assert summary["communication_tiers"]["inter_node_collective_possible"] is False


def test_tn_expectation_reports_placement_and_owner_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_single_process_group(monkeypatch, world_size=4, local_world_size=2)
    result = execution.distributed_tensor_network_expectation(
        _circuit(),
        z=(1, 3),
        world_size=4,
        distributed_executor="torch",
        backend="gloo",
        sliced_labels=_sliced_labels(),
    )
    summary = result.summary()

    assert summary["distribution_semantics"] == "sharded_across_ranks"
    assert summary["local_world_size"] == 2
    assert summary["node_count"] == 2
    assert summary["rank_placement"]["local_rank"] == 0
    assert summary["slice_tasks"] >= 4
    assert sorted(summary["tasks_by_rank"]) == [0, 1, 2, 3]
    assert sum(summary["tasks_by_rank"].values()) == summary["slice_tasks"]
    # A mixed placement crosses a node boundary on some legs only, so no bytes
    # are attributed to either tier.
    assert (
        summary["communication_tiers"]["unattributed_collective_bytes"]
        == (summary["reduction_payload_bytes"])
    )
    assert summary["communication_tiers"]["inter_node_collective_possible"] is True


def test_tn_sharded_summary_satisfies_the_distributed_evidence_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flagquantum.runtime.audit.engine import evaluate_distributed_evidence_contract

    _install_single_process_group(monkeypatch, world_size=4, local_world_size=1)
    result = execution.distributed_tensor_network_expectation(
        _circuit(),
        z=(1, 3),
        world_size=4,
        distributed_executor="torch",
        backend="gloo",
        sliced_labels=_sliced_labels(),
    )
    contract = evaluate_distributed_evidence_contract(result.summary())

    assert contract.backend_family == "tensor_network"
    for check in (
        "known_backend_family",
        "known_claim_evidence_type",
        "sharding_semantics_available",
        "not_replicated_or_rank_local",
        "topology_reported",
        "rank_ownership_reported",
        "memory_plan_reported",
        "communication_plan_reported",
        "multi_node_transport_evidence_complete",
    ):
        assert contract.checks[check], check
    # The runtime result is not itself a release claim, so it stays fail-closed.
    assert contract.claimable_production_training is False


def test_tn_local_development_simulation_is_blocked_not_sharded() -> None:
    result = execution.distributed_tensor_network_amplitude(
        _circuit(),
        "10001",
        world_size=2,
        local_world_size=1,
        distributed_profile="development",
        torch_backend="local_tensor",
        sliced_labels=_sliced_labels(),
    )
    summary = result.summary()

    assert summary["distribution_semantics"] == (
        "local_simulated_slice_parallel_sparse_output_reduction"
    )
    assert summary["scalability_claim_allowed"] is False
    assert summary["scalability_blockers"] == (
        "slice_parallel_local_development_simulator",
    )
    # The declared topology is still reported, so a reader can tell the work
    # was simulated at a two-node placement inside one process.
    assert summary["node_count"] == 2
    assert summary["local_world_size"] == 1


def test_tn_single_device_result_claims_no_sharding() -> None:
    result = execution.distributed_tensor_network_amplitude(
        _circuit(),
        "10001",
        world_size=1,
        distributed_profile="development",
        torch_backend="local_tensor",
        sliced_labels=_sliced_labels(),
    )
    summary = result.summary()

    assert summary["distribution_semantics"] == "single_device_sparse_output"
    assert summary["scalability_claim_allowed"] is False
    assert summary["node_count"] == 1
    assert summary["local_world_size"] == 1
    assert summary["scalability_blockers"] == ()


def test_tn_run_state_reports_the_forward_and_facade_distribution_separately() -> None:
    result = execution.run_distributed_tensor_network(
        _circuit(),
        world_size=2,
        local_world_size=1,
        distributed_profile="development",
        torch_backend="local_tensor",
        sliced_labels=_sliced_labels(),
    )
    summary = result.summary()

    assert summary["distribution_semantics"] == "slice_parallel_state_with_local_facade"
    assert summary["forward_distribution_semantics"] == "sharded_across_ranks"
    assert summary["state_distribution_semantics"] == (
        "replicated_full_state_after_all_reduce"
    )
    assert summary["full_state_materialized"] is True
    assert summary["scalability_claim_allowed"] is False
    assert summary["node_count"] == 2
    assert set(summary["scalability_blockers"]) == {
        "single_process_development_simulator",
        "full_local_tensor_network_facade",
        "expectation_methods_use_local_state",
    }
    assert summary["communication_tiers"]["inter_node_collective_possible"] is True


@pytest.mark.parametrize(
    ("backend", "world_size", "expected"),
    [
        ("nccl", 2, "production_runtime"),
        ("gloo", 2, "development_smoke"),
        ("gloo", 1, "development_smoke"),
    ],
)
def test_tn_claim_evidence_type_follows_the_executed_group(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    world_size: int,
    expected: str,
) -> None:
    """A run may only be cited for what it actually executed on."""

    _install_single_process_group(
        monkeypatch, world_size=world_size, local_world_size=1, backend=backend
    )
    summary = execution.distributed_tensor_network_expectation(
        _circuit(),
        z=(1, 3),
        world_size=world_size,
        distributed_executor="torch",
        backend=backend,
        sliced_labels=_sliced_labels(),
    ).summary()

    assert summary["claim_evidence_type"] == expected


def test_tn_claim_evidence_type_never_claims_the_slice_simulator() -> None:
    """A declared topology is not an executed one; simulated runs say so."""

    simulated = execution.distributed_tensor_network_amplitude(
        _circuit(),
        "10001",
        world_size=2,
        local_world_size=1,
        distributed_profile="development",
        torch_backend="local_tensor",
        sliced_labels=_sliced_labels(),
    ).summary()
    single = execution.distributed_tensor_network_amplitude(
        _circuit(),
        "10001",
        world_size=1,
        distributed_profile="development",
        torch_backend="local_tensor",
        sliced_labels=_sliced_labels(),
    ).summary()

    assert simulated["claim_evidence_type"] == "development_smoke"
    assert single["claim_evidence_type"] == "development_smoke"


def test_tn_training_requires_a_process_group() -> None:
    from flagquantum.runtime.executors.tensor_network.training import (
        train_distributed_tensor_network,
    )

    theta = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(3)
    circuit.ry(0, theta=theta).cx(0, 2)

    with pytest.raises(RuntimeError, match="process group"):
        train_distributed_tensor_network(
            circuit, (theta,), steps=1, observable={0: "z"}, sliced_labels=(1,)
        )
