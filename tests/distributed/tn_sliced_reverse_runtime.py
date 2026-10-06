"""Run rank-owned sliced TN forward and reverse on torch.distributed."""

from __future__ import annotations

import json
import os
from collections import Counter

import torch
import torch.distributed as dist

import flagquantum as fq
from flagquantum.runtime.executors.tensor_network.distributed_optimizer import (
    execute_rank_owned_tn_sgd_step,
    plan_tn_parameter_owners,
    plan_tn_parameter_ownership,
)
from flagquantum.runtime.executors.tensor_network.distributed_sliced_reverse import (
    execute_distributed_sliced_tn_explicit_reverse,
)
from flagquantum.runtime.executors.tensor_network.sliced_tasks import (
    plan_distributed_tn_slice_tasks,
)
from flagquantum.simulation.tensor_network.entrypoints import (
    build_tensor_network_expectation,
)


def _slicing_for(expectation):
    counts = Counter(label for node in expectation.nodes for label in node.labels)
    labels = tuple(
        label
        for label, count in counts.items()
        if count >= 2 and label not in expectation.output_labels
    )[:2]
    return expectation.slicing_plan(sliced_labels=labels)


def _four_parameter_circuit():
    """Four parameters, all with a nonzero gradient, across two ranks.

    An owner-scoped reduction can only be distinguished from a replicated one
    when every owned gradient is materially nonzero, so the two fixed rotations
    keep all four parameters entangled with the ``Z0 Z2`` observable.
    """

    parameters = tuple(
        torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for value in (0.23, -0.37, 0.11, 0.53)
    )
    circuit = fq.Circuit(4)
    circuit.ry(0, theta=parameters[0])
    circuit.ry(1, theta=parameters[1])
    circuit.rzz(0, 1, theta=0.4)
    circuit.cx(1, 2)
    circuit.ry(2, theta=parameters[2])
    circuit.rzz(2, 3, theta=0.9)
    circuit.ry(3, theta=parameters[3])
    circuit.cx(3, 0)
    return circuit, parameters


def _check_owner_scoped_gradients(world_size: int) -> dict[str, object]:
    """An owner-scoped reduction must leave each gradient on exactly one rank."""

    circuit, parameters = _four_parameter_circuit()
    expectation = build_tensor_network_expectation(circuit, z=[0, 2])
    slicing = _slicing_for(expectation)
    tasks = plan_distributed_tn_slice_tasks(
        slicing, world_size=world_size, local_world_size=world_size
    )
    result = execute_distributed_sliced_tn_explicit_reverse(
        expectation,
        slicing,
        tasks,
        parameters,
        gradient_reduction="owner_reduce",
        slice_batch_size=2,
    )
    reference = build_tensor_network_expectation(circuit, z=[0, 2])
    reference_gradients = torch.autograd.grad(
        reference.contract(strategy="greedy").real, parameters
    )
    owners = plan_tn_parameter_owners(len(parameters), world_size)
    if len(set(owners)) != world_size or max(owners.count(r) for r in set(owners)) < 2:
        raise RuntimeError(
            "the ownership check needs at least one rank owning two parameters"
        )
    reference_values = [float(value) for value in reference_gradients]
    if any(abs(value) <= 1e-6 for value in reference_values):
        raise RuntimeError(
            "every parameter needs a materially nonzero gradient for this check "
            f"to distinguish an owner-scoped reduction, got {reference_values}"
        )
    local = [
        float(gradient.detach().reshape(-1)[0])
        for gradient in result.parameter_gradients
    ]
    gathered: list[list[float] | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    rank = dist.get_rank()
    rows = [row for row in gathered if row is not None]
    if len(rows) != world_size:
        raise RuntimeError("every rank must report its reduced gradients")
    for index, owner in enumerate(owners):
        if owner == rank:
            if abs(local[index] - reference_values[index]) > 1e-9:
                raise RuntimeError(
                    f"parameter {index} owner {owner} holds gradient "
                    f"{local[index]} instead of the reduced {reference_values[index]}"
                )
        elif abs(local[index]) > 1e-9:
            raise RuntimeError(
                f"parameter {index} is owned by rank {owner}, but rank {rank} "
                f"materialized {local[index]} of its reduced gradient"
            )
    materialized = sum(1 for value in local if abs(value) > 1e-9)
    if materialized != sum(1 for owner in owners if owner == rank):
        raise RuntimeError(
            f"rank {rank} materialized {materialized} nonzero reduced gradients "
            "but owns a different number of parameters"
        )
    return {
        "owners": owners,
        "reference_gradients": reference_values,
        "gathered_gradients": rows,
        "gradient_distribution_semantics": (
            result.summary()["gradient_distribution_semantics"]
        ),
        "gradient_aggregation_semantics": (
            result.summary()["gradient_aggregation_semantics"]
        ),
    }


def _check_replicated_gradients(world_size: int) -> dict[str, object]:
    """A full all-reduce must not be reported as owner-scoped gradients."""

    circuit, parameters = _four_parameter_circuit()
    expectation = build_tensor_network_expectation(circuit, z=[0, 2])
    slicing = _slicing_for(expectation)
    tasks = plan_distributed_tn_slice_tasks(
        slicing, world_size=world_size, local_world_size=world_size
    )
    result = execute_distributed_sliced_tn_explicit_reverse(
        expectation,
        slicing,
        tasks,
        parameters,
        gradient_reduction="all_reduce",
        slice_batch_size=2,
    )
    reference = build_tensor_network_expectation(circuit, z=[0, 2])
    reference_gradients = torch.autograd.grad(
        reference.contract(strategy="greedy").real, parameters
    )
    local = [
        float(gradient.detach().reshape(-1)[0])
        for gradient in result.parameter_gradients
    ]
    gathered: list[list[float] | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    rows = [row for row in gathered if row is not None]
    for index, expected in enumerate(reference_gradients):
        for row in rows:
            if abs(row[index] - float(expected)) > 1e-9:
                raise RuntimeError(
                    f"all_reduce did not replicate gradient {index}: "
                    f"{row[index]} != {float(expected)}"
                )
    step = execute_rank_owned_tn_sgd_step(
        parameters,
        result.parameter_gradients,
        learning_rate=0.01,
        gradient_reduction="all_reduce",
    )
    summary = step.summary()
    if summary["gradient_ownership_semantics"] != "replicated_after_all_reduce":
        raise RuntimeError(
            "a replicated gradient was reported as owner-scoped: "
            f"{summary['gradient_ownership_semantics']}"
        )
    if summary["gradient_ownership"]:
        raise RuntimeError(
            "a replicated gradient must not publish a per-rank ownership map"
        )
    if summary["optimizer_update_ownership_semantics"] != "sharded_across_ranks":
        raise RuntimeError(
            "the owner-applied update is owner-scoped regardless of the reduction"
        )
    return {
        "replicated_gradients": rows,
        "optimizer": summary,
    }


def main() -> None:
    dist.init_process_group("gloo")
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError("sliced TN reverse semantic runtime requires two ranks")

    theta = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    phi = torch.tensor(-0.37, dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(4)
    circuit.ry(0, theta=theta).cx(0, 3).rzz(1, 2, theta=phi).cx(2, 3)
    expectation = build_tensor_network_expectation(circuit, z=[0, 2])
    slicing = _slicing_for(expectation)
    tasks = plan_distributed_tn_slice_tasks(
        slicing,
        world_size=world_size,
        local_world_size=world_size,
    )
    result = execute_distributed_sliced_tn_explicit_reverse(
        expectation,
        slicing,
        tasks,
        (theta, phi),
        gradient_reduction="owner_reduce",
        slice_batch_size=2,
    )

    reference_value = expectation.contract(strategy="greedy")
    reference_gradients = torch.autograd.grad(reference_value.real, (theta, phi))
    torch.testing.assert_close(result.value, reference_value, atol=1e-10, rtol=1e-10)
    owners = plan_tn_parameter_owners(2, world_size)
    for index, (actual, expected) in enumerate(
        zip(result.parameter_gradients, reference_gradients, strict=True)
    ):
        if owners[index] == dist.get_rank():
            torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
        else:
            torch.testing.assert_close(actual, torch.zeros_like(actual))
    initial_parameters = tuple(parameter.detach().clone() for parameter in (theta, phi))
    optimizer = execute_rank_owned_tn_sgd_step(
        (theta, phi),
        result.parameter_gradients,
        learning_rate=0.01,
        gradient_reduction="owner_reduce",
    )
    for parameter, initial, gradient in zip(
        (theta, phi), initial_parameters, reference_gradients, strict=True
    ):
        torch.testing.assert_close(parameter, initial - 0.01 * gradient)
    gathered_parameters: list[tuple[torch.Tensor, ...] | None] = [None] * world_size
    dist.all_gather_object(
        gathered_parameters,
        tuple(parameter.detach().cpu() for parameter in (theta, phi)),
    )
    if any(values != gathered_parameters[0] for values in gathered_parameters[1:]):
        raise RuntimeError("optimizer parameter broadcast diverged across ranks")
    summary = result.summary()
    if summary["distribution_semantics"] != "sharded_across_ranks":
        raise RuntimeError("sliced TN execution did not report sharded semantics")
    if summary["gradient_distribution_semantics"] != "sharded_across_ranks":
        raise RuntimeError("sliced TN gradients did not report sharded semantics")
    if result.local_task_count != slicing.n_slices // world_size:
        raise RuntimeError("rank did not execute exactly its owned TN slices")

    owner_scoped = _check_owner_scoped_gradients(world_size)
    replicated = _check_replicated_gradients(world_size)
    if owner_scoped["gradient_distribution_semantics"] != "sharded_across_ranks":
        raise RuntimeError("an owner-scoped reduction did not report sharded gradients")
    if replicated["optimizer"]["gradient_ownership_semantics"] == (
        "sharded_across_ranks"
    ):
        raise RuntimeError("a replicated reduction was reported as owner-scoped")

    ownership = plan_tn_parameter_ownership(len(owner_scoped["owners"]), world_size)
    print(
        json.dumps(
            {
                **summary,
                "optimizer": optimizer.summary(),
                "parameter_ownership": ownership,
                "owner_scoped_gradient_check": owner_scoped,
                "replicated_gradient_check": replicated,
                "rank_owned_sliced_reverse_passed": True,
                "pid": os.getpid(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
