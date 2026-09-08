"""Run rank-owned sliced TN forward and reverse on torch.distributed."""

from __future__ import annotations

import json
import os
from collections import Counter

import torch
import torch.distributed as dist

import flagquantum as fq
from flagquantum.runtime.backends.tensor_network.distributed_optimizer import (
    execute_rank_owned_tn_sgd_step,
    plan_tn_parameter_owners,
)
from flagquantum.runtime.backends.tensor_network.distributed_sliced_reverse import (
    execute_distributed_sliced_tn_explicit_reverse,
)
from flagquantum.runtime.backends.tensor_network.sliced_tasks import (
    plan_distributed_tn_slice_tasks,
)
from flagquantum.simulation.tensor_network.entrypoints import (
    build_tensor_network_expectation,
)


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
    counts = Counter(label for node in expectation.nodes for label in node.labels)
    labels = tuple(
        label
        for label, count in counts.items()
        if count >= 2 and label not in expectation.output_labels
    )[:2]
    slicing = expectation.slicing_plan(sliced_labels=labels)
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
        zip(result.parameter_gradients, reference_gradients)
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
    )
    for parameter, initial, gradient in zip(
        (theta, phi), initial_parameters, reference_gradients
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
    print(
        json.dumps(
            {
                **summary,
                "optimizer": optimizer.summary(),
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
