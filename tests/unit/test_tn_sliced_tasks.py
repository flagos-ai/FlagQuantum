"""Contracts for deterministic distributed TN slice ownership."""

from __future__ import annotations

from collections import Counter

import pytest

import flagquantum as fq
from flagquantum.runtime.backends.tensor_network import (
    estimate_sliced_tn_full_tape_bytes,
    plan_distributed_tn_slice_tasks,
    plan_sliced_tn_checkpoint_memory,
)
from flagquantum.simulation.tensor_execution import build_tensor_network_expectation
from flagquantum.simulation.tensor_network.models import TensorNetworkSlicingPlan


def _slicing() -> TensorNetworkSlicingPlan:
    circuit = fq.Circuit(5)
    circuit.h(0).cx(0, 4).ry(1, theta=0.2).rzz(2, 3, theta=-0.4).cx(1, 3)
    plan = build_tensor_network_expectation(circuit, z=[0, 2, 4])
    counts = Counter(label for node in plan.nodes for label in node.labels)
    labels = tuple(
        label
        for label, count in counts.items()
        if count >= 2 and label not in plan.output_labels
    )[:4]
    return plan.slicing_plan(sliced_labels=labels)


def test_distributed_tn_slice_task_plan_is_complete_and_deterministic():
    slicing = _slicing()
    plan = plan_distributed_tn_slice_tasks(
        slicing,
        world_size=8,
        local_world_size=8,
    )
    repeated = plan_distributed_tn_slice_tasks(
        slicing,
        world_size=8,
        local_world_size=8,
    )

    plan.validate()
    assert plan == repeated
    assert plan.identity == repeated.identity
    assert len(plan.tasks) == slicing.n_slices
    assert len({task.assignments for task in plan.tasks}) == slicing.n_slices
    assert {task.owner_node for task in plan.tasks} == {0}
    assert all(task.estimated_flops == slicing.per_slice_cost for task in plan.tasks)
    assert all(task.estimated_peak_bytes == slicing.peak_bytes for task in plan.tasks)
    summary = plan.summary()
    assert summary["distribution_semantics"] == "planned_rank_owned_tn_slices"
    assert summary["scalability_claim_allowed"] is False
    assert sum(summary["tasks_by_rank"].values()) == slicing.n_slices
    assert summary["estimated_load_imbalance"] == 0.0


def test_distributed_tn_slice_task_plan_maps_two_nodes():
    plan = plan_distributed_tn_slice_tasks(
        _slicing(),
        world_size=16,
        local_world_size=8,
    )

    assert plan.node_count == 2
    assert {task.owner_node for task in plan.tasks} == {0, 1}
    assert all(
        task.owner_rank
        == task.owner_node * plan.local_world_size + task.owner_local_rank
        for task in plan.tasks
    )


def test_distributed_tn_slice_task_plan_rejects_invalid_topology():
    with pytest.raises(ValueError, match="divisible"):
        plan_distributed_tn_slice_tasks(
            _slicing(),
            world_size=8,
            local_world_size=3,
        )


def test_sliced_reverse_full_tape_estimate_is_deterministic():
    circuit = fq.Circuit(5)
    circuit.h(0).cx(0, 4).ry(1, theta=0.2).rzz(2, 3, theta=-0.4)
    plan = build_tensor_network_expectation(circuit, z=[0, 2, 4])
    counts = Counter(label for node in plan.nodes for label in node.labels)
    labels = tuple(
        label
        for label, count in counts.items()
        if count >= 2 and label not in plan.output_labels
    )[:2]
    slicing = plan.slicing_plan(sliced_labels=labels)

    first = estimate_sliced_tn_full_tape_bytes(plan, slicing)
    repeated = estimate_sliced_tn_full_tape_bytes(plan, slicing)

    assert first == repeated
    assert first >= slicing.peak_bytes
    checkpointed = plan_sliced_tn_checkpoint_memory(
        plan,
        slicing,
        checkpoint_budget_bytes=0,
    )
    assert checkpointed.saved_checkpoint_bytes == 0
    assert checkpointed.forward_peak_bytes >= slicing.peak_bytes
    assert checkpointed.rematerialization_peak_bytes > 0
    assert checkpointed.predicted_working_set_bytes >= (
        checkpointed.reverse_cotangent_peak_bytes
    )
    with pytest.raises(ValueError, match="do not match"):
        estimate_sliced_tn_full_tape_bytes(
            plan,
            slicing,
            assignments=tuple(reversed(tuple(zip(labels, (0, 0))))),
        )
