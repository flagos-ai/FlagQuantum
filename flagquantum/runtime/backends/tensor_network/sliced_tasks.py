"""Versioned ownership planning for independent tensor-network slices."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import product

from ....simulation.tensor_models import TensorNetworkSlicingPlan

TN_SLICE_TASK_PLAN_VERSION = "flagquantum.tn_slice_task_plan.v1"


@dataclass(frozen=True)
class DistributedTNSliceTask:
    """One complete slice assignment with a unique execution owner."""

    task_id: str
    task_index: int
    owner_rank: int
    owner_node: int
    owner_local_rank: int
    assignments: tuple[tuple[int, int], ...]
    estimated_flops: int
    estimated_peak_bytes: int


@dataclass(frozen=True)
class DistributedTNSliceTaskPlan:
    """Auditable, deterministic ownership plan for one sliced TN workload."""

    version: str
    identity: str
    world_size: int
    local_world_size: int
    node_count: int
    slicing_labels: tuple[int, ...]
    slice_shape: tuple[int, ...]
    tasks: tuple[DistributedTNSliceTask, ...]
    planning_only: bool = True

    def validate(self) -> None:
        if self.version != TN_SLICE_TASK_PLAN_VERSION:
            raise ValueError("unsupported TN slice task plan version")
        if self.world_size < 1 or self.local_world_size < 1:
            raise ValueError("TN slice task plan sizes must be positive")
        if self.world_size % self.local_world_size:
            raise ValueError("world_size must be divisible by local_world_size")
        if self.node_count != self.world_size // self.local_world_size:
            raise ValueError("TN slice task plan node count is inconsistent")
        expected_assignments = tuple(
            tuple(zip(self.slicing_labels, values))
            for values in (
                product(*(range(size) for size in self.slice_shape))
                if self.slice_shape
                else ((),)
            )
        )
        if len(self.tasks) != len(expected_assignments):
            raise ValueError("TN slice task plan does not cover every slice")
        if tuple(task.task_index for task in self.tasks) != tuple(
            range(len(self.tasks))
        ):
            raise ValueError("TN slice task indices must be contiguous")
        if len({task.task_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("TN slice task ids must be unique")
        if tuple(task.assignments for task in self.tasks) != expected_assignments:
            raise ValueError("TN slice task assignments are incomplete or duplicated")
        for task in self.tasks:
            if not 0 <= task.owner_rank < self.world_size:
                raise ValueError("TN slice task owner rank is out of range")
            if task.owner_node != task.owner_rank // self.local_world_size:
                raise ValueError("TN slice task node ownership is inconsistent")
            if task.owner_local_rank != task.owner_rank % self.local_world_size:
                raise ValueError("TN slice task local-rank ownership is inconsistent")
            if task.estimated_flops < 0 or task.estimated_peak_bytes < 0:
                raise ValueError("TN slice task estimates must be non-negative")
        if self.identity != _identity(self._payload()):
            raise ValueError("TN slice task plan identity does not match its contents")

    def _payload(self) -> dict[str, object]:
        return {
            "version": self.version,
            "world_size": self.world_size,
            "local_world_size": self.local_world_size,
            "node_count": self.node_count,
            "slicing_labels": self.slicing_labels,
            "slice_shape": self.slice_shape,
            "tasks": tuple(asdict(task) for task in self.tasks),
            "planning_only": self.planning_only,
        }

    def summary(self) -> dict[str, object]:
        self.validate()
        tasks_by_rank = {
            rank: sum(task.owner_rank == rank for task in self.tasks)
            for rank in range(self.world_size)
        }
        flops_by_rank = {
            rank: sum(
                task.estimated_flops for task in self.tasks if task.owner_rank == rank
            )
            for rank in range(self.world_size)
        }
        nonzero = tuple(value for value in flops_by_rank.values() if value > 0)
        imbalance = (
            0.0
            if not nonzero
            else float(max(nonzero) - min(nonzero)) / float(max(nonzero))
        )
        return {
            **self._payload(),
            "identity": self.identity,
            "task_count": len(self.tasks),
            "tasks_by_rank": tasks_by_rank,
            "estimated_flops_by_rank": flops_by_rank,
            "estimated_load_imbalance": imbalance,
            "distribution_semantics": "planned_rank_owned_tn_slices",
            "scalability_claim_allowed": False,
        }


def plan_distributed_tn_slice_tasks(
    slicing: TensorNetworkSlicingPlan,
    *,
    world_size: int,
    local_world_size: int,
) -> DistributedTNSliceTaskPlan:
    """Assign complete independent slices to topology-aware rank owners."""

    if world_size < 1 or local_world_size < 1:
        raise ValueError("TN slice task plan sizes must be positive")
    if world_size % local_world_size:
        raise ValueError("world_size must be divisible by local_world_size")
    assignments = tuple(
        tuple(zip(slicing.sliced_labels, values))
        for values in (
            product(*(range(size) for size in slicing.slice_shape))
            if slicing.slice_shape
            else ((),)
        )
    )
    if len(assignments) != slicing.n_slices:
        raise ValueError("slicing plan count does not match its assignments")
    tasks = tuple(
        DistributedTNSliceTask(
            task_id=f"slice:{task_index:012d}",
            task_index=task_index,
            owner_rank=task_index % world_size,
            owner_node=(task_index % world_size) // local_world_size,
            owner_local_rank=(task_index % world_size) % local_world_size,
            assignments=task_assignments,
            estimated_flops=slicing.per_slice_cost,
            estimated_peak_bytes=slicing.peak_bytes,
        )
        for task_index, task_assignments in enumerate(assignments)
    )
    payload = {
        "version": TN_SLICE_TASK_PLAN_VERSION,
        "world_size": int(world_size),
        "local_world_size": int(local_world_size),
        "node_count": int(world_size) // int(local_world_size),
        "slicing_labels": slicing.sliced_labels,
        "slice_shape": slicing.slice_shape,
        "tasks": tuple(asdict(task) for task in tasks),
        "planning_only": True,
    }
    plan = DistributedTNSliceTaskPlan(
        identity=_identity(payload),
        version=TN_SLICE_TASK_PLAN_VERSION,
        world_size=int(world_size),
        local_world_size=int(local_world_size),
        node_count=int(world_size) // int(local_world_size),
        slicing_labels=slicing.sliced_labels,
        slice_shape=slicing.slice_shape,
        tasks=tasks,
    )
    plan.validate()
    return plan


def _identity(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
