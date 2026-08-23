"""Joint forward-peak, checkpoint, communication, and rematerialization planning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import combinations
from math import ceil, log2, prod
from typing import Any

from .distributed_dag import DistributedTNContractionDAG
from .multi_axis_sharding import _simulate_multi_axis_live_bytes

TN_JOINT_PLAN_VERSION = "flagquantum.distributed_tn_joint_plan.v3"


@dataclass(frozen=True)
class DistributedTNWorkingSetPolicy:
    """Conservative non-tensor memory allowances for one TN rank.

    The core DAG simulation accounts for live tensor values.  Production
    execution also needs temporary contraction workspace, collective buffers,
    and allocator headroom.  Keeping these terms explicit prevents a largest-
    intermediate estimate from being mistaken for an end-to-end memory bound.
    """

    kernel_workspace_output_multiplier: float = 1.0
    communication_buffer_output_multiplier: float = 2.0
    allocator_headroom_fraction: float = 0.1
    minimum_allocator_headroom_bytes: int = 256 << 20

    def validate(self) -> None:
        if self.kernel_workspace_output_multiplier < 0:
            raise ValueError(
                "TN kernel workspace output multiplier must be non-negative"
            )
        if self.communication_buffer_output_multiplier < 0:
            raise ValueError(
                "TN communication buffer output multiplier must be non-negative"
            )
        if not 0 <= self.allocator_headroom_fraction < 1:
            raise ValueError("TN allocator headroom fraction must be in [0, 1)")
        if self.minimum_allocator_headroom_bytes < 0:
            raise ValueError("TN minimum allocator headroom must be non-negative")


@dataclass(frozen=True)
class DistributedTNJointPlan:
    """One deterministic multi-objective forward/reverse execution plan."""

    version: str
    identity: str
    dag_identity: str
    target_operation_id: str
    shard_labels: tuple[int, ...]
    mesh_shape: tuple[int, ...]
    checkpoint_budget_local_bytes: int
    memory_budget_local_bytes: int | None
    checkpoint_value_ids: tuple[str, ...]
    saved_checkpoint_local_bytes: int
    predicted_forward_peak_local_bytes: int
    predicted_forward_peak_logical_bytes: int
    predicted_reverse_cotangent_peak_local_bytes: int
    predicted_rematerialization_peak_local_bytes: int
    predicted_raw_input_local_bytes: int
    predicted_reverse_peak_local_bytes: int
    predicted_tensor_working_set_local_bytes: int
    predicted_kernel_workspace_local_bytes: int
    predicted_communication_buffer_local_bytes: int
    predicted_allocator_headroom_local_bytes: int
    predicted_working_set_local_bytes: int
    working_set_policy: dict[str, Any] | None
    memory_budget_satisfied: bool
    estimated_collective_network_bytes: int
    estimated_rematerialization_cost: int
    communication_weight: int
    rematerialization_weight: int
    objective_score: int
    candidate_count: int

    def summary(self) -> dict[str, Any]:
        return asdict(self)


def plan_joint_distributed_tn_execution(
    dag: DistributedTNContractionDAG,
    *,
    checkpoint_budget_local_bytes: int,
    target_operation_id: str | None = None,
    communication_weight: int = 1,
    rematerialization_weight: int = 1,
    memory_budget_local_bytes: int | None = None,
    working_set_policy: DistributedTNWorkingSetPolicy | None = None,
) -> DistributedTNJointPlan:
    """Jointly select mesh labels and locally retained reverse checkpoints.

    ``checkpoint_budget_local_bytes`` limits retained reverse checkpoints.
    ``memory_budget_local_bytes`` is an optional hard per-rank working-set
    budget covering the simulated forward peak plus retained checkpoints.
    The latter is deliberately conservative: checkpoint storage is treated as
    live alongside the forward peak rather than relying on allocator reuse.
    """

    dag.validate()
    if checkpoint_budget_local_bytes < 0:
        raise ValueError("joint TN checkpoint budget must be non-negative")
    if memory_budget_local_bytes is not None and memory_budget_local_bytes <= 0:
        raise ValueError("joint TN memory budget must be positive")
    if communication_weight < 0 or rematerialization_weight < 0:
        raise ValueError("joint TN objective weights must be non-negative")
    if working_set_policy is not None:
        working_set_policy.validate()
    if dag.world_size <= 1 or dag.world_size & (dag.world_size - 1):
        raise ValueError("joint TN planning requires a power-of-two world size")
    target = (
        dag.operations[-1]
        if target_operation_id is None
        else next(
            (
                operation
                for operation in dag.operations
                if operation.operation_id == target_operation_id
            ),
            None,
        )
    )
    if target is None:
        raise ValueError("joint TN target operation is absent from the DAG")
    values = {value.value_id: value for value in dag.values}
    left, right = (values[value_id] for value_id in target.input_value_ids)
    contracted = tuple(
        label
        for label in left.labels
        if label in right.labels
        and label not in target.output_labels
        and left.shape[left.labels.index(label)] > 1
    )
    label_extents = {
        label: left.shape[left.labels.index(label)] for label in contracted
    }
    max_labels = min(len(contracted), int(log2(dag.world_size)))
    label_candidates = tuple(
        candidate
        for count in range(1, max_labels + 1)
        for candidate in combinations(contracted, count)
        if prod(label_extents[label] for label in candidate) == dag.world_size
    )
    ranked = []
    for labels in label_candidates:
        simulation = _simulate_multi_axis_live_bytes(
            dag,
            target_operation_id=target.operation_id,
            shard_labels=labels,
            label_extents=label_extents,
        )
        if simulation is None:
            continue
        local_peak, logical_peak, _ = simulation
        if (
            memory_budget_local_bytes is not None
            and local_peak > memory_budget_local_bytes
        ):
            continue
        effective_checkpoint_budget = checkpoint_budget_local_bytes
        if memory_budget_local_bytes is not None:
            effective_checkpoint_budget = min(
                effective_checkpoint_budget,
                memory_budget_local_bytes - local_peak,
            )
        checkpoint_candidates = []
        local_bytes_by_value = {}
        for value in dag.values:
            divisor = prod(
                label_extents[label] for label in labels if label in value.labels
            )
            local_bytes_by_value[value.value_id] = value.nbytes // divisor
        for operation in dag.operations:
            if operation.output_value_id == dag.output_value_id:
                continue
            value = values[operation.output_value_id]
            local_bytes = local_bytes_by_value[value.value_id]
            checkpoint_candidates.append(
                (
                    -(operation.estimated_cost / max(1, local_bytes)),
                    value.value_id,
                    local_bytes,
                    operation.estimated_cost,
                )
            )
        checkpoint_candidates.sort()
        saved_ids = []
        saved_bytes = 0
        saved_cost = 0
        checkpoint_prefixes = [((), 0, 0)]
        for _, value_id, local_bytes, cost in checkpoint_candidates:
            if saved_bytes + local_bytes > effective_checkpoint_budget:
                continue
            saved_ids.append(value_id)
            saved_bytes += local_bytes
            saved_cost += cost
            checkpoint_prefixes.append((tuple(saved_ids), saved_bytes, saved_cost))
        total_cost = sum(
            operation.estimated_cost
            for operation in dag.operations
            if operation.output_value_id != dag.output_value_id
        )
        output_bytes = values[target.output_value_id].nbytes
        collective_bytes = 2 * (dag.world_size - 1) * output_bytes
        cotangent_peak = _predict_reverse_cotangent_peak(dag, local_bytes_by_value)
        raw_input_bytes = sum(
            local_bytes_by_value[value.value_id]
            for value in dag.values
            if value.producer_id is None
        )
        largest_operation_output_local_bytes = max(
            local_bytes_by_value[operation.output_value_id]
            for operation in dag.operations
        )
        kernel_workspace_bytes = (
            0
            if working_set_policy is None
            else ceil(
                largest_operation_output_local_bytes
                * working_set_policy.kernel_workspace_output_multiplier
            )
        )
        communication_buffer_bytes = (
            0
            if working_set_policy is None
            else ceil(
                local_bytes_by_value[target.output_value_id]
                * working_set_policy.communication_buffer_output_multiplier
            )
        )
        for checkpoint_ids, selected_bytes, selected_cost in checkpoint_prefixes:
            rematerialization_cost = total_cost - selected_cost
            rematerialization_peak = _predict_rematerialization_peak(
                dag,
                local_bytes_by_value,
                checkpoint_ids,
            )
            forward_working_set = local_peak + selected_bytes
            reverse_peak = (
                raw_input_bytes
                + selected_bytes
                + cotangent_peak
                + rematerialization_peak
            )
            tensor_working_set_bytes = max(forward_working_set, reverse_peak)
            pre_headroom_working_set_bytes = (
                tensor_working_set_bytes
                + kernel_workspace_bytes
                + communication_buffer_bytes
            )
            allocator_headroom_bytes = (
                0
                if working_set_policy is None
                else max(
                    working_set_policy.minimum_allocator_headroom_bytes,
                    ceil(
                        pre_headroom_working_set_bytes
                        * working_set_policy.allocator_headroom_fraction
                    ),
                )
            )
            working_set_bytes = (
                pre_headroom_working_set_bytes + allocator_headroom_bytes
            )
            if (
                memory_budget_local_bytes is not None
                and working_set_bytes > memory_budget_local_bytes
            ):
                continue
            score = (
                working_set_bytes
                + communication_weight * collective_bytes
                + rematerialization_weight * rematerialization_cost
            )
            ranked.append(
                (
                    score,
                    working_set_bytes,
                    local_peak,
                    collective_bytes,
                    rematerialization_cost,
                    labels,
                    checkpoint_ids,
                    selected_bytes,
                    logical_peak,
                    cotangent_peak,
                    rematerialization_peak,
                    raw_input_bytes,
                    reverse_peak,
                    tensor_working_set_bytes,
                    kernel_workspace_bytes,
                    communication_buffer_bytes,
                    allocator_headroom_bytes,
                )
            )
    if not ranked:
        if memory_budget_local_bytes is not None:
            raise ValueError(
                "joint TN planner found no mesh candidate satisfying the "
                f"{memory_budget_local_bytes}-byte per-rank memory budget"
            )
        raise ValueError("joint TN planner found no executable mesh candidate")
    ranked.sort()
    (
        score,
        working_set_bytes,
        local_peak,
        collective_bytes,
        rematerialization_cost,
        labels,
        checkpoint_ids,
        saved_bytes,
        logical_peak,
        cotangent_peak,
        rematerialization_peak,
        raw_input_bytes,
        reverse_peak,
        tensor_working_set_bytes,
        kernel_workspace_bytes,
        communication_buffer_bytes,
        allocator_headroom_bytes,
    ) = ranked[0]
    payload = {
        "version": TN_JOINT_PLAN_VERSION,
        "dag_identity": dag.identity,
        "target_operation_id": target.operation_id,
        "shard_labels": labels,
        "mesh_shape": tuple(label_extents[label] for label in labels),
        "checkpoint_budget_local_bytes": checkpoint_budget_local_bytes,
        "memory_budget_local_bytes": memory_budget_local_bytes,
        "checkpoint_value_ids": checkpoint_ids,
        "saved_checkpoint_local_bytes": saved_bytes,
        "predicted_forward_peak_local_bytes": local_peak,
        "predicted_forward_peak_logical_bytes": logical_peak,
        "predicted_reverse_cotangent_peak_local_bytes": cotangent_peak,
        "predicted_rematerialization_peak_local_bytes": rematerialization_peak,
        "predicted_raw_input_local_bytes": raw_input_bytes,
        "predicted_reverse_peak_local_bytes": reverse_peak,
        "predicted_tensor_working_set_local_bytes": tensor_working_set_bytes,
        "predicted_kernel_workspace_local_bytes": kernel_workspace_bytes,
        "predicted_communication_buffer_local_bytes": communication_buffer_bytes,
        "predicted_allocator_headroom_local_bytes": allocator_headroom_bytes,
        "predicted_working_set_local_bytes": working_set_bytes,
        "working_set_policy": (
            None if working_set_policy is None else asdict(working_set_policy)
        ),
        "memory_budget_satisfied": (
            memory_budget_local_bytes is None
            or working_set_bytes <= memory_budget_local_bytes
        ),
        "estimated_collective_network_bytes": collective_bytes,
        "estimated_rematerialization_cost": rematerialization_cost,
        "communication_weight": communication_weight,
        "rematerialization_weight": rematerialization_weight,
        "objective_score": score,
        "candidate_count": len(ranked),
    }
    return DistributedTNJointPlan(
        **payload,
        identity=hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    )


def _predict_reverse_cotangent_peak(
    dag: DistributedTNContractionDAG,
    local_bytes_by_value: dict[str, int],
) -> int:
    """Simulate cotangent liveness in the explicit reverse order."""

    live = {dag.output_value_id}
    peak = local_bytes_by_value[dag.output_value_id]
    for operation in reversed(dag.operations):
        live.discard(operation.output_value_id)
        live.update(operation.input_value_ids)
        peak = max(
            peak,
            sum(local_bytes_by_value[value_id] for value_id in live),
        )
    return peak


def _predict_rematerialization_peak(
    dag: DistributedTNContractionDAG,
    local_bytes_by_value: dict[str, int],
    checkpoint_value_ids: tuple[str, ...],
) -> int:
    """Bound the largest minimal-subgraph recomputation transient."""

    retained = {value.value_id for value in dag.values if value.producer_id is None}
    retained.update(checkpoint_value_ids)
    producers = {operation.output_value_id: operation for operation in dag.operations}
    peak = 0
    for target_id in producers:
        if target_id in retained or target_id == dag.output_value_id:
            continue
        selected_operation_ids = set()
        pending = [target_id]
        while pending:
            value_id = pending.pop()
            if value_id in retained:
                continue
            operation = producers[value_id]
            if operation.operation_id in selected_operation_ids:
                continue
            selected_operation_ids.add(operation.operation_id)
            pending.extend(operation.input_value_ids)
        operations = tuple(
            operation
            for operation in dag.operations
            if operation.operation_id in selected_operation_ids
        )
        remaining_uses = {operation.output_value_id: 0 for operation in operations}
        for operation in operations:
            for input_id in operation.input_value_ids:
                if input_id in remaining_uses:
                    remaining_uses[input_id] += 1
        live = {}
        for operation in operations:
            live[operation.output_value_id] = local_bytes_by_value[
                operation.output_value_id
            ]
            peak = max(peak, sum(live.values()))
            for input_id in operation.input_value_ids:
                if input_id not in remaining_uses:
                    continue
                remaining_uses[input_id] -= 1
                if remaining_uses[input_id] == 0:
                    live.pop(input_id, None)
    return peak


__all__ = (
    "DistributedTNJointPlan",
    "DistributedTNWorkingSetPolicy",
    "TN_JOINT_PLAN_VERSION",
    "plan_joint_distributed_tn_execution",
)
