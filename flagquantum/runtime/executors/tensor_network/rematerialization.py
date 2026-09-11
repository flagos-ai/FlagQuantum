"""Dependency-aware forward rematerialization on a fixed partial mesh."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from .distributed_dag import DistributedTNContractionDAG, DistributedTNContractionRecord
from .partial_mesh import (
    DistributedTNMeshGroupCache,
    DistributedTNPartialMeshLayout,
    execute_partial_mesh_forward_pair,
    plan_partial_mesh_tn_layout,
)


@dataclass(frozen=True)
class DistributedTNRematerializationPlan:
    checkpoint_value_ids: tuple[str, ...]
    checkpoint_local_bytes: int
    raw_input_local_bytes: int
    budget_bytes: int


def plan_partial_mesh_rematerialization(
    dag: DistributedTNContractionDAG,
    *,
    mesh_labels: tuple[int, ...],
    mesh_shape: tuple[int, ...],
    budget_bytes: int,
) -> DistributedTNRematerializationPlan:
    """Select high avoided-cost checkpoints under a rank-local byte budget."""

    dag.validate()
    if budget_bytes < 0:
        raise ValueError("rematerialization checkpoint budget must be non-negative")
    operations = {operation.output_value_id: operation for operation in dag.operations}
    layouts = {
        value.value_id: plan_partial_mesh_tn_layout(
            value, mesh_labels=mesh_labels, mesh_shape=mesh_shape
        )
        for value in dag.values
    }
    candidates = [
        value
        for value in dag.values
        if value.producer_id is not None
        and value.value_id != dag.output_value_id
        and layouts[value.value_id].local_nbytes <= budget_bytes
    ]
    candidates.sort(
        key=lambda value: (
            -operations[value.value_id].estimated_cost
            / max(1, layouts[value.value_id].local_nbytes),
            value.value_id,
        )
    )
    selected = []
    selected_bytes = 0
    for value in candidates:
        local_bytes = layouts[value.value_id].local_nbytes
        if selected_bytes + local_bytes <= budget_bytes:
            selected.append(value.value_id)
            selected_bytes += local_bytes
    selected_set = set(selected)
    return DistributedTNRematerializationPlan(
        checkpoint_value_ids=tuple(
            value.value_id for value in dag.values if value.value_id in selected_set
        ),
        checkpoint_local_bytes=selected_bytes,
        raw_input_local_bytes=sum(
            layouts[value.value_id].local_nbytes
            for value in dag.values
            if value.producer_id is None
        ),
        budget_bytes=budget_bytes,
    )


class DistributedTNRematerializationProvider:
    """Recompute a missing value from the nearest retained DAG ancestors."""

    def __init__(
        self,
        dag: DistributedTNContractionDAG,
        retained_values: Mapping[str, torch.Tensor],
        *,
        mesh_labels: tuple[int, ...],
        mesh_shape: tuple[int, ...],
        group_cache: DistributedTNMeshGroupCache,
        max_transient_bytes: int | None = None,
    ) -> None:
        dag.validate()
        self.dag = dag
        self.retained_values = retained_values
        self.mesh_labels = tuple(int(label) for label in mesh_labels)
        self.mesh_shape = tuple(int(extent) for extent in mesh_shape)
        self.group_cache = group_cache
        if max_transient_bytes is not None and max_transient_bytes <= 0:
            raise ValueError("rematerialization memory budget must be positive")
        self.max_transient_bytes = max_transient_bytes
        self.operation_count = 0
        self.collective_count = 0
        self.collective_bytes = 0
        self.peak_transient_bytes = 0
        self.released_transient_value_count = 0
        self.last_predicted_peak_transient_bytes = 0
        self._values = {value.value_id: value for value in dag.values}
        self._producers = {
            operation.output_value_id: operation for operation in dag.operations
        }
        unknown = set(retained_values) - set(self._values)
        if unknown:
            raise ValueError(
                f"rematerialization tape has unknown values {tuple(sorted(unknown))}"
            )

    def __call__(self, value_id: str) -> torch.Tensor:
        requested = str(value_id)
        retained = self.retained_values.get(requested)
        if retained is not None:
            return retained
        selected = self._required_operation_ids(requested)
        operations = tuple(
            operation
            for operation in self.dag.operations
            if operation.operation_id in selected
        )
        remaining_uses = {operation.output_value_id: 0 for operation in operations}
        for operation in operations:
            for input_id in operation.input_value_ids:
                if input_id in remaining_uses:
                    remaining_uses[input_id] += 1
        predicted_peak = self._predict_peak(operations, remaining_uses)
        self.last_predicted_peak_transient_bytes = predicted_peak
        if (
            self.max_transient_bytes is not None
            and predicted_peak > self.max_transient_bytes
        ):
            raise RuntimeError(
                "rematerialization predicted transient peak exceeds budget"
            )

        transient: dict[str, torch.Tensor] = {}
        for operation in operations:
            left_id, right_id = operation.input_value_ids
            left = self.retained_values.get(left_id, transient.get(left_id))
            right = self.retained_values.get(right_id, transient.get(right_id))
            if left is None or right is None:
                raise RuntimeError(
                    "rematerialization dependency schedule is incomplete"
                )
            result = execute_partial_mesh_forward_pair(
                left,
                self._layout(left_id),
                right,
                self._layout(right_id),
                self._layout(operation.output_value_id),
                group_cache=self.group_cache,
            )
            transient[operation.output_value_id] = result.value
            self.operation_count += 1
            self.collective_count += result.subgroup_collective_count
            self.collective_bytes += result.subgroup_collective_bytes
            self.peak_transient_bytes = max(
                self.peak_transient_bytes,
                sum(
                    int(tensor.numel()) * int(tensor.element_size())
                    for tensor in transient.values()
                ),
            )
            for input_id in operation.input_value_ids:
                if input_id not in remaining_uses:
                    continue
                remaining_uses[input_id] -= 1
                if remaining_uses[input_id] == 0 and input_id in transient:
                    del transient[input_id]
                    self.released_transient_value_count += 1
        return transient[requested]

    def _required_operation_ids(self, value_id: str) -> set[str]:
        selected = set()
        pending = [value_id]
        while pending:
            current = pending.pop()
            if current in self.retained_values:
                continue
            operation = self._producers.get(current)
            if operation is None:
                raise KeyError(f"rematerialization is missing raw ancestor {current!r}")
            if operation.operation_id in selected:
                continue
            selected.add(operation.operation_id)
            pending.extend(operation.input_value_ids)
        return selected

    def _layout(self, value_id: str) -> DistributedTNPartialMeshLayout:
        return plan_partial_mesh_tn_layout(
            self._values[value_id],
            mesh_labels=self.mesh_labels,
            mesh_shape=self.mesh_shape,
        )

    def _predict_peak(
        self,
        operations: Sequence[DistributedTNContractionRecord],
        remaining_uses: Mapping[str, int],
    ) -> int:
        live: dict[str, int] = {}
        peak = 0
        uses = dict(remaining_uses)
        for operation in operations:
            output_id = operation.output_value_id
            live[output_id] = self._layout(output_id).local_nbytes
            peak = max(peak, sum(live.values()))
            for input_id in operation.input_value_ids:
                if input_id not in uses:
                    continue
                uses[input_id] -= 1
                if uses[input_id] == 0:
                    live.pop(input_id, None)
        return peak


__all__ = (
    "DistributedTNRematerializationPlan",
    "DistributedTNRematerializationProvider",
    "plan_partial_mesh_rematerialization",
)
