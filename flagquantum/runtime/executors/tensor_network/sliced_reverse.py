"""Single-device sliced forward and explicit reverse contraction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from math import prod
from typing import Sequence

import torch

from ....simulation.tensor_network.contraction import (
    _canonicalize_unit_extent_nodes,
    _slice_nodes,
)
from ....simulation.tensor_network.models import (
    TensorNetworkContractionPlan,
    TensorNetworkExpectationPlan,
    TensorNetworkNode,
    TensorNetworkSlicingPlan,
)
from ....simulation.tensor_network.path_search import _label_dims
from ....simulation.tensor_network.stages import kahan_add
from .checkpointing import (
    execute_checkpointed_tn_reverse_dag,
    execute_tn_forward_with_checkpoint_tape,
    plan_tn_checkpoints,
)
from .compiled_execution import (
    execute_compiled_tn_forward_with_tape,
    execute_compiled_tn_reverse_dag,
)
from .distributed_dag import (
    DistributedTNContractionDAG,
    plan_distributed_tn_contraction_dag,
)
from .joint_planning import (
    _predict_rematerialization_peak,
    _predict_reverse_cotangent_peak,
)
from .reverse_dag import (
    execute_explicit_tn_reverse_dag,
    plan_explicit_tn_reverse_dag,
)


@dataclass(frozen=True)
class SlicedTNExplicitReverseResult:
    """Auditable result of local sliced forward, reverse, and pullback."""

    value: torch.Tensor
    parameter_gradients: tuple[torch.Tensor | None, ...]
    slice_count: int
    forward_operation_count: int
    reverse_operation_count: int
    nonfinite_cotangent_count: int
    nonfinite_parameter_gradient_count: int
    saved_tape_bytes: int
    rematerialized_operation_count: int
    reduction_method: str = "kahan_compensated"
    execution_semantics: str = "single_device_sliced_reverse"
    scalability_claim_allowed: bool = False

    def summary(self) -> dict[str, object]:
        return {
            "slice_count": self.slice_count,
            "forward_operation_count": self.forward_operation_count,
            "reverse_operation_count": self.reverse_operation_count,
            "nonfinite_cotangent_count": self.nonfinite_cotangent_count,
            "nonfinite_parameter_gradient_count": (
                self.nonfinite_parameter_gradient_count
            ),
            "saved_tape_bytes": self.saved_tape_bytes,
            "rematerialized_operation_count": (self.rematerialized_operation_count),
            "reduction_method": self.reduction_method,
            "execution_semantics": self.execution_semantics,
            "scalability_claim_allowed": self.scalability_claim_allowed,
        }


@dataclass(frozen=True)
class SlicedTNCheckpointMemoryPlan:
    """Conservative one-slice checkpoint/reverse working-set estimate."""

    checkpoint_budget_bytes: int
    checkpoint_value_ids: tuple[str, ...]
    saved_checkpoint_bytes: int
    logical_forward_peak_bytes: int
    forward_peak_bytes: int
    forward_workspace_multiplier: int
    raw_input_bytes: int
    reverse_cotangent_peak_bytes: int
    rematerialization_peak_bytes: int
    predicted_working_set_bytes: int

    def summary(self, *, include_value_ids: bool = True) -> dict[str, object]:
        summary: dict[str, object] = {
            "checkpoint_budget_bytes": self.checkpoint_budget_bytes,
            "checkpoint_count": len(self.checkpoint_value_ids),
            "saved_checkpoint_bytes": self.saved_checkpoint_bytes,
            "logical_forward_peak_bytes": self.logical_forward_peak_bytes,
            "forward_peak_bytes": self.forward_peak_bytes,
            "forward_workspace_multiplier": self.forward_workspace_multiplier,
            "raw_input_bytes": self.raw_input_bytes,
            "reverse_cotangent_peak_bytes": (self.reverse_cotangent_peak_bytes),
            "rematerialization_peak_bytes": self.rematerialization_peak_bytes,
            "predicted_working_set_bytes": self.predicted_working_set_bytes,
        }
        if include_value_ids:
            summary["checkpoint_value_ids"] = self.checkpoint_value_ids
        return summary


def _sliced_tn_dag(
    plan: TensorNetworkContractionPlan | TensorNetworkExpectationPlan,
    slicing: TensorNetworkSlicingPlan,
    assignments: Sequence[tuple[int, int]] | None,
) -> DistributedTNContractionDAG:
    if not slicing.budget_satisfied:
        raise ValueError("cannot plan tape for an unsatisfied slicing plan")
    source_nodes = plan.nodes
    source_outputs = plan.output_labels
    if slicing.canonicalize_unit_extent_labels:
        source_nodes, source_outputs = _canonicalize_unit_extent_nodes(
            source_nodes, source_outputs
        )
    dims = _label_dims(source_nodes)
    selected = (
        tuple((int(label), int(value)) for label, value in assignments)
        if assignments is not None
        else tuple((label, 0) for label in slicing.sliced_labels)
    )
    if tuple(label for label, _ in selected) != slicing.sliced_labels:
        raise ValueError("TN tape estimate assignments do not match slicing labels")
    if any(value < 0 or value >= dims[label] for label, value in selected):
        raise ValueError("TN tape estimate assignment is outside the label extent")
    subnodes = _slice_nodes(source_nodes, dict(selected))
    subplan = replace(plan, nodes=subnodes, output_labels=source_outputs, path=())
    return plan_distributed_tn_contraction_dag(
        subplan,
        world_size=1,
        objective=(
            "external_cotengra" if slicing.contraction_path else "quality_multistart"
        ),
        small_tensor_replication_bytes=1 << 62,
        pair_steps=slicing.contraction_path or None,
    )


def estimate_sliced_tn_full_tape_bytes(
    plan: TensorNetworkContractionPlan | TensorNetworkExpectationPlan,
    slicing: TensorNetworkSlicingPlan,
    *,
    assignments: Sequence[tuple[int, int]] | None = None,
) -> int:
    """Estimate bytes retained by the current full-tape reverse for one slice."""

    if not slicing.budget_satisfied:
        raise ValueError("cannot estimate tape for an unsatisfied slicing plan")
    dag = _sliced_tn_dag(plan, slicing, assignments)
    return sum(value.nbytes for value in dag.values)


def plan_sliced_tn_checkpoint_memory(
    plan: TensorNetworkContractionPlan | TensorNetworkExpectationPlan,
    slicing: TensorNetworkSlicingPlan,
    *,
    checkpoint_budget_bytes: int,
    assignments: Sequence[tuple[int, int]] | None = None,
) -> SlicedTNCheckpointMemoryPlan:
    """Plan a conservative complete reverse working set for one slice."""

    if checkpoint_budget_bytes < 0:
        raise ValueError("sliced TN checkpoint budget must be non-negative")
    dag = _sliced_tn_dag(plan, slicing, assignments)
    checkpoints = plan_tn_checkpoints(
        dag,
        budget_bytes=checkpoint_budget_bytes,
    )
    local_bytes = {value.value_id: value.nbytes for value in dag.values}
    raw_input_bytes = sum(
        value.nbytes for value in dag.values if value.producer_id is None
    )
    cotangent_peak = _predict_reverse_cotangent_peak(dag, local_bytes)
    rematerialization_peak = _predict_rematerialization_peak(
        dag,
        local_bytes,
        checkpoints.checkpoint_value_ids,
    )
    reverse_peak = (
        raw_input_bytes
        + checkpoints.saved_intermediate_bytes
        + cotangent_peak
        + rematerialization_peak
    )
    logical_forward_peak = _predict_checkpoint_forward_peak(
        dag,
        local_bytes,
        checkpoints.checkpoint_value_ids,
    )
    forward_workspace_multiplier = 2
    forward_peak = logical_forward_peak * forward_workspace_multiplier
    predicted = max(forward_peak, reverse_peak)
    return SlicedTNCheckpointMemoryPlan(
        checkpoint_budget_bytes=checkpoint_budget_bytes,
        checkpoint_value_ids=checkpoints.checkpoint_value_ids,
        saved_checkpoint_bytes=checkpoints.saved_intermediate_bytes,
        logical_forward_peak_bytes=logical_forward_peak,
        forward_peak_bytes=forward_peak,
        forward_workspace_multiplier=forward_workspace_multiplier,
        raw_input_bytes=raw_input_bytes,
        reverse_cotangent_peak_bytes=cotangent_peak,
        rematerialization_peak_bytes=rematerialization_peak,
        predicted_working_set_bytes=predicted,
    )


def _predict_checkpoint_forward_peak(
    dag: DistributedTNContractionDAG,
    local_bytes: dict[str, int],
    checkpoint_value_ids: tuple[str, ...],
) -> int:
    """Simulate allocation-before-release in the streaming checkpoint forward."""

    retained = {value.value_id for value in dag.values if value.producer_id is None}
    retained.update(checkpoint_value_ids)
    retained.add(dag.output_value_id)
    remaining_uses = {value.value_id: 0 for value in dag.values}
    for operation in dag.operations:
        for value_id in operation.input_value_ids:
            remaining_uses[value_id] += 1
    live = {value.value_id for value in dag.values if value.producer_id is None}
    peak = sum(local_bytes[value_id] for value_id in live)
    for operation in dag.operations:
        live.add(operation.output_value_id)
        peak = max(peak, sum(local_bytes[value_id] for value_id in live))
        for value_id in operation.input_value_ids:
            remaining_uses[value_id] -= 1
            if remaining_uses[value_id] == 0 and value_id not in retained:
                live.remove(value_id)
    return peak


def execute_sliced_tn_explicit_reverse(
    plan: TensorNetworkContractionPlan | TensorNetworkExpectationPlan,
    slicing: TensorNetworkSlicingPlan,
    parameters: Sequence[torch.Tensor],
    *,
    output_cotangent: torch.Tensor | None = None,
    checkpoint_budget_bytes: int | None = None,
    compiled_reverse: bool = False,
    deferred_parameter_pullback: bool = False,
    slice_batch_size: int = 1,
    _task_assignments: Sequence[Sequence[tuple[int, int]]] | None = None,
) -> SlicedTNExplicitReverseResult:
    """Run every slice through the existing explicit DAG reverse path.

    This is a single-device development path. Each slice obtains all requested
    parameter gradients in one pullback; contraction is never repeated once per
    parameter.
    """

    if not slicing.budget_satisfied:
        raise ValueError("cannot execute a slicing plan with an unsatisfied budget")
    parameters = tuple(parameters)
    if any(not parameter.requires_grad for parameter in parameters):
        raise ValueError("all sliced TN pullback parameters must require gradients")
    if checkpoint_budget_bytes is not None and checkpoint_budget_bytes < 0:
        raise ValueError("sliced TN checkpoint budget must be non-negative")
    batch_size = int(slice_batch_size)
    if batch_size <= 0:
        raise ValueError("slice_batch_size must be positive")
    if deferred_parameter_pullback and batch_size != 1:
        raise ValueError("deferred parameter pullback does not support slice batching")
    source_nodes = plan.nodes
    source_outputs = plan.output_labels
    if slicing.canonicalize_unit_extent_labels:
        source_nodes, source_outputs = _canonicalize_unit_extent_nodes(
            source_nodes, source_outputs
        )
    dims = _label_dims(source_nodes)
    expected_shape = tuple(dims[label] for label in slicing.sliced_labels)
    if expected_shape != slicing.slice_shape:
        raise ValueError("slicing plan shape does not match the tensor network")
    expected_slices = prod(expected_shape) if expected_shape else 1
    if expected_slices != slicing.n_slices:
        raise ValueError("slicing plan count does not match its slice shape")

    value_total: torch.Tensor | None = None
    value_compensation: torch.Tensor | None = None
    gradient_totals: list[torch.Tensor | None] = [None] * len(parameters)
    gradient_compensations: list[torch.Tensor | None] = [None] * len(parameters)
    deferred_node_totals: list[torch.Tensor | None] = [None] * len(source_nodes)
    deferred_node_compensations: list[torch.Tensor | None] = [None] * len(source_nodes)
    forward_operations = 0
    reverse_operations = 0
    nonfinite_cotangents = 0
    saved_tape_bytes = 0
    rematerialized_operations = 0

    if _task_assignments is None:
        value_ranges = [range(dims[label]) for label in slicing.sliced_labels]
        assignment_records = tuple(
            tuple(zip(slicing.sliced_labels, values))
            for values in (product(*value_ranges) if value_ranges else ((),))
        )
    else:
        assignment_records = tuple(
            tuple((int(label), int(value)) for label, value in assignments)
            for assignments in _task_assignments
        )
    if len(assignment_records) % batch_size:
        raise ValueError("slice_batch_size must divide the local slice count")
    assignment_batches = tuple(
        assignment_records[index : index + batch_size]
        for index in range(0, len(assignment_records), batch_size)
    )
    batch_label = max(dims, default=-1) + 1

    def sliced_batch(
        records: Sequence[Sequence[tuple[int, int]]],
    ) -> tuple[TensorNetworkNode, ...]:
        members = tuple(_slice_nodes(source_nodes, dict(record)) for record in records)
        if len(members) == 1:
            return members[0]
        return tuple(
            TensorNetworkNode(
                tensor=torch.stack(tuple(member[index].tensor for member in members)),
                labels=(batch_label, *members[0][index].labels),
                name=members[0][index].name,
                metadata=members[0][index].metadata,
            )
            for index in range(len(source_nodes))
        )

    template_subnodes = sliced_batch(assignment_batches[0])
    batched_outputs = (
        source_outputs if batch_size == 1 else (batch_label, *source_outputs)
    )
    pair_steps = slicing.contraction_path or None
    if batch_size != 1 and pair_steps is not None:
        pair_steps = tuple(
            replace(
                step,
                left_labels=(batch_label, *step.left_labels),
                right_labels=(batch_label, *step.right_labels),
                output_labels=(batch_label, *step.output_labels),
                output_shape=(batch_size, *step.output_shape),
                intermediate_size=batch_size * step.intermediate_size,
            )
            for step in pair_steps
        )
    template_plan = replace(
        plan,
        nodes=template_subnodes,
        output_labels=batched_outputs,
        path=(),
    )
    dag = plan_distributed_tn_contraction_dag(
        template_plan,
        world_size=1,
        objective=(
            "external_cotengra" if slicing.contraction_path else "quality_multistart"
        ),
        small_tensor_replication_bytes=1 << 62,
        pair_steps=pair_steps,
    )
    reverse = plan_explicit_tn_reverse_dag(dag)
    full_tape_bytes = sum(value.nbytes for value in dag.values)
    checkpoints = (
        None
        if checkpoint_budget_bytes is None or full_tape_bytes <= checkpoint_budget_bytes
        else plan_tn_checkpoints(
            dag,
            budget_bytes=checkpoint_budget_bytes,
        )
    )
    template_labels = tuple(node.labels for node in template_subnodes)
    for assignment_batch in assignment_batches:
        assignments = dict(assignment_batch[0])
        subnodes = sliced_batch(assignment_batch)
        if tuple(node.labels for node in subnodes) != template_labels:
            raise RuntimeError("TN slice assignments changed the contraction topology")
        detached_inputs = {
            f"input:{index}": node.tensor.detach()
            for index, node in enumerate(subnodes)
        }
        tape = (
            execute_compiled_tn_forward_with_tape(dag, detached_inputs)
            if checkpoints is None
            else execute_tn_forward_with_checkpoint_tape(
                dag,
                detached_inputs,
                checkpoints,
            )
        )
        output = tape[dag.output_value_id]
        reduced_output = output if batch_size == 1 else output.sum(dim=0)
        value_total, value_compensation = kahan_add(
            value_total, value_compensation, reduced_output
        )
        explicit = (
            (
                execute_compiled_tn_reverse_dag(
                    dag,
                    reverse,
                    tape,
                    output_cotangent=output_cotangent,
                )
                if compiled_reverse
                else execute_explicit_tn_reverse_dag(
                    dag,
                    reverse,
                    tape,
                    output_cotangent=output_cotangent,
                )
            )
            if checkpoints is None
            else execute_checkpointed_tn_reverse_dag(
                dag,
                reverse,
                checkpoints,
                tape,
                output_cotangent=output_cotangent,
            )
        )
        differentiable_nodes = []
        node_cotangents = []
        for index, node in enumerate(subnodes):
            value_id = f"input:{index}"
            if node.tensor.requires_grad and value_id in explicit.input_cotangents:
                differentiable_nodes.append(node.tensor)
                node_cotangents.append(explicit.input_cotangents[value_id])
        if deferred_parameter_pullback:
            for node_index, node_cotangent in zip(
                (
                    index
                    for index, node in enumerate(subnodes)
                    if node.tensor.requires_grad
                    and f"input:{index}" in explicit.input_cotangents
                ),
                node_cotangents,
            ):
                source_node = source_nodes[node_index]
                source_cotangent = torch.zeros_like(source_node.tensor)
                source_index = tuple(
                    assignments.get(label, slice(None)) for label in source_node.labels
                )
                source_cotangent[source_index] = node_cotangent
                (
                    deferred_node_totals[node_index],
                    deferred_node_compensations[node_index],
                ) = kahan_add(
                    deferred_node_totals[node_index],
                    deferred_node_compensations[node_index],
                    source_cotangent,
                )
        else:
            slice_gradients = (
                torch.autograd.grad(
                    tuple(differentiable_nodes),
                    parameters,
                    grad_outputs=tuple(node_cotangents),
                    allow_unused=True,
                    retain_graph=True,
                )
                if differentiable_nodes and parameters
                else (None,) * len(parameters)
            )
            for index, gradient in enumerate(slice_gradients):
                if gradient is None:
                    continue
                gradient_totals[index], gradient_compensations[index] = kahan_add(
                    gradient_totals[index],
                    gradient_compensations[index],
                    gradient,
                )
        forward_operations += len(dag.operations)
        reverse_operations += len(explicit.executed_reverse_ids)
        nonfinite_cotangents += explicit.nonfinite_cotangent_count
        saved_tape_bytes = max(saved_tape_bytes, explicit.saved_tape_bytes)
        rematerialized_operations += explicit.rematerialized_operation_count

    if value_total is None:
        raise ValueError("sliced TN reverse did not execute any slices")
    if deferred_parameter_pullback:
        deferred_nodes = tuple(
            node.tensor
            for index, node in enumerate(source_nodes)
            if node.tensor.requires_grad and deferred_node_totals[index] is not None
        )
        deferred_cotangents = tuple(
            cotangent
            for index, cotangent in enumerate(deferred_node_totals)
            if source_nodes[index].tensor.requires_grad and cotangent is not None
        )
        gradient_totals = list(
            torch.autograd.grad(
                deferred_nodes,
                parameters,
                grad_outputs=deferred_cotangents,
                allow_unused=True,
                retain_graph=True,
            )
            if deferred_nodes and parameters
            else (None,) * len(parameters)
        )
    nonfinite_parameters = sum(
        int(not bool(torch.isfinite(gradient).all()))
        for gradient in gradient_totals
        if gradient is not None
    )
    return SlicedTNExplicitReverseResult(
        value=value_total,
        parameter_gradients=tuple(gradient_totals),
        slice_count=len(assignment_records),
        forward_operation_count=forward_operations,
        reverse_operation_count=reverse_operations,
        nonfinite_cotangent_count=nonfinite_cotangents,
        nonfinite_parameter_gradient_count=nonfinite_parameters,
        saved_tape_bytes=saved_tape_bytes,
        rematerialized_operation_count=rematerialized_operations,
    )
