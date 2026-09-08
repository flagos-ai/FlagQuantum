"""Ownership-enforcing execution for a distributed TN contraction DAG.

This path transfers or shards intermediate tensors according to a planned DAG.
It does not schedule complete independent slices or perform the differentiable
slice-output reduction owned by :mod:`execution`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import torch
import torch.distributed as dist

from ....compute import resolve_platform_device
from ....simulation.tensor_network.stages import einsum_pair_by_labels
from ...distributed.flagos_runtime import current_flagos_device
from .distributed_dag import (
    DistributedTNContractionDAG,
    DistributedTNContractionRecord,
    DistributedTNShard,
    DistributedTNValueLayout,
    shard_distributed_tn_value_layout,
)
from .redistribution import (
    execute_distributed_tn_redistribution,
    plan_distributed_tn_redistribution,
)
from .sharded_kernels import (
    DistributedTNShardedPairResult,
    execute_pre_sharded_pair_contraction,
    partition_tn_tensor_for_shard,
    select_sharded_pair_mode,
)


@dataclass(frozen=True)
class DistributedTNExecutionResult:
    """Rank-local result and evidence for one ownership DAG execution."""

    value: torch.Tensor
    dag_identity: str
    rank: int
    world_size: int
    executed_operation_ids: tuple[str, ...]
    sent_value_ids: tuple[str, ...]
    received_value_ids: tuple[str, ...]
    sent_bytes: int
    received_bytes: int
    peak_live_tensor_bytes: int
    released_value_count: int
    full_state_materialization: bool = False
    silent_statevector_fallback: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "executor": "distributed_tn_ownership_dag_v1",
            "dag_identity": self.dag_identity,
            "rank": self.rank,
            "world_size": self.world_size,
            "executed_operation_ids": self.executed_operation_ids,
            "sent_value_ids": self.sent_value_ids,
            "received_value_ids": self.received_value_ids,
            "sent_bytes": self.sent_bytes,
            "received_bytes": self.received_bytes,
            "peak_live_tensor_bytes": self.peak_live_tensor_bytes,
            "released_value_count": self.released_value_count,
            "output_shape": tuple(int(dim) for dim in self.value.shape),
            "output_bytes": _tensor_nbytes(self.value),
            "full_state_materialization": self.full_state_materialization,
            "silent_statevector_fallback": self.silent_statevector_fallback,
            "distribution_semantics": (
                "owned_intermediate_contraction"
                if self.world_size > 1
                else "single_device_fast_path"
            ),
            "scalability_claim_allowed": False,
            "scalability_blockers": (
                "large_intermediate_rank_group_sharding_pending",
                "accelerator_capacity_evidence_pending",
            ),
        }


@dataclass(frozen=True)
class DistributedTNShardedDAGExecutionResult:
    """Rank-local evidence from a multi-operation sharded DAG execution."""

    value: torch.Tensor
    dag_identity: str
    rank: int
    world_size: int
    executed_operation_ids: tuple[str, ...]
    replicated_operation_ids: tuple[str, ...]
    sharded_value_ids: tuple[str, ...]
    collective_count: int
    collective_bytes: int
    redistribution_count: int
    redistribution_sent_bytes: int
    redistribution_received_bytes: int
    redistributed_value_ids: tuple[str, ...]
    peak_live_tensor_bytes: int
    released_value_count: int
    full_input_materialized: bool = False
    full_state_materialization: bool = False
    silent_statevector_fallback: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "executor": "distributed_tn_sharded_dag_v1",
            "dag_identity": self.dag_identity,
            "rank": self.rank,
            "world_size": self.world_size,
            "executed_operation_ids": self.executed_operation_ids,
            "replicated_operation_ids": self.replicated_operation_ids,
            "sharded_value_ids": self.sharded_value_ids,
            "collective_count": self.collective_count,
            "collective_bytes": self.collective_bytes,
            "redistribution_count": self.redistribution_count,
            "redistribution_sent_bytes": self.redistribution_sent_bytes,
            "redistribution_received_bytes": self.redistribution_received_bytes,
            "redistributed_value_ids": self.redistributed_value_ids,
            "peak_live_tensor_bytes": self.peak_live_tensor_bytes,
            "released_value_count": self.released_value_count,
            "output_shape": tuple(int(dim) for dim in self.value.shape),
            "output_bytes": _tensor_nbytes(self.value),
            "full_input_materialized": self.full_input_materialized,
            "full_state_materialization": self.full_state_materialization,
            "silent_statevector_fallback": self.silent_statevector_fallback,
            "distribution_semantics": "multi_operation_sharded_dag",
            "scalability_claim_allowed": False,
            "scalability_blockers": ("end_to_end_capacity_evidence_pending",),
        }


def execute_sharded_tn_contraction_dag(
    dag: DistributedTNContractionDAG,
    local_inputs: Mapping[str, torch.Tensor],
    *,
    rank: int | None = None,
    process_group: Any | None = None,
    max_output_bytes: int = 4096,
) -> DistributedTNShardedDAGExecutionResult:
    """Execute a strict retained-shard chain ending in a reduced small output.

    Version 1 accepts replicated raw inputs and sharded intermediates.  Every
    operation must participate in the shard chain.  A retained shard label may
    flow through multiple operations; when it disappears, the rank-local
    partials are summed.  Compatible layout-axis changes are executed directly
    between rank-local shards.  Unique-owner raw inputs and changes to labels
    absent from a sharded value are rejected instead of materializing it whole.
    """

    dag.validate()
    if max_output_bytes <= 0:
        raise ValueError("distributed TN max_output_bytes must be positive")
    if not dist.is_initialized():
        raise RuntimeError("sharded TN DAG execution requires torch.distributed")
    actual_rank = int(dist.get_rank(group=process_group)) if rank is None else int(rank)
    actual_world_size = int(dist.get_world_size(group=process_group))
    if actual_world_size != dag.world_size:
        raise RuntimeError(
            "distributed TN DAG world size does not match the initialized process group"
        )
    _validate_rank(actual_rank, actual_world_size)
    values_by_id = {value.value_id: value for value in dag.values}
    raw_layouts = tuple(value for value in dag.values if value.producer_id is None)
    if any(value.semantics != "replicated_small" for value in raw_layouts):
        raise RuntimeError("sharded TN DAG v1 requires replicated-small raw inputs")
    expected = {value.value_id for value in raw_layouts}
    provided = {str(value_id) for value_id in local_inputs}
    if provided != expected:
        raise ValueError("sharded TN DAG requires exactly all replicated raw inputs")
    local_values: dict[str, torch.Tensor] = {}
    for value in raw_layouts:
        tensor = local_inputs[value.value_id]
        _validate_tensor(tensor, value)
        local_values[value.value_id] = tensor

    final_layout = values_by_id[dag.output_value_id]
    if final_layout.semantics == "sharded":
        raise RuntimeError("sharded TN DAG v1 requires a reduced final output")
    if final_layout.nbytes > int(max_output_bytes):
        raise RuntimeError("sharded TN DAG output exceeds the direct-observable limit")

    remaining_uses = _remaining_uses(dag)
    executed: list[str] = []
    replicated_operations: list[str] = []
    sharded_values: list[str] = []
    collective_count = 0
    collective_bytes = 0
    redistribution_count = 0
    redistribution_sent_bytes = 0
    redistribution_received_bytes = 0
    redistributed_value_ids: list[str] = []
    released = 0
    peak_live = _live_tensor_bytes(local_values)
    local_layouts = dict(values_by_id)

    for operation in dag.operations:
        input_layouts = tuple(
            local_layouts[value_id] for value_id in operation.input_value_ids
        )
        output_layout = values_by_id[operation.output_value_id]
        participates_in_shard_chain = output_layout.semantics == "sharded" or any(
            layout.semantics == "sharded" for layout in input_layouts
        )
        if not participates_in_shard_chain:
            operation_inputs = tuple(
                local_values.get(value_id) for value_id in operation.input_value_ids
            )
            if any(tensor is None for tensor in operation_inputs):
                raise RuntimeError("replicated TN DAG operation lost a local input")
            if any(layout.semantics != "replicated_small" for layout in input_layouts):
                raise RuntimeError(
                    "unsharded TN DAG operation requires locally replicated inputs"
                )
            left_id, right_id = operation.input_value_ids
            output = einsum_pair_by_labels(
                operation_inputs[0],
                values_by_id[left_id].labels,
                operation_inputs[1],
                values_by_id[right_id].labels,
                operation.output_labels,
            )
            _validate_tensor(output, output_layout)
            local_values[operation.output_value_id] = output
            local_layouts[operation.output_value_id] = replace(
                output_layout,
                semantics="replicated_small",
                owner_ranks=tuple(range(actual_world_size)),
            )
            executed.append(operation.operation_id)
            replicated_operations.append(operation.operation_id)
            peak_live = max(peak_live, _live_tensor_bytes(local_values))
            for value_id in operation.input_value_ids:
                remaining_uses[value_id] -= 1
                if remaining_uses[value_id] == 0 and value_id in local_values:
                    del local_values[value_id]
                    released += 1
            peak_live = max(peak_live, _live_tensor_bytes(local_values))
            continue

        mode, shard_label, shard = prepare_sharded_tn_dag_operation(
            dag, operation, rank=actual_rank
        )
        operation_inputs: dict[str, torch.Tensor] = {}
        for value_id in operation.input_value_ids:
            layout = local_layouts[value_id]
            tensor = local_values.get(value_id)
            if tensor is None:
                raise RuntimeError(
                    f"sharded TN rank {actual_rank} lost local value {value_id}"
                )
            if layout.semantics == "sharded":
                if layout.shard_label != shard_label:
                    if shard_label not in layout.labels:
                        raise RuntimeError(
                            "sharded TN DAG cannot redistribute a value along "
                            "an absent label"
                        )
                    destination = _reshard_layout(
                        layout,
                        shard_label=shard_label,
                        world_size=actual_world_size,
                    )
                    redistributed = execute_distributed_tn_redistribution(
                        tensor,
                        layout,
                        destination,
                        plan_distributed_tn_redistribution(layout, destination),
                        process_group=process_group,
                    )
                    tensor = redistributed.local_tensor
                    local_values[value_id] = tensor
                    local_layouts[value_id] = destination
                    layout = destination
                    redistribution_count += 1
                    redistribution_sent_bytes += redistributed.sent_bytes
                    redistribution_received_bytes += redistributed.received_bytes
                    redistributed_value_ids.append(value_id)
                expected_shard = next(
                    item for item in layout.shards if item.rank == actual_rank
                )
                if tuple(tensor.shape) != expected_shard.local_shape:
                    raise ValueError(
                        f"sharded TN value {value_id} local shape mismatch"
                    )
                operation_inputs[value_id] = tensor
            elif layout.semantics == "replicated_small":
                operation_inputs[value_id] = (
                    partition_tn_tensor_for_shard(
                        tensor,
                        layout.labels,
                        shard_label=shard_label,
                        shard=shard,
                    )
                    if shard_label in layout.labels
                    else tensor
                )
            else:
                raise RuntimeError(
                    "sharded TN DAG v1 forbids unique-owner operation inputs"
                )

        result = execute_sharded_tn_dag_operation(
            dag,
            operation,
            operation_inputs,
            rank=actual_rank,
            process_group=process_group,
        )
        if output_layout.semantics == "sharded":
            if tuple(result.value.shape) != shard.local_shape:
                raise RuntimeError(
                    "sharded TN DAG operation produced an invalid local shard"
                )
            sharded_values.append(operation.output_value_id)
            local_layouts[operation.output_value_id] = output_layout
        else:
            _validate_tensor(result.value, output_layout)
            local_layouts[operation.output_value_id] = replace(
                output_layout,
                semantics="replicated_small",
                owner_ranks=tuple(range(actual_world_size)),
            )
        local_values[operation.output_value_id] = result.value
        collective_count += result.collective_count
        collective_bytes += result.collective_bytes
        executed.append(operation.operation_id)
        peak_live = max(peak_live, _live_tensor_bytes(local_values))
        for value_id in operation.input_value_ids:
            remaining_uses[value_id] -= 1
            if remaining_uses[value_id] == 0 and value_id in local_values:
                del local_values[value_id]
                released += 1
        peak_live = max(peak_live, _live_tensor_bytes(local_values))

    final_value = local_values.get(dag.output_value_id)
    if final_value is None:
        raise RuntimeError("sharded TN DAG did not produce its final output")
    return DistributedTNShardedDAGExecutionResult(
        value=final_value,
        dag_identity=dag.identity,
        rank=actual_rank,
        world_size=actual_world_size,
        executed_operation_ids=tuple(executed),
        replicated_operation_ids=tuple(replicated_operations),
        sharded_value_ids=tuple(sharded_values),
        collective_count=collective_count,
        collective_bytes=collective_bytes,
        redistribution_count=redistribution_count,
        redistribution_sent_bytes=redistribution_sent_bytes,
        redistribution_received_bytes=redistribution_received_bytes,
        redistributed_value_ids=tuple(redistributed_value_ids),
        peak_live_tensor_bytes=peak_live,
        released_value_count=released,
    )


def _reshard_layout(
    layout: DistributedTNValueLayout,
    *,
    shard_label: int,
    world_size: int,
) -> DistributedTNValueLayout:
    """Build a compatible target layout without mutating the planned DAG."""

    unique = replace(
        layout,
        semantics="unique_owner",
        owner_ranks=(0,),
        shard_label=None,
        shard_axis=None,
        shards=(),
    )
    return shard_distributed_tn_value_layout(
        unique,
        world_size=world_size,
        shard_label=shard_label,
    )


def prepare_sharded_tn_dag_operation(
    dag: DistributedTNContractionDAG,
    operation: DistributedTNContractionRecord,
    *,
    rank: int,
) -> tuple[str, int, DistributedTNShard]:
    """Resolve the kernel mode, shard label, and this rank's local range."""

    mode, shard_label = select_sharded_pair_mode(dag, operation)
    values = {value.value_id: value for value in dag.values}
    output = values[operation.output_value_id]
    inputs = tuple(values[value_id] for value_id in operation.input_value_ids)
    layouts = (
        (output,)
        if mode == "retained_output_shard"
        else tuple(value for value in inputs if value.semantics == "sharded")
    )
    rank_shards = []
    for layout in layouts:
        shard = next((item for item in layout.shards if item.rank == int(rank)), None)
        if shard is None:
            raise ValueError(
                f"distributed TN rank {rank} has no shard for value {layout.value_id}"
            )
        rank_shards.append(shard)
    selected = rank_shards[0]
    if any(
        (shard.start, shard.stop) != (selected.start, selected.stop)
        for shard in rank_shards[1:]
    ):
        raise ValueError("distributed TN input shard ranges require redistribution")
    return mode, shard_label, selected


def execute_sharded_tn_dag_operation(
    dag: DistributedTNContractionDAG,
    operation: DistributedTNContractionRecord,
    local_inputs: Mapping[str, torch.Tensor],
    *,
    rank: int | None = None,
    process_group: Any | None = None,
) -> DistributedTNShardedPairResult:
    """Execute one DAG operation from already-localized input values."""

    actual_rank = (
        int(dist.get_rank(group=process_group))
        if rank is None and dist.is_initialized()
        else int(0 if rank is None else rank)
    )
    expected = set(operation.input_value_ids)
    provided = set(str(value_id) for value_id in local_inputs)
    if provided != expected:
        raise ValueError(
            "distributed TN sharded operation requires exactly its two local inputs"
        )
    mode, shard_label, shard = prepare_sharded_tn_dag_operation(
        dag, operation, rank=actual_rank
    )
    values = {value.value_id: value for value in dag.values}
    left_id, right_id = operation.input_value_ids
    return execute_pre_sharded_pair_contraction(
        local_inputs[left_id],
        values[left_id].labels,
        local_inputs[right_id],
        values[right_id].labels,
        operation.output_labels,
        mode=mode,
        shard_label=shard_label,
        shard=shard,
        rank=actual_rank,
        world_size=dag.world_size,
        process_group=process_group,
    )


def required_local_tn_inputs(
    dag: DistributedTNContractionDAG, *, rank: int
) -> tuple[str, ...]:
    """Return the input ids that a rank must materialize before execution."""

    dag.validate()
    _validate_rank(rank, dag.world_size)
    return tuple(
        value.value_id
        for value in dag.values
        if value.producer_id is None and int(rank) in value.owner_ranks
    )


def execute_distributed_tn_contraction_dag(
    dag: DistributedTNContractionDAG,
    local_inputs: Mapping[str, torch.Tensor],
    *,
    rank: int | None = None,
    process_group: Any | None = None,
    max_output_bytes: int = 4096,
) -> DistributedTNExecutionResult:
    """Execute a contraction DAG without accepting non-owned large inputs.

    Version 1 supports replicated-small and unique-owner values.  Rank-group
    sharded intermediates are rejected until the sharded contraction kernels
    exist.  The final value is broadcast only when it is no larger than
    ``max_output_bytes``; this keeps the path suitable for scalar observables
    and prevents accidental full-state materialization.
    """

    dag.validate()
    if max_output_bytes <= 0:
        raise ValueError("distributed TN max_output_bytes must be positive")
    actual_world_size = (
        int(dist.get_world_size(group=process_group)) if dist.is_initialized() else 1
    )
    actual_rank = (
        int(dist.get_rank(group=process_group))
        if rank is None and dist.is_initialized()
        else int(0 if rank is None else rank)
    )
    if actual_world_size != dag.world_size:
        raise RuntimeError(
            "distributed TN DAG world size does not match the initialized process group"
        )
    _validate_rank(actual_rank, dag.world_size)
    if dag.world_size > 1 and not dist.is_initialized():
        raise RuntimeError(
            "multi-rank distributed TN DAG execution requires torch.distributed"
        )
    sharded = tuple(
        value.value_id for value in dag.values if value.semantics == "sharded"
    )
    if sharded:
        raise RuntimeError(
            f"distributed TN sharded intermediates are not executable yet: {sharded}"
        )
    values_by_id = {value.value_id: value for value in dag.values}
    expected_inputs = set(required_local_tn_inputs(dag, rank=actual_rank))
    provided_inputs = set(str(value_id) for value_id in local_inputs)
    missing = tuple(sorted(expected_inputs - provided_inputs))
    unexpected = tuple(sorted(provided_inputs - expected_inputs))
    if missing:
        raise ValueError(f"distributed TN rank {actual_rank} missing inputs {missing}")
    if unexpected:
        raise ValueError(
            f"distributed TN rank {actual_rank} received non-owned inputs {unexpected}"
        )

    local_values: dict[str, torch.Tensor] = {}
    for value_id in expected_inputs:
        tensor = local_inputs[value_id]
        _validate_tensor(tensor, values_by_id[value_id])
        local_values[value_id] = tensor

    final_layout = values_by_id[dag.output_value_id]
    if final_layout.nbytes > int(max_output_bytes):
        raise RuntimeError(
            "distributed TN output exceeds the direct-observable limit; "
            "full-state output materialization is forbidden"
        )

    remaining_uses = _remaining_uses(dag)
    executed: list[str] = []
    sent: list[str] = []
    received: list[str] = []
    sent_bytes = 0
    received_bytes = 0
    released = 0
    peak_live = _live_tensor_bytes(local_values)

    for operation in dag.operations:
        owner = int(operation.owner_ranks[0])
        operation_inputs: list[torch.Tensor] = []
        for input_position, value_id in enumerate(operation.input_value_ids):
            layout = values_by_id[value_id]
            if actual_rank == owner:
                tensor = local_values.get(value_id)
                if tensor is None:
                    if owner in layout.owner_ranks:
                        raise RuntimeError(
                            f"distributed TN owner rank {owner} lost local value {value_id}"
                        )
                    source = _source_rank_for_value(layout, destination_rank=owner)
                    tensor = torch.empty(
                        layout.shape,
                        dtype=_torch_dtype(layout.dtype),
                        device=_communication_device(local_values),
                    )
                    dist.recv(
                        tensor,
                        src=source,
                        group=process_group,
                        tag=_communication_tag(operation.sequence, input_position),
                    )
                    local_values[value_id] = tensor
                    received.append(value_id)
                    received_bytes += layout.nbytes
                operation_inputs.append(tensor)
            else:
                if owner in layout.owner_ranks:
                    continue
                source = _source_rank_for_value(layout, destination_rank=owner)
                if actual_rank == source:
                    tensor = local_values.get(value_id)
                    if tensor is None:
                        raise RuntimeError(
                            f"distributed TN source rank {actual_rank} lost value {value_id}"
                        )
                    dist.send(
                        tensor,
                        dst=owner,
                        group=process_group,
                        tag=_communication_tag(operation.sequence, input_position),
                    )
                    sent.append(value_id)
                    sent_bytes += layout.nbytes

        if actual_rank == owner:
            left, right = operation_inputs
            output = einsum_pair_by_labels(
                left,
                values_by_id[operation.input_value_ids[0]].labels,
                right,
                values_by_id[operation.input_value_ids[1]].labels,
                operation.output_labels,
            )
            _validate_tensor(output, values_by_id[operation.output_value_id])
            local_values[operation.output_value_id] = output
            executed.append(operation.operation_id)
            peak_live = max(peak_live, _live_tensor_bytes(local_values))

        for value_id in operation.input_value_ids:
            remaining_uses[value_id] -= 1
            if remaining_uses[value_id] == 0 and value_id in local_values:
                del local_values[value_id]
                released += 1
        peak_live = max(peak_live, _live_tensor_bytes(local_values))

    output_owner = int(final_layout.owner_ranks[0])
    if actual_rank == output_owner:
        final_value = local_values.get(dag.output_value_id)
        if final_value is None:
            raise RuntimeError("distributed TN output owner did not produce the result")
    else:
        final_value = torch.empty(
            final_layout.shape,
            dtype=_torch_dtype(final_layout.dtype),
            device=_communication_device(local_values),
        )
    if dag.world_size > 1:
        dist.broadcast(final_value, src=output_owner, group=process_group)
    return DistributedTNExecutionResult(
        value=final_value,
        dag_identity=dag.identity,
        rank=actual_rank,
        world_size=dag.world_size,
        executed_operation_ids=tuple(executed),
        sent_value_ids=tuple(sent),
        received_value_ids=tuple(received),
        sent_bytes=int(sent_bytes),
        received_bytes=int(received_bytes),
        peak_live_tensor_bytes=int(peak_live),
        released_value_count=int(released),
    )


def _source_rank_for_value(
    layout: DistributedTNValueLayout, *, destination_rank: int
) -> int:
    if destination_rank in layout.owner_ranks:
        return int(destination_rank)
    if layout.semantics == "replicated_small":
        return int(layout.owner_ranks[0])
    return int(layout.owner_ranks[0])


def _remaining_uses(dag: DistributedTNContractionDAG) -> dict[str, int]:
    uses = {value.value_id: 0 for value in dag.values}
    for operation in dag.operations:
        for value_id in operation.input_value_ids:
            uses[value_id] += 1
    return uses


def _validate_rank(rank: int, world_size: int) -> None:
    if not 0 <= int(rank) < int(world_size):
        raise ValueError(
            f"distributed TN rank {rank} is outside world size {world_size}"
        )


def _validate_tensor(tensor: torch.Tensor, layout: DistributedTNValueLayout) -> None:
    if tuple(int(dim) for dim in tensor.shape) != layout.shape:
        raise ValueError(
            f"distributed TN value {layout.value_id} shape mismatch: "
            f"{tuple(tensor.shape)} != {layout.shape}"
        )
    if str(tensor.dtype) != layout.dtype:
        raise ValueError(
            f"distributed TN value {layout.value_id} dtype mismatch: "
            f"{tensor.dtype} != {layout.dtype}"
        )
    if _tensor_nbytes(tensor) != layout.nbytes:
        raise ValueError(f"distributed TN value {layout.value_id} byte size mismatch")


def _torch_dtype(name: str) -> torch.dtype:
    normalized = str(name).removeprefix("torch.")
    dtype = getattr(torch, normalized, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported distributed TN dtype {name!r}")
    return dtype


def _communication_device(local_values: Mapping[str, torch.Tensor]) -> torch.device:
    if local_values:
        return next(iter(local_values.values())).device
    if dist.is_initialized():
        backend = str(dist.get_backend()).strip().lower()
        if backend == "nccl":
            return resolve_platform_device("cuda")
        if backend == "flagos":
            return current_flagos_device()
    return torch.device("cpu")


def _communication_tag(sequence: int, input_position: int) -> int:
    return 6_000_000 + int(sequence) * 2 + int(input_position)


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _live_tensor_bytes(values: Mapping[str, torch.Tensor]) -> int:
    return sum(_tensor_nbytes(value) for value in values.values())


__all__ = (
    "DistributedTNExecutionResult",
    "DistributedTNShardedDAGExecutionResult",
    "execute_distributed_tn_contraction_dag",
    "execute_sharded_tn_contraction_dag",
    "execute_sharded_tn_dag_operation",
    "prepare_sharded_tn_dag_operation",
    "required_local_tn_inputs",
)
