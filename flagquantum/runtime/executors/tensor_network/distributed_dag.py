"""Versioned ownership records for distributed tensor-network contractions.

The planner in this module is intentionally planning-only.  It gives every
input, intermediate, operation, and cross-owner dependency a stable identity so
that later executors and reverse-mode planners can consume the same graph.  A
planned graph is never execution or scalability evidence by itself.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, Sequence

from ....simulation.tensor_network.models import (
    PairContractionStep,
    TensorNetworkContractionPlan,
)

TN_DAG_VERSION = "flagquantum.distributed_tn_dag.v1"

LayoutSemantics = Literal["replicated_small", "unique_owner", "sharded"]


def _tensor_nbytes(tensor: Any) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


@dataclass(frozen=True)
class DistributedTNShard:
    """One contiguous rank-local range of a logical TN value."""

    rank: int
    start: int
    stop: int
    local_shape: tuple[int, ...]
    nbytes: int

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise ValueError("distributed TN shard rank must be non-negative")
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("distributed TN shard range must be non-empty")
        if self.nbytes <= 0:
            raise ValueError("distributed TN shard bytes must be positive")


@dataclass(frozen=True)
class DistributedTNValueLayout:
    """Ownership and shape of one input or intermediate tensor-network value."""

    value_id: str
    producer_id: str | None
    labels: tuple[int, ...]
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    semantics: LayoutSemantics
    owner_ranks: tuple[int, ...]
    source_name: str = ""
    shard_label: int | None = None
    shard_axis: int | None = None
    shards: tuple[DistributedTNShard, ...] = ()

    def __post_init__(self) -> None:
        if not self.value_id:
            raise ValueError("distributed TN value_id must not be empty")
        if self.nbytes < 0:
            raise ValueError("distributed TN value bytes must be non-negative")
        if not self.owner_ranks:
            raise ValueError("distributed TN value requires at least one owner rank")
        if self.semantics == "unique_owner" and len(self.owner_ranks) != 1:
            raise ValueError("unique-owner TN values require exactly one owner rank")
        if self.semantics == "replicated_small" and len(self.owner_ranks) < 1:
            raise ValueError("replicated TN values require owner ranks")
        if self.semantics == "sharded":
            if len(self.owner_ranks) < 2:
                raise ValueError("sharded TN values require multiple owners")
            if self.shard_label is None or self.shard_axis is None or not self.shards:
                raise ValueError("sharded TN values require shard metadata")
            if self.shard_axis >= len(self.shape):
                raise ValueError("distributed TN shard axis is out of range")
            if tuple(shard.rank for shard in self.shards) != self.owner_ranks:
                raise ValueError("distributed TN shard ranks must match owners")
            cursor = 0
            for shard in self.shards:
                if shard.start != cursor:
                    raise ValueError("distributed TN shard ranges must be contiguous")
                local_shape = list(self.shape)
                local_shape[self.shard_axis] = shard.stop - shard.start
                if shard.local_shape != tuple(local_shape):
                    raise ValueError("distributed TN shard local shape is inconsistent")
                cursor = shard.stop
            if cursor != self.shape[self.shard_axis]:
                raise ValueError("distributed TN shards must cover the shard axis")
            if sum(shard.nbytes for shard in self.shards) != self.nbytes:
                raise ValueError("distributed TN shard bytes must cover the value")
        elif self.shard_label is not None or self.shard_axis is not None or self.shards:
            raise ValueError("non-sharded TN values cannot contain shard metadata")


@dataclass(frozen=True)
class DistributedTNCommunicationEdge:
    """A required value transfer across operation ownership boundaries."""

    value_id: str
    source_rank: int
    destination_rank: int
    consumer_operation_id: str
    estimated_bytes: int

    def __post_init__(self) -> None:
        if self.source_rank == self.destination_rank:
            raise ValueError("distributed TN communication edge must cross ranks")
        if self.estimated_bytes < 0:
            raise ValueError("distributed TN communication bytes must be non-negative")


@dataclass(frozen=True)
class DistributedTNContractionRecord:
    """One pair contraction with stable inputs, output, and execution ownership."""

    operation_id: str
    sequence: int
    input_value_ids: tuple[str, str]
    output_value_id: str
    owner_ranks: tuple[int, ...]
    output_labels: tuple[int, ...]
    output_shape: tuple[int, ...]
    estimated_cost: int
    intermediate_elements: int

    def __post_init__(self) -> None:
        if len(self.input_value_ids) != 2:
            raise ValueError("distributed TN pair contraction requires two inputs")
        if not self.owner_ranks:
            raise ValueError("distributed TN contraction requires an owner rank")
        if self.estimated_cost < 0 or self.intermediate_elements < 0:
            raise ValueError(
                "distributed TN contraction estimates must be non-negative"
            )


@dataclass(frozen=True)
class DistributedTNContractionDAG:
    """Deterministic, planning-only distributed contraction graph."""

    version: str
    identity: str
    world_size: int
    objective: str
    small_tensor_replication_bytes: int
    values: tuple[DistributedTNValueLayout, ...]
    operations: tuple[DistributedTNContractionRecord, ...]
    communication_edges: tuple[DistributedTNCommunicationEdge, ...]
    output_value_id: str
    planning_only: bool = True

    def validate(self) -> None:
        if self.version != TN_DAG_VERSION:
            raise ValueError(f"unsupported distributed TN DAG version {self.version!r}")
        if self.world_size <= 0:
            raise ValueError("distributed TN DAG world_size must be positive")
        if not self.planning_only:
            raise ValueError("distributed TN DAG v1 must remain planning-only")
        value_by_id = {value.value_id: value for value in self.values}
        if len(value_by_id) != len(self.values):
            raise ValueError("distributed TN DAG contains duplicate value ids")
        operation_ids = {operation.operation_id for operation in self.operations}
        if len(operation_ids) != len(self.operations):
            raise ValueError("distributed TN DAG contains duplicate operation ids")
        available = {
            value.value_id for value in self.values if value.producer_id is None
        }
        for sequence, operation in enumerate(self.operations):
            if operation.sequence != sequence:
                raise ValueError(
                    "distributed TN DAG operation sequence is not contiguous"
                )
            if any(value_id not in available for value_id in operation.input_value_ids):
                raise ValueError(
                    f"distributed TN operation {operation.operation_id} uses an unavailable input"
                )
            output = value_by_id.get(operation.output_value_id)
            if output is None or output.producer_id != operation.operation_id:
                raise ValueError(
                    f"distributed TN operation {operation.operation_id} has an invalid output"
                )
            available.add(operation.output_value_id)
        if self.output_value_id not in available:
            raise ValueError("distributed TN DAG output was not produced")
        for edge in self.communication_edges:
            if edge.value_id not in value_by_id:
                raise ValueError(
                    "distributed TN communication references an unknown value"
                )
            if edge.consumer_operation_id not in operation_ids:
                raise ValueError(
                    "distributed TN communication references an unknown operation"
                )
            for rank in (edge.source_rank, edge.destination_rank):
                if not 0 <= rank < self.world_size:
                    raise ValueError(
                        "distributed TN communication rank is out of range"
                    )
        if self.identity != _dag_identity(self._identity_payload()):
            raise ValueError("distributed TN DAG identity does not match its contents")

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "world_size": self.world_size,
            "objective": self.objective,
            "small_tensor_replication_bytes": self.small_tensor_replication_bytes,
            "values": tuple(asdict(value) for value in self.values),
            "operations": tuple(asdict(operation) for operation in self.operations),
            "communication_edges": tuple(
                asdict(edge) for edge in self.communication_edges
            ),
            "output_value_id": self.output_value_id,
            "planning_only": self.planning_only,
        }

    def summary(self) -> dict[str, Any]:
        self.validate()
        communication_bytes = sum(
            edge.estimated_bytes for edge in self.communication_edges
        )
        return {
            **self._identity_payload(),
            "identity": self.identity,
            "value_count": len(self.values),
            "operation_count": len(self.operations),
            "communication_edge_count": len(self.communication_edges),
            "estimated_communication_bytes": int(communication_bytes),
            "layout_semantics": tuple(
                sorted({value.semantics for value in self.values})
            ),
            "distribution_semantics": "planned_tensor_ownership",
            "scalability_claim_allowed": False,
        }


@dataclass
class _ActiveValue:
    value_id: str
    name: str
    labels: tuple[int, ...]
    layout: DistributedTNValueLayout


def _dag_identity(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _find_active(
    active: Sequence[_ActiveValue],
    *,
    name: str,
    labels: tuple[int, ...],
    excluded_index: int | None = None,
) -> int:
    for index, value in enumerate(active):
        if excluded_index is not None and index == excluded_index:
            continue
        if value.name == name and value.labels == labels:
            return index
    raise ValueError(
        f"distributed TN path references missing value name={name!r}, labels={labels}"
    )


def plan_distributed_tn_contraction_dag(
    plan: TensorNetworkContractionPlan,
    *,
    world_size: int,
    objective: str = "memory",
    small_tensor_replication_bytes: int = 4096,
    pair_steps: Sequence[PairContractionStep] | None = None,
) -> DistributedTNContractionDAG:
    """Build a deterministic ownership DAG from a native TN contraction plan.

    Version 1 assigns each contraction to one rank.  It deliberately does not
    claim intermediate sharding; the ``sharded`` layout is reserved for a later
    rank-group executor.
    """

    if world_size <= 0:
        raise ValueError("distributed TN DAG world_size must be positive")
    if small_tensor_replication_bytes < 0:
        raise ValueError("small tensor replication threshold must be non-negative")
    if objective not in {
        "balanced",
        "memory",
        "quality",
        "quality_multistart",
        "quality_reconfigured",
        "external_cotengra",
    }:
        raise ValueError(
            "distributed TN DAG objective must be 'balanced', 'memory', "
            "'quality', 'quality_multistart', 'quality_reconfigured', or "
            "'external_cotengra'"
        )
    if objective == "external_cotengra" and pair_steps is None:
        raise ValueError("external_cotengra objective requires explicit pair_steps")
    steps = (
        tuple(pair_steps)
        if pair_steps is not None
        else {
            "balanced": plan.greedy_path,
            "memory": plan.memory_greedy_path,
            "quality": plan.quality_greedy_path,
            "quality_multistart": plan.quality_multistart_path,
            "quality_reconfigured": plan.quality_reconfigured_path,
        }[objective]()
    )
    all_ranks = tuple(range(int(world_size)))
    values: list[DistributedTNValueLayout] = []
    active: list[_ActiveValue] = []
    for index, node in enumerate(plan.nodes):
        nbytes = _tensor_nbytes(node.tensor)
        replicated = nbytes <= int(small_tensor_replication_bytes)
        owner_ranks = all_ranks if replicated else (index % int(world_size),)
        layout = DistributedTNValueLayout(
            value_id=f"input:{index}",
            producer_id=None,
            labels=tuple(int(label) for label in node.labels),
            shape=tuple(int(dim) for dim in node.tensor.shape),
            dtype=str(node.tensor.dtype),
            nbytes=nbytes,
            semantics="replicated_small" if replicated else "unique_owner",
            owner_ranks=owner_ranks,
            source_name=str(node.name),
        )
        values.append(layout)
        active.append(
            _ActiveValue(
                value_id=layout.value_id,
                name=str(node.name),
                labels=layout.labels,
                layout=layout,
            )
        )

    operations: list[DistributedTNContractionRecord] = []
    edges: list[DistributedTNCommunicationEdge] = []
    for step in steps:
        left_index = _find_active(
            active, name=str(step.left), labels=tuple(step.left_labels)
        )
        left = active[left_index]
        right_index = _find_active(
            active,
            name=str(step.right),
            labels=tuple(step.right_labels),
            excluded_index=left_index,
        )
        right = active[right_index]
        owner = int(step.step) % int(world_size)
        operation_id = f"contract:{int(step.step)}"
        for input_value in (left, right):
            if owner not in input_value.layout.owner_ranks:
                source = int(input_value.layout.owner_ranks[0])
                edges.append(
                    DistributedTNCommunicationEdge(
                        value_id=input_value.value_id,
                        source_rank=source,
                        destination_rank=owner,
                        consumer_operation_id=operation_id,
                        estimated_bytes=input_value.layout.nbytes,
                    )
                )
        output_value_id = f"intermediate:{int(step.step)}"
        dtype = left.layout.dtype
        element_size = (
            max(1, left.layout.nbytes // max(1, _product(left.layout.shape)))
            if left.layout.shape
            else max(1, left.layout.nbytes)
        )
        output_nbytes = int(step.intermediate_size) * int(element_size)
        output = DistributedTNValueLayout(
            value_id=output_value_id,
            producer_id=operation_id,
            labels=tuple(int(label) for label in step.output_labels),
            shape=tuple(int(dim) for dim in step.output_shape),
            dtype=dtype,
            nbytes=output_nbytes,
            semantics="unique_owner",
            owner_ranks=(owner,),
            source_name=f"({left.name},{right.name})",
        )
        values.append(output)
        operations.append(
            DistributedTNContractionRecord(
                operation_id=operation_id,
                sequence=int(step.step),
                input_value_ids=(left.value_id, right.value_id),
                output_value_id=output_value_id,
                owner_ranks=(owner,),
                output_labels=output.labels,
                output_shape=output.shape,
                estimated_cost=int(step.estimated_cost),
                intermediate_elements=int(step.intermediate_size),
            )
        )
        for index in sorted((left_index, right_index), reverse=True):
            active.pop(index)
        active.append(
            _ActiveValue(
                value_id=output.value_id,
                name=output.source_name,
                labels=output.labels,
                layout=output,
            )
        )

    if len(active) != 1:
        raise ValueError("distributed TN DAG did not reduce to one output")
    payload = {
        "version": TN_DAG_VERSION,
        "world_size": int(world_size),
        "objective": str(objective),
        "small_tensor_replication_bytes": int(small_tensor_replication_bytes),
        "values": tuple(asdict(value) for value in values),
        "operations": tuple(asdict(operation) for operation in operations),
        "communication_edges": tuple(asdict(edge) for edge in edges),
        "output_value_id": active[0].value_id,
        "planning_only": True,
    }
    dag = DistributedTNContractionDAG(
        version=TN_DAG_VERSION,
        identity=_dag_identity(payload),
        world_size=int(world_size),
        objective=str(objective),
        small_tensor_replication_bytes=int(small_tensor_replication_bytes),
        values=tuple(values),
        operations=tuple(operations),
        communication_edges=tuple(edges),
        output_value_id=active[0].value_id,
        planning_only=True,
    )
    dag.validate()
    return dag


def _product(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return int(result)


def shard_distributed_tn_value_layout(
    layout: DistributedTNValueLayout,
    *,
    world_size: int,
    shard_label: int | None = None,
) -> DistributedTNValueLayout:
    """Replace a unique-owner layout with contiguous rank-group shards.

    This function plans layout only.  The execution path continues to reject
    ``sharded`` values until a compatible contraction kernel is selected.
    """

    if layout.semantics != "unique_owner":
        raise ValueError("only unique-owner TN values can be repartitioned")
    if world_size < 2:
        raise ValueError("distributed TN value sharding requires multiple ranks")
    if not layout.labels or not layout.shape:
        raise ValueError("scalar TN values cannot be sharded")
    if shard_label is None:
        axis = max(range(len(layout.shape)), key=lambda index: layout.shape[index])
        shard_label = int(layout.labels[axis])
    elif int(shard_label) not in layout.labels:
        raise ValueError("distributed TN shard label is not present in the value")
    else:
        axis = layout.labels.index(int(shard_label))
    extent = int(layout.shape[axis])
    shard_count = min(int(world_size), extent)
    if shard_count < 2:
        raise ValueError("distributed TN shard axis is too small")
    element_size = layout.nbytes // max(1, _product(layout.shape))
    base, remainder = divmod(extent, shard_count)
    shards = []
    start = 0
    for rank in range(shard_count):
        length = base + (1 if rank < remainder else 0)
        stop = start + length
        local_shape = list(layout.shape)
        local_shape[axis] = length
        shards.append(
            DistributedTNShard(
                rank=rank,
                start=start,
                stop=stop,
                local_shape=tuple(local_shape),
                nbytes=_product(local_shape) * element_size,
            )
        )
        start = stop
    return DistributedTNValueLayout(
        value_id=layout.value_id,
        producer_id=layout.producer_id,
        labels=layout.labels,
        shape=layout.shape,
        dtype=layout.dtype,
        nbytes=layout.nbytes,
        semantics="sharded",
        owner_ranks=tuple(shard.rank for shard in shards),
        source_name=layout.source_name,
        shard_label=int(shard_label),
        shard_axis=axis,
        shards=tuple(shards),
    )


def with_sharded_tn_intermediate(
    dag: DistributedTNContractionDAG,
    value_id: str,
    *,
    world_size: int | None = None,
    shard_label: int | None = None,
) -> DistributedTNContractionDAG:
    """Return a new identity-valid DAG with one produced value rank-sharded."""

    dag.validate()
    target_index = next(
        (index for index, value in enumerate(dag.values) if value.value_id == value_id),
        None,
    )
    if target_index is None:
        raise ValueError(f"distributed TN DAG has no value {value_id!r}")
    target = dag.values[target_index]
    if target.producer_id is None:
        raise ValueError("distributed TN input sharding requires an input partitioner")
    sharded = shard_distributed_tn_value_layout(
        target,
        world_size=int(dag.world_size if world_size is None else world_size),
        shard_label=shard_label,
    )
    if any(rank >= dag.world_size for rank in sharded.owner_ranks):
        raise ValueError("distributed TN shards exceed DAG world size")
    values = dag.values[:target_index] + (sharded,) + dag.values[target_index + 1 :]
    operations = tuple(
        (
            replace(operation, owner_ranks=sharded.owner_ranks)
            if operation.output_value_id == value_id
            else operation
        )
        for operation in dag.operations
    )
    candidate = replace(
        dag,
        identity="",
        values=values,
        operations=operations,
    )
    candidate = replace(
        candidate,
        identity=_dag_identity(candidate._identity_payload()),
    )
    candidate.validate()
    return candidate


def with_sharded_tn_input(
    dag: DistributedTNContractionDAG,
    value_id: str,
    *,
    world_size: int | None = None,
    shard_label: int | None = None,
) -> DistributedTNContractionDAG:
    """Return an identity-valid DAG with one original node rank-sharded."""

    dag.validate()
    target_index = next(
        (index for index, value in enumerate(dag.values) if value.value_id == value_id),
        None,
    )
    if target_index is None:
        raise ValueError(f"distributed TN DAG has no value {value_id!r}")
    target = dag.values[target_index]
    if target.producer_id is not None:
        raise ValueError("produced TN values require intermediate sharding")
    sharded = shard_distributed_tn_value_layout(
        target,
        world_size=int(dag.world_size if world_size is None else world_size),
        shard_label=shard_label,
    )
    if any(rank >= dag.world_size for rank in sharded.owner_ranks):
        raise ValueError("distributed TN input shards exceed DAG world size")
    values = dag.values[:target_index] + (sharded,) + dag.values[target_index + 1 :]
    candidate = replace(dag, identity="", values=values)
    candidate = replace(
        candidate,
        identity=_dag_identity(candidate._identity_payload()),
    )
    candidate.validate()
    return candidate


__all__ = (
    "DistributedTNCommunicationEdge",
    "DistributedTNContractionDAG",
    "DistributedTNContractionRecord",
    "DistributedTNShard",
    "DistributedTNValueLayout",
    "TN_DAG_VERSION",
    "plan_distributed_tn_contraction_dag",
    "shard_distributed_tn_value_layout",
    "with_sharded_tn_intermediate",
    "with_sharded_tn_input",
)
