"""Owner-sharded tensor-network training over rank-owned contraction slices.

The forward and reverse of one logical tensor-network workload run on the rank
that owns each slice, the value and the gradients are reduced across ranks, and
one owner rank consumes each gradient for the optimizer update. That is the
same division of labour the distributed sliced reverse already implements; what
this module adds is the training lifecycle around it -- a step loop, an
objective, and a checkpoint a restarted run can resume from.

The caller owns process-group initialization and cleanup, exactly as the
statevector and MPS training entries do, so failures reach the launcher instead
of being hidden behind a locally simulated rank.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from ....simulation.tensor_network.entrypoints import build_tensor_network_expectation
from ...distributed.context import resolve_local_world_size, resolve_node_count
from .distributed_optimizer import (
    execute_rank_owned_tn_sgd_step,
    plan_tn_parameter_owners,
)
from .distributed_sliced_reverse import execute_distributed_sliced_tn_explicit_reverse
from .sliced_tasks import plan_distributed_tn_slice_tasks
from .state import _claim_evidence_type, _communication_tiers

TN_TRAINING_CHECKPOINT_SCHEMA = "flagquantum.tn_training_checkpoint.v1"

#: The objective is a single Pauli-product expectation, so the reverse seeds the
#: scalar output with ones. A Hermitian observable has a real expectation value,
#: which makes that seed the derivative of the value the training run minimizes.
_TN_OBJECTIVE = "pauli_product_expectation_value"


@dataclass(frozen=True)
class ShardedTensorNetworkTrainingResult:
    """Owner-sharded tensor-network training evidence for one run.

    ``distribution_semantics`` describes the whole path: distinct slices are
    contracted on distinct ranks, the value and every gradient are reduced
    across them, and the optimizer update is consumed by one owner per
    parameter. ``scalability_claim_allowed`` stays false because a run like this
    is evidence about semantics, not about capacity.
    """

    losses: tuple[float, ...]
    completed_steps: int
    start_step: int
    world_size: int
    local_world_size: int
    node_count: int
    rank: int
    task_plan_identity: str
    slice_tasks: int
    slice_labels: tuple[int, ...]
    tasks_by_rank: Mapping[int, int]
    local_memory_bytes_by_rank: tuple[int, ...]
    owned_parameter_bytes_by_rank: tuple[int, ...]
    parameter_owner_ranks: tuple[int, ...]
    parameter_ownership: tuple[Mapping[str, Any], ...]
    step_records: tuple[Mapping[str, Any], ...]
    optimizer_name: str
    learning_rate: float
    objective: str
    checkpoint_files: tuple[str, ...]
    checkpoint_storage_semantics: str
    nonfinite_parameter_gradient_count: int
    communication_bytes: int
    scalability_blockers: tuple[str, ...]

    @property
    def final_loss(self) -> float:
        return self.losses[-1]

    def summary(self) -> dict[str, Any]:
        return {
            "state_mode": "distributed_tensor_network_training",
            "objective": self.objective,
            "losses": self.losses,
            "final_loss": self.final_loss,
            "completed_steps": self.completed_steps,
            "start_step": self.start_step,
            "rank": self.rank,
            "world_size": self.world_size,
            "local_world_size": self.local_world_size,
            "node_count": self.node_count,
            "rank_placement": {
                "rank": self.rank,
                "world_size": self.world_size,
                "local_world_size": self.local_world_size,
                "node_count": self.node_count,
                "node_rank": self.rank // self.local_world_size,
                "local_rank": self.rank % self.local_world_size,
            },
            "distribution_semantics": "sharded_across_ranks",
            "claim_evidence_type": _claim_evidence_type(
                backend=str(dist.get_backend()),
                initialized=dist.is_initialized(),
            ),
            "forward_distribution_semantics": "sharded_across_ranks",
            "backward_distribution_semantics": "sharded_across_ranks",
            "gradient_distribution_semantics": "sharded_across_ranks",
            "optimizer_update_semantics": "sharded_across_ranks",
            "task_plan_identity": self.task_plan_identity,
            "slice_tasks": self.slice_tasks,
            "slice_labels": self.slice_labels,
            "tasks_by_rank": dict(self.tasks_by_rank),
            "local_memory_bytes_by_rank": self.local_memory_bytes_by_rank,
            "owned_parameter_bytes_by_rank": self.owned_parameter_bytes_by_rank,
            "parameter_owner_ranks": self.parameter_owner_ranks,
            "parameter_ownership": tuple(
                dict(item) for item in self.parameter_ownership
            ),
            "step_records": tuple(dict(item) for item in self.step_records),
            "optimizer_name": self.optimizer_name,
            "learning_rate": self.learning_rate,
            "checkpoint_files": self.checkpoint_files,
            "checkpoint_storage_semantics": self.checkpoint_storage_semantics,
            "nonfinite_parameter_gradient_count": (
                self.nonfinite_parameter_gradient_count
            ),
            "communication_bytes": self.communication_bytes,
            "communication_tiers": _communication_tiers(
                reduction_tensor_bytes=self.communication_bytes,
                rank_placement={
                    "node_count": self.node_count,
                    "local_world_size": self.local_world_size,
                },
                model="collective_topology_dependent",
                collective="all_reduce_sum",
            ),
            "replicated_autograd": False,
            "full_state_materialized": False,
            "scalability_claim_allowed": False,
            "scalability_blockers": self.scalability_blockers,
        }


def _checkpoint_path(directory: Path, rank: int) -> Path:
    return directory / f"rank-{rank}-tn-training.pt"


def _checksum_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def _tensor_digest(parameters: Sequence[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for parameter in parameters:
        value = parameter.detach().to("cpu").contiguous()
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _write_checkpoint(
    directory: Path,
    *,
    rank: int,
    step: int,
    parameters: Sequence[torch.Tensor],
    learning_rate: float,
    task_plan_identity: str,
    world_size: int,
    local_world_size: int,
    node_count: int,
) -> str:
    """Write one rank's parameters, fully written before it becomes visible.

    Each rank holds the replicated parameter values the owner all-gather left
    behind, so a rank-local file is the whole resumed state. The checksum is
    written after the payload is in place, because a restart that trusted a
    payload the writer never finished would resume a trajectory that never ran.
    """

    directory.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(directory, rank)
    payload = {
        "schema_version": TN_TRAINING_CHECKPOINT_SCHEMA,
        "step": int(step),
        "rank": int(rank),
        "world_size": int(world_size),
        "local_world_size": int(local_world_size),
        "node_count": int(node_count),
        "learning_rate": float(learning_rate),
        "task_plan_identity": task_plan_identity,
        "parameters": {
            index: parameter.detach().to("cpu").clone()
            for index, parameter in enumerate(parameters)
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    _checksum_path(path).write_text(
        json.dumps(
            {
                "schema_version": TN_TRAINING_CHECKPOINT_SCHEMA,
                "step": int(step),
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "parameter_sha256": _tensor_digest(parameters),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return str(path)


def _read_checkpoint(
    directory: Path,
    *,
    rank: int,
    world_size: int,
    learning_rate: float,
    task_plan_identity: str,
) -> tuple[int, dict[int, torch.Tensor]]:
    """Return ``(next_step, parameters)`` from the rank's own checkpoint."""

    path = _checkpoint_path(directory, rank)
    if not path.is_file():
        raise RuntimeError(f"TN training resume found no checkpoint at {path}")
    checksum_path = _checksum_path(path)
    if not checksum_path.is_file():
        raise RuntimeError(
            f"TN training checkpoint integrity metadata is missing: {path}"
        )
    metadata = json.loads(checksum_path.read_text(encoding="utf-8"))
    if metadata.get("file_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise RuntimeError(f"TN training checkpoint failed its integrity check: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != TN_TRAINING_CHECKPOINT_SCHEMA:
        raise RuntimeError(f"TN training checkpoint schema is not supported: {path}")
    for key, expected in (
        ("rank", rank),
        ("world_size", world_size),
        ("local_world_size", int(resolve_local_world_size(world_size))),
        ("task_plan_identity", task_plan_identity),
    ):
        if payload.get(key) != expected:
            raise RuntimeError(
                f"TN training checkpoint {key} does not match the resuming run"
            )
    if float(payload.get("learning_rate", -1.0)) != learning_rate:
        raise RuntimeError(
            "TN training checkpoint learning rate does not match the resuming run"
        )
    restored = payload.get("parameters")
    if not isinstance(restored, dict) or not restored:
        raise RuntimeError("TN training checkpoint carries no parameters")
    return int(payload["step"]), {
        int(index): value for index, value in restored.items()
    }


def _observable_axes(
    observable: Mapping[int, str],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    x: list[int] = []
    y: list[int] = []
    z: list[int] = []
    for wire, axis in sorted(observable.items()):
        normalized = str(axis).upper()
        if normalized == "X":
            x.append(int(wire))
        elif normalized == "Y":
            y.append(int(wire))
        elif normalized == "Z":
            z.append(int(wire))
        else:
            raise ValueError(f"unsupported observable axis {axis!r} on wire {wire}")
    return tuple(x), tuple(y), tuple(z)


def _default_sliced_labels(expectation: Any, *, slice_count: int) -> tuple[int, ...]:
    """Pick the internal labels that split the contraction into live slices.

    An internal label that appears in only one node cannot be sliced at all. A
    label carried by a state-copy node -- a node whose tensor has fewer than two
    indexed dimensions, such as the `[1, 0]` a fresh qubit contributes -- is a
    single index that is already determined, so every branch of that slice but
    one contracts to an exact zero and the rank owning it would add nothing to
    the reduction. Slicing those would report a partitioned workload while one
    rank did all of the arithmetic, which is why the candidates are the
    remaining contracted labels.
    """

    if slice_count < 1:
        raise ValueError("slice_count must be positive")
    counts = Counter(label for node in expectation.nodes for label in node.labels)
    copy_labels: set[int] = set()
    for node in expectation.nodes:
        if sum(dim > 1 for dim in node.tensor.shape) < 2:
            copy_labels.update(node.labels)
    candidates = [
        label
        for label, count in counts.items()
        if count >= 2
        and label not in expectation.output_labels
        and label not in copy_labels
    ]
    if len(candidates) < slice_count:
        raise ValueError(
            "the tensor network has fewer slicable internal labels than slice_count"
        )
    return tuple(candidates[:slice_count])


def train_distributed_tensor_network(
    circuit_or_ir: Any,
    parameters: Sequence[torch.Tensor],
    *,
    steps: int,
    observable: Mapping[int, str],
    learning_rate: float = 0.05,
    sliced_labels: Sequence[int] | None = None,
    slice_count: int = 2,
    slice_batch_size: int = 1,
    gradient_reduction: str = "owner_reduce",
    checkpoint_directory: str | Path | None = None,
    checkpoint_interval: int = 1,
    resume: bool = False,
    process_group: Any | None = None,
) -> ShardedTensorNetworkTrainingResult:
    """Train one rank-sliced tensor network with owner-sharded SGD.

    ``parameters`` are the tensors the caller bound into the circuit; each rank
    contracts only the slices it owns, and one owner rank consumes each reduced
    gradient. The caller owns process-group initialization and cleanup.

    Returns the run's losses together with the placement, ownership, memory,
    communication and checkpoint evidence the run actually produced.
    """

    if not dist.is_initialized():
        raise RuntimeError(
            "distributed tensor-network training requires a process group; the "
            "caller owns initialization so a failure reaches the launcher"
        )
    step_count = int(steps)
    if step_count < 1:
        raise ValueError("steps must be positive")
    learning_rate = float(learning_rate)
    if not torch.isfinite(torch.tensor(learning_rate)) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    parameters = tuple(parameters)
    if not parameters:
        raise ValueError("distributed tensor-network training requires parameters")
    if any(not parameter.requires_grad for parameter in parameters):
        raise ValueError(
            "every tensor-network training parameter must require gradients"
        )
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")
    if checkpoint_directory is None and resume:
        raise ValueError("resume requires a checkpoint_directory")

    world_size = int(dist.get_world_size(group=process_group))
    rank = int(dist.get_rank(group=process_group))
    local_world_size = resolve_local_world_size(world_size)
    node_count = resolve_node_count(world_size, local_world_size)

    x, y, z = _observable_axes(observable)
    expectation = build_tensor_network_expectation(circuit_or_ir, x=x, y=y, z=z)
    labels = (
        tuple(int(label) for label in sliced_labels)
        if sliced_labels is not None
        else _default_sliced_labels(expectation, slice_count=slice_count)
    )
    slicing = expectation.slicing_plan(sliced_labels=labels)
    tasks = plan_distributed_tn_slice_tasks(
        slicing,
        world_size=world_size,
        local_world_size=local_world_size,
    )
    tasks_by_rank = {
        owner: sum(task.owner_rank == owner for task in tasks.tasks)
        for owner in range(world_size)
    }
    if any(count == 0 for count in tasks_by_rank.values()):
        raise RuntimeError(
            "every rank must own at least one tensor-network slice, got "
            f"{tasks_by_rank}; increase slice_count or reduce world_size"
        )
    owners = plan_tn_parameter_owners(len(parameters), world_size)
    ownership = tuple(
        {
            "rank": owner,
            "parameter_indices": tuple(
                index for index, assigned in enumerate(owners) if assigned == owner
            ),
            "ownership": "rank_owned_optimizer_update",
        }
        for owner in range(world_size)
    )

    directory = None if checkpoint_directory is None else Path(checkpoint_directory)
    start_step = 0
    losses: list[float] = []
    if directory is not None and resume:
        start_step, restored = _read_checkpoint(
            directory,
            rank=rank,
            world_size=world_size,
            learning_rate=learning_rate,
            task_plan_identity=tasks.identity,
        )
        for index, parameter in enumerate(parameters):
            if index not in restored:
                raise RuntimeError(
                    f"TN training checkpoint is missing parameter {index}"
                )
            parameter.detach().copy_(restored[index].to(parameter.device))

    step_records: list[Mapping[str, Any]] = []
    checkpoint_files: list[str] = []
    communication_bytes = 0
    nonfinite_gradients = 0
    peak_local_tape_bytes = 0
    for step in range(start_step, step_count):
        # A contraction plan materializes its node tensors, so parameter values
        # are baked in when it is built. Rebuilding it after every optimizer
        # update is what makes the next forward value describe the current
        # parameters; the label topology is parameter-independent, so the
        # slicing plan and the rank-owned task plan stay valid across steps.
        expectation = build_tensor_network_expectation(circuit_or_ir, x=x, y=y, z=z)
        reverse = execute_distributed_sliced_tn_explicit_reverse(
            expectation,
            slicing,
            tasks,
            parameters,
            process_group=process_group,
            gradient_reduction=gradient_reduction,
            slice_batch_size=slice_batch_size,
        )
        # The objective is the expectation value the reverse already produced;
        # no separate forward pass is taken, so the value and the gradients it
        # seeded describe the same contraction.
        loss = reverse.value.reshape(-1)[0].real
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"tensor-network loss became nonfinite at step {step}")
        optimizer = execute_rank_owned_tn_sgd_step(
            parameters,
            reverse.parameter_gradients,
            learning_rate=learning_rate,
            process_group=process_group,
        )
        losses.append(float(loss.detach()))
        nonfinite_gradients += reverse.nonfinite_parameter_gradient_count
        communication_bytes += reverse.collective_payload_bytes_per_rank
        peak_local_tape_bytes = max(
            peak_local_tape_bytes,
            reverse.local_saved_tape_bytes,
            reverse.local_peak_slice_tensor_bytes,
        )
        step_records.append(
            {
                "step": step,
                "loss": losses[-1],
                "reverse": reverse.summary(),
                "optimizer": optimizer.summary(),
                "local_task_count": reverse.local_task_count,
                "local_forward_operation_count": reverse.local_forward_operation_count,
                "local_reverse_operation_count": reverse.local_reverse_operation_count,
                "collective_count": reverse.collective_count,
                "collective_seconds": reverse.collective_seconds,
            }
        )
        if directory is not None and (step + 1) % checkpoint_interval == 0:
            checkpoint_files.append(
                _write_checkpoint(
                    directory,
                    rank=rank,
                    step=step + 1,
                    parameters=parameters,
                    learning_rate=learning_rate,
                    task_plan_identity=tasks.identity,
                    world_size=world_size,
                    local_world_size=local_world_size,
                    node_count=node_count,
                )
            )

    # Per-rank memory is gathered as a plain tensor collective: this is an
    # audit field, and the launched lanes run without NumPy, so an object
    # collective is not available to carry it. `local_memory_bytes_by_rank` is
    # the measured peak of the slice tensors this rank contracted, and the
    # owned-parameter bytes are the part of the optimizer state it owns.
    owned_parameter_bytes = sum(
        int(parameters[index].numel()) * int(parameters[index].element_size())
        for index, owner in enumerate(owners)
        if owner == rank
    )
    device = parameters[0].device
    local_memory = torch.tensor(
        [[peak_local_tape_bytes, owned_parameter_bytes]],
        dtype=torch.int64,
        device=device,
    )
    gathered_memory = [torch.empty_like(local_memory) for _ in range(world_size)]
    dist.all_gather(gathered_memory, local_memory, group=process_group)
    gathered_rows = tuple(row for value in gathered_memory for row in value.tolist())
    local_memory_bytes_by_rank = tuple(int(row[0]) for row in gathered_rows)
    owned_parameter_bytes_by_rank = tuple(int(row[1]) for row in gathered_rows)

    return ShardedTensorNetworkTrainingResult(
        losses=tuple(losses),
        # `steps` is the total target step count, so a resumed run reports the
        # total it reached while `losses` holds only the steps it ran itself.
        completed_steps=step_count,
        start_step=start_step,
        world_size=world_size,
        local_world_size=local_world_size,
        node_count=node_count,
        rank=rank,
        task_plan_identity=tasks.identity,
        slice_tasks=len(tasks.tasks),
        slice_labels=tuple(slicing.sliced_labels),
        tasks_by_rank=tasks_by_rank,
        local_memory_bytes_by_rank=local_memory_bytes_by_rank,
        owned_parameter_bytes_by_rank=owned_parameter_bytes_by_rank,
        parameter_owner_ranks=owners,
        parameter_ownership=ownership,
        step_records=tuple(step_records),
        optimizer_name="rank_owned_sgd",
        learning_rate=learning_rate,
        objective=_TN_OBJECTIVE,
        checkpoint_files=tuple(checkpoint_files),
        checkpoint_storage_semantics=(
            "rank_local_shared_filesystem" if directory is not None else "not_requested"
        ),
        nonfinite_parameter_gradient_count=nonfinite_gradients,
        communication_bytes=communication_bytes,
        scalability_blockers=(
            "accelerator_capacity_evidence_pending",
            "production_benchmark_audit_pending",
        ),
    )


__all__ = ["ShardedTensorNetworkTrainingResult", "train_distributed_tensor_network"]
