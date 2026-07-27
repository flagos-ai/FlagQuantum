"""Distributed context, plans, and rank-owned state models.

This module provides FlagQuantum-native distributed result objects without
depending on an external graph or tensor-network package. Development backends
must preserve the same rank ownership and communication semantics as production
backends, while production backends are expected to execute one logical workload
across rank-local shards instead of replicating the full circuit per rank.
"""

from __future__ import annotations

import datetime
import os
import socket
from dataclasses import dataclass, replace
from itertools import product
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ...simulation.mps import (
    MPSAdaptiveBondPlan,
    MPSConfig,
    MPSState,
)
from ...simulation.tensor import (
    TensorNetworkState,
)
from ..backends.mps.operations import (
    apply_one_mps_tensor as _apply_one_mps_tensor,
)
from ..backends.mps.operations import (
    apply_two_mps_tensors as _apply_two_mps_tensors,
)
from ..backends.mps.operations import (
    tensor_nbytes as _tensor_nbytes,
)
from .backend_policy import (
    DistributedBackendPolicy,
    resolve_distributed_backend_policy,
)

_PARAM_ALIASES = {
    "rx": ("theta",),
    "ry": ("theta",),
    "rz": ("theta",),
    "phase": ("theta",),
    "p": ("theta",),
    "u1": ("theta",),
    "u2": ("phi", "lbd"),
    "u3": ("theta", "phi", "lbd"),
}


@dataclass(frozen=True)
class TorchDistributedContext:
    """torch.distributed execution context used by FlagQuantum distributed modes."""

    rank: int
    world_size: int
    local_rank: int
    backend: str
    device: torch.device
    initialized: bool
    initialized_by_flagquantum: bool = False
    node_rank: int = 0
    local_world_size: int = 1
    node_count: int = 1
    hostname: str = ""


@dataclass(frozen=True)
class DistributedShardPlan:
    """A contiguous MPS wire shard assigned to one distributed rank."""

    rank: int
    world_size: int
    wires: tuple[int, ...]
    left_boundary: int | None
    right_boundary: int | None


@dataclass(frozen=True)
class DistributedSliceTask:
    """A tensor-network slice task assigned to one distributed rank."""

    rank: int
    world_size: int
    task_index: int
    assignments: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class DistributedBoundarySync:
    """A nearest-neighbor MPS gate crossing two adjacent rank shards."""

    left_wire: int
    right_wire: int
    left_rank: int
    right_rank: int
    owner_rank: int


@dataclass(frozen=True)
class DistributedBoundaryProtocol:
    """Communication protocol for a cross-shard nearest-neighbor MPS update."""

    sync: DistributedBoundarySync
    stages: tuple[str, ...]
    tensor_bytes: int
    recipient_ranks: tuple[int, ...] = ()
    transport: str = "async_p2p"
    estimated_transfer_bytes: int = 0
    point_to_point_messages: int = 0
    collective_messages: int = 0


def torch_distributed_is_available() -> bool:
    """Return whether torch.distributed can be used in this runtime."""

    return bool(dist.is_available())


def _infer_backend(
    device: torch.device | str | None = None, backend: str | None = None
) -> str:
    if backend is not None:
        return backend
    device_type = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    ).type
    return "nccl" if device_type == "cuda" and torch.cuda.is_available() else "gloo"


def _resolve_local_world_size(world_size: int) -> int:
    raw = os.environ.get("LOCAL_WORLD_SIZE") or os.environ.get("NPROC_PER_NODE")
    if raw is None:
        return max(1, int(world_size))
    return max(1, min(int(world_size), int(raw)))


def _resolve_node_rank(rank: int, local_world_size: int) -> int:
    raw = os.environ.get("GROUP_RANK") or os.environ.get("NODE_RANK")
    if raw is not None:
        return max(0, int(raw))
    return int(rank) // max(1, int(local_world_size))


def _node_count(world_size: int, local_world_size: int) -> int:
    return max(
        1,
        (max(1, int(world_size)) + max(1, int(local_world_size)) - 1)
        // max(1, int(local_world_size)),
    )


def _rank_node(rank: int, local_world_size: int) -> int:
    return int(rank) // max(1, int(local_world_size))


def _communication_tier(src_rank: int, dst_rank: int, *, local_world_size: int) -> str:
    return (
        "intra_node"
        if _rank_node(src_rank, local_world_size)
        == _rank_node(dst_rank, local_world_size)
        else "inter_node"
    )


def _rank_placement_summary(
    context: TorchDistributedContext | None, *, world_size: int | None = None
) -> dict[str, Any]:
    resolved_world_size = int(
        world_size if world_size is not None else (context.world_size if context else 1)
    )
    local_world_size = context.local_world_size if context else resolved_world_size
    node_count = (
        context.node_count
        if context
        else _node_count(resolved_world_size, local_world_size)
    )
    return {
        "rank": context.rank if context else 0,
        "local_rank": context.local_rank if context else 0,
        "world_size": resolved_world_size,
        "local_world_size": local_world_size,
        "node_rank": context.node_rank if context else 0,
        "node_count": node_count,
        "hostname": context.hostname if context else "",
        "device": str(context.device) if context else "cpu",
    }


def _resolve_backend_policy(options: dict[str, Any]) -> DistributedBackendPolicy:
    backend_policy = options.pop("distributed_backend_policy", None)
    distributed_profile = options.pop("distributed_profile", None)
    jax_backend = options.pop("jax_backend", None)
    torch_backend = options.pop("torch_backend", None)
    if backend_policy is not None:
        policy = backend_policy
    else:
        policy = resolve_distributed_backend_policy(profile=distributed_profile)
    if jax_backend is None and torch_backend is None:
        return policy
    source = dict(policy.source)
    if jax_backend is not None:
        source["runtime_jax_backend"] = str(jax_backend)
    if torch_backend is not None:
        source["runtime_torch_backend"] = str(torch_backend)
    return replace(
        policy,
        jax_backend=str(jax_backend or policy.jax_backend),
        torch_backend=str(torch_backend or policy.torch_backend),
        source=source,
    )


def _should_use_torch_distributed(
    distributed_executor: str,
    policy: DistributedBackendPolicy,
    *,
    world_size: int,
) -> bool:
    if distributed_executor == "torch":
        return True
    if distributed_executor not in {"auto", None}:
        return False
    if dist.is_available() and dist.is_initialized():
        return True
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return (
        policy.profile == "production"
        and policy.torch_backend == "torch_distributed"
        and max(int(world_size), env_world_size) > 1
    )


def _boundary_communication_tiers(
    protocols: Sequence[DistributedBoundaryProtocol],
    *,
    context: TorchDistributedContext | None,
    world_size: int,
    sync_bytes: int,
    boundary_transfer_bytes: int,
) -> dict[str, Any]:
    local_world_size = context.local_world_size if context else max(1, int(world_size))
    intra_bytes = 0
    inter_bytes = 0
    intra_messages = 0
    inter_messages = 0
    for protocol in protocols:
        tier = _communication_tier(
            protocol.sync.left_rank,
            protocol.sync.right_rank,
            local_world_size=local_world_size,
        )
        if tier == "intra_node":
            intra_bytes += int(protocol.estimated_transfer_bytes)
            intra_messages += int(
                protocol.point_to_point_messages + protocol.collective_messages
            )
        else:
            inter_bytes += int(protocol.estimated_transfer_bytes)
            inter_messages += int(
                protocol.point_to_point_messages + protocol.collective_messages
            )
    return {
        "model": "rank_endpoint_attributed_for_boundary_syncs",
        "local_world_size": local_world_size,
        "node_count": (
            context.node_count if context else _node_count(world_size, local_world_size)
        ),
        "intra_node_boundary_bytes": intra_bytes,
        "inter_node_boundary_bytes": inter_bytes,
        "intra_node_boundary_messages": intra_messages,
        "inter_node_boundary_messages": inter_messages,
        "unclassified_sync_bytes": max(
            0, int(sync_bytes) - int(boundary_transfer_bytes)
        ),
    }


def init_torch_distributed(
    *,
    backend: str | None = None,
    init_method: str = "env://",
    rank: int | None = None,
    world_size: int | None = None,
    local_rank: int | None = None,
    device: torch.device | str | None = None,
    force_initialize: bool = False,
    timeout_seconds: float | None = None,
) -> TorchDistributedContext:
    """Initialize or attach to a torch.distributed process group."""

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build.")
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_rank = int(os.environ.get("RANK", "0"))
    env_local_rank = int(os.environ.get("LOCAL_RANK", env_rank))
    world_size = int(world_size if world_size is not None else env_world_size)
    rank = int(rank if rank is not None else env_rank)
    local_rank = int(local_rank if local_rank is not None else env_local_rank)
    local_world_size = _resolve_local_world_size(world_size)
    node_rank = _resolve_node_rank(rank, local_world_size)
    node_count = _node_count(world_size, local_world_size)
    hostname = socket.gethostname()
    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if resolved_device.type == "cuda":
        torch.cuda.set_device(local_rank)
        resolved_device = torch.device("cuda", local_rank)
    resolved_backend = _infer_backend(resolved_device, backend)

    initialized_by_flagquantum = False
    if (world_size > 1 or force_initialize) and not dist.is_initialized():
        dist.init_process_group(
            backend=resolved_backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=(
                None
                if timeout_seconds is None
                else datetime.timedelta(seconds=float(timeout_seconds))
            ),
        )
        initialized_by_flagquantum = True
    if dist.is_initialized():
        rank = int(dist.get_rank())
        world_size = int(dist.get_world_size())
        local_world_size = _resolve_local_world_size(world_size)
        node_rank = _resolve_node_rank(rank, local_world_size)
        node_count = _node_count(world_size, local_world_size)
    return TorchDistributedContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        backend=resolved_backend,
        device=resolved_device,
        initialized=dist.is_initialized(),
        initialized_by_flagquantum=initialized_by_flagquantum,
        node_rank=node_rank,
        local_world_size=local_world_size,
        node_count=node_count,
        hostname=hostname,
    )


def destroy_torch_distributed() -> None:
    """Destroy the active torch.distributed process group if initialized."""

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _split_contiguous(n_items: int, world_size: int) -> tuple[tuple[int, ...], ...]:
    world_size = max(1, int(world_size))
    n_items = int(n_items)
    out = []
    for rank in range(world_size):
        start = rank * n_items // world_size
        stop = (rank + 1) * n_items // world_size
        out.append(tuple(range(start, stop)))
    return tuple(out)


def _mps_shards(n_wires: int, world_size: int) -> tuple[DistributedShardPlan, ...]:
    shards = []
    for rank, wires in enumerate(_split_contiguous(n_wires, world_size)):
        left = wires[0] - 1 if wires and wires[0] > 0 else None
        right = wires[-1] if wires and wires[-1] < n_wires - 1 else None
        shards.append(
            DistributedShardPlan(
                rank=rank,
                world_size=int(world_size),
                wires=wires,
                left_boundary=left,
                right_boundary=right,
            )
        )
    return tuple(shards)


def _tensor_slice_tasks(
    sliced_labels: Sequence[int],
    slice_shape: Sequence[int],
    world_size: int,
) -> tuple[DistributedSliceTask, ...]:
    tasks = []
    labels = tuple(int(label) for label in sliced_labels)
    ranges = [range(int(size)) for size in slice_shape]
    for task_index, values in enumerate(product(*ranges) if ranges else [()]):
        rank = task_index % max(1, int(world_size))
        tasks.append(
            DistributedSliceTask(
                rank=rank,
                world_size=int(world_size),
                task_index=task_index,
                assignments=tuple(zip(labels, tuple(int(value) for value in values))),
            )
        )
    return tuple(tasks)


class DistributedMPSState:
    """Distributed MPS result facade with rank-shard metadata."""

    def __init__(
        self,
        local_state: MPSState,
        *,
        world_size: int,
        shards: Sequence[DistributedShardPlan] | None = None,
        context: TorchDistributedContext | None = None,
        local_shard_tensors: Mapping[int, torch.Tensor] | None = None,
        owned_instruction_count: int = 0,
        sharded_kernel_count: int = 0,
        tensor_sync_count: int = 0,
        boundary_sync_count: int = 0,
        full_sync_count: int = 0,
        sync_bytes: int = 0,
        boundary_transfer_bytes: int = 0,
        boundary_syncs: Sequence[DistributedBoundarySync] = (),
        boundary_protocols: Sequence[DistributedBoundaryProtocol] = (),
        sharded_state: ShardedMPSState | None = None,
        adaptive_initial_plan: MPSAdaptiveBondPlan | None = None,
        adaptive_final_plan: MPSAdaptiveBondPlan | None = None,
        adaptive_rerun: bool = False,
        autograd_state: MPSState | None = None,
        backend_policy: DistributedBackendPolicy | None = None,
        local_simulation: bool = False,
        development_sharded_states: Sequence[ShardedMPSState] = (),
        full_mps_reconstruction_count: int = 0,
        unsupported_instruction_count: int = 0,
        strict_sharded: bool = False,
        jax_distributed_plan: Mapping[str, Any] | None = None,
    ) -> None:
        self.local_state = local_state
        self.world_size = int(world_size)
        self.shards = (
            tuple(shards)
            if shards is not None
            else _mps_shards(local_state.n_wires, self.world_size)
        )
        self.context = context
        self.local_shard_tensors = dict(local_shard_tensors or {})
        self.owned_instruction_count = int(owned_instruction_count)
        self.sharded_kernel_count = int(sharded_kernel_count)
        self.tensor_sync_count = int(tensor_sync_count)
        self.boundary_sync_count = int(boundary_sync_count)
        self.full_sync_count = int(full_sync_count)
        self.sync_bytes = int(sync_bytes)
        self.boundary_transfer_bytes = int(boundary_transfer_bytes)
        self.boundary_syncs = tuple(boundary_syncs)
        self.boundary_protocols = tuple(boundary_protocols)
        self.sharded_state = sharded_state
        self.adaptive_initial_plan = adaptive_initial_plan
        self.adaptive_final_plan = adaptive_final_plan
        self.adaptive_rerun = bool(adaptive_rerun)
        self.autograd_state = autograd_state
        self.backend_policy = backend_policy
        self.local_simulation = bool(local_simulation)
        self.development_sharded_states = tuple(development_sharded_states)
        self.full_mps_reconstruction_count = int(full_mps_reconstruction_count)
        self.unsupported_instruction_count = int(unsupported_instruction_count)
        self.strict_sharded = bool(strict_sharded)
        self.jax_distributed_plan = (
            dict(jax_distributed_plan) if jax_distributed_plan is not None else None
        )

    @property
    def n_wires(self) -> int:
        return self.local_state.n_wires

    @property
    def bsz(self) -> int:
        return self.local_state.bsz

    def state(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.autograd_state is not None and not kwargs.get("refresh", False):
            return self.autograd_state.to_statevector(*args, **kwargs)
        return self.local_state.to_statevector(*args, **kwargs)

    to_statevector = state
    wavefunction = state

    def probabilities(self) -> torch.Tensor:
        if self.autograd_state is not None:
            return self.autograd_state.probabilities()
        return self.local_state.probabilities()

    def expectation_z(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.autograd_state is not None:
            return self.autograd_state.expectation_z(*args, **kwargs)
        return self.local_state.expectation_z(*args, **kwargs)

    def expectation_z_sum(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.autograd_state is not None:
            return self.autograd_state.expectation_z_sum(*args, **kwargs)
        return self.local_state.expectation_z_sum(*args, **kwargs)

    def expectation_ps(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.autograd_state is not None:
            return self.autograd_state.expectation_ps(*args, **kwargs)
        return self.local_state.expectation_ps(*args, **kwargs)

    def sample(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.local_state.sample(*args, **kwargs)

    def counts(self, *args: Any, **kwargs: Any) -> list[dict[str | int, int]]:
        return self.local_state.counts(*args, **kwargs)

    def summary(self) -> dict[str, Any]:
        summary = dict(self.local_state.summary())
        initialized = bool(self.context and self.context.initialized)
        has_development_sharded_forward = (
            self.local_simulation
            and self.world_size > 1
            and bool(self.development_sharded_states)
        )
        has_sharded_forward = (
            initialized
            and self.sharded_state is not None
            and bool(self.local_shard_tensors)
        ) or has_development_sharded_forward
        has_replicated_state_view = self.local_state is not None
        has_replicated_autograd = self.autograd_state is not None
        rank_placement = _rank_placement_summary(
            self.context, world_size=self.world_size
        )
        communication_tiers = _boundary_communication_tiers(
            self.boundary_protocols,
            context=self.context,
            world_size=self.world_size,
            sync_bytes=self.sync_bytes,
            boundary_transfer_bytes=self.boundary_transfer_bytes,
        )
        summary.update(
            {
                "state_mode": "distributed_mps",
                "claim_evidence_type": (
                    "production_runtime" if initialized else "development_smoke"
                ),
                "world_size": self.world_size,
                "local_world_size": rank_placement["local_world_size"],
                "node_count": rank_placement["node_count"],
                "executor": (
                    "torch_distributed"
                    if initialized
                    else (
                        "local_tensor_development_simulator"
                        if self.local_simulation
                        else "local"
                    )
                ),
                "distributed_backend_policy": (
                    self.backend_policy.summary() if self.backend_policy else None
                ),
                "distribution_semantics": (
                    "hybrid_sharded_forward_with_replicated_state"
                    if has_sharded_forward
                    and (has_replicated_state_view or has_replicated_autograd)
                    else (
                        "sharded_across_ranks"
                        if has_sharded_forward
                        else "replicated_single_rank"
                    )
                ),
                "scalability_claim_allowed": False,
                "scalability_blockers": tuple(
                    dict.fromkeys(
                        (
                            *(
                                blocker
                                for blocker, active in (
                                    (
                                        "single_process_development_simulator",
                                        self.local_simulation,
                                    ),
                                    (
                                        "full_local_mps_state_view",
                                        has_sharded_forward
                                        and has_replicated_state_view,
                                    ),
                                    (
                                        "full_mps_sync_fallback",
                                        self.full_sync_count > 0,
                                    ),
                                    (
                                        "unsupported_nonlocal_mps_gate",
                                        self.unsupported_instruction_count > 0,
                                    ),
                                    (
                                        "replicated_mps_autograd",
                                        has_replicated_autograd,
                                    ),
                                )
                                if active
                            ),
                            "mps_boundary_adjoint_exchange_pending",
                            "mps_optimizer_update_ownership_pending",
                        )
                    )
                ),
                "mps_execution": (
                    "site_sharded_sync"
                    if initialized
                    else (
                        "local_tensor_site_sharded_simulator"
                        if self.local_simulation
                        else "local"
                    )
                ),
                "rank": self.context.rank if self.context else 0,
                "rank_placement": rank_placement,
                "owned_instruction_count": self.owned_instruction_count,
                "sharded_kernel_count": self.sharded_kernel_count,
                "tensor_sync_count": self.tensor_sync_count,
                "boundary_sync_count": self.boundary_sync_count,
                "full_sync_count": self.full_sync_count,
                "sync_bytes": self.sync_bytes,
                "boundary_transfer_bytes": self.boundary_transfer_bytes,
                "communication_tiers": communication_tiers,
                "storage": (
                    "sharded_development"
                    if self.local_simulation
                    else "sharded" if self.sharded_state is not None else "replicated"
                ),
                "gradient_execution": (
                    "replicated_mps_autograd"
                    if self.autograd_state is not None
                    else "sharded_forward"
                ),
                "full_mps_reconstruction_count": self.full_mps_reconstruction_count,
                "unsupported_instruction_count": self.unsupported_instruction_count,
                "strict_sharded": self.strict_sharded,
                "jax_distributed_plan": self.jax_distributed_plan,
                "adaptive": self.adaptive_initial_plan is not None,
                "adaptive_rerun": self.adaptive_rerun,
                "adaptive_initial_plan": (
                    self.adaptive_initial_plan.summary()
                    if self.adaptive_initial_plan
                    else None
                ),
                "adaptive_final_plan": (
                    self.adaptive_final_plan.summary()
                    if self.adaptive_final_plan
                    else None
                ),
                "adaptive_refinement_plan": (
                    self.local_state.local_refinement_plan(
                        global_error_budget=(
                            self.adaptive_initial_plan.global_error_budget
                            if self.adaptive_initial_plan
                            else None
                        )
                    ).summary()
                    if self.adaptive_initial_plan
                    else None
                ),
                "local_tensor_wires": (
                    tuple(sorted(self.sharded_state.local_tensors))
                    if self.sharded_state
                    else tuple(sorted(self.local_shard_tensors))
                ),
                "local_tensor_wires_by_rank": (
                    {
                        state.rank: tuple(sorted(state.local_tensors))
                        for state in self.development_sharded_states
                    }
                    if self.development_sharded_states
                    else None
                ),
                "local_tensor_bytes_by_rank": (
                    {
                        state.rank: sum(
                            _tensor_nbytes(tensor)
                            for tensor in state.local_tensors.values()
                        )
                        for state in self.development_sharded_states
                    }
                    if self.development_sharded_states
                    else None
                ),
                "local_memory_bytes_by_rank": (
                    tuple(
                        sum(
                            _tensor_nbytes(tensor)
                            for tensor in state.local_tensors.values()
                        )
                        for state in sorted(
                            self.development_sharded_states, key=lambda item: item.rank
                        )
                    )
                    if self.development_sharded_states
                    else None
                ),
                "boundary_syncs": tuple(
                    {
                        "left_wire": item.left_wire,
                        "right_wire": item.right_wire,
                        "left_rank": item.left_rank,
                        "right_rank": item.right_rank,
                        "owner_rank": item.owner_rank,
                    }
                    for item in self.boundary_syncs
                ),
                "boundary_protocols": tuple(
                    {
                        "stages": item.stages,
                        "tensor_bytes": item.tensor_bytes,
                        "recipient_ranks": item.recipient_ranks,
                        "transport": item.transport,
                        "communication_tier": _communication_tier(
                            item.sync.left_rank,
                            item.sync.right_rank,
                            local_world_size=rank_placement["local_world_size"],
                        ),
                        "estimated_transfer_bytes": item.estimated_transfer_bytes,
                        "point_to_point_messages": item.point_to_point_messages,
                        "collective_messages": item.collective_messages,
                        "left_rank": item.sync.left_rank,
                        "right_rank": item.sync.right_rank,
                        "owner_rank": item.sync.owner_rank,
                    }
                    for item in self.boundary_protocols
                ),
                "rank_shards": tuple(
                    {
                        "rank": shard.rank,
                        "wires": shard.wires,
                        "left_boundary": shard.left_boundary,
                        "right_boundary": shard.right_boundary,
                    }
                    for shard in self.shards
                ),
            }
        )
        site_ownership = tuple(
            {
                "rank": int(shard.rank),
                "wires": tuple(int(wire) for wire in shard.wires),
                "ownership_semantics": "mps_site_range",
            }
            for shard in self.shards
        )
        bond_ownership = tuple(
            {
                "left_rank": int(shard.rank),
                "right_rank": int(shard.rank + 1),
                "left_wire": int(shard.right_boundary),
                "right_wire": int(shard.right_boundary + 1),
                "ownership_semantics": "adjacent_rank_boundary_bond",
            }
            for shard in self.shards
            if shard.right_boundary is not None
            and int(shard.rank + 1) < int(self.world_size)
        )
        fallback_semantics = tuple(
            item
            for item, active in (
                ("local_simulation", self.local_simulation),
                (
                    "full_local_mps_state_view",
                    has_sharded_forward and has_replicated_state_view,
                ),
                ("full_local_mps_replay", self.full_sync_count > 0),
                ("replicated_mps_autograd", has_replicated_autograd),
            )
            if active
        ) or ("none",)
        summary.update(
            {
                "mps_forward_distribution_semantics": (
                    "sharded_across_ranks"
                    if has_sharded_forward
                    else "replicated_single_rank"
                ),
                "mps_backward_distribution_semantics": (
                    "replicated_per_rank" if has_replicated_autograd else "incomplete"
                ),
                "site_shard_ownership": site_ownership,
                "bond_shard_ownership": bond_ownership,
                "parameter_gradient_ownership": (),
                "boundary_gradient_ownership": "unknown",
                "boundary_adjoint_exchange": "pending",
                "boundary_gradient_routes": (
                    bond_ownership if bond_ownership else "not_required"
                ),
                "canonicalization_backward_strategy": "pending",
                "truncation_gradient_metadata": "not_measured",
                "mps_backward_memory_plan": {
                    "local_memory_bytes_by_rank": summary.get(
                        "local_memory_bytes_by_rank"
                    ),
                    "communication_buffer_bytes": int(self.boundary_transfer_bytes),
                },
                "mps_backward_communication_plan": {
                    "estimated_transfer_bytes": int(self.boundary_transfer_bytes),
                    "boundary_adjoint_exchange": "pending",
                    "boundary_gradient_routes": (
                        bond_ownership if bond_ownership else "not_required"
                    ),
                },
                "optimizer_update_semantics": "not_measured",
                "optimizer_update_ownership": (),
                "fallback_semantics": fallback_semantics,
                "backward_execution": summary.get("gradient_execution", "incomplete"),
            }
        )
        from ..audit import attach_mps_runtime_summary

        return attach_mps_runtime_summary(summary)


class ShardedMPSState:
    """Rank-local MPS tensor storage with explicit distributed reconstruction."""

    def __init__(
        self,
        *,
        n_wires: int,
        bsz: int,
        config: MPSConfig,
        local_tensors: Mapping[int, torch.Tensor],
        shards: Sequence[DistributedShardPlan],
        context: TorchDistributedContext | None = None,
        rank: int | None = None,
    ) -> None:
        self.n_wires = int(n_wires)
        self.bsz = int(bsz)
        self.config = config
        self.local_tensors = dict(local_tensors)
        self.shards = tuple(shards)
        self.context = context
        self._rank = int(rank) if rank is not None else None

    @property
    def world_size(self) -> int:
        return (
            self.context.world_size
            if self.context
            else max((shard.world_size for shard in self.shards), default=1)
        )

    @property
    def rank(self) -> int:
        return self.context.rank if self.context else int(self._rank or 0)

    def gather_tensors(self) -> dict[int, torch.Tensor]:
        local_payload = {
            wire: tensor.detach().cpu() for wire, tensor in self.local_tensors.items()
        }
        if self.context is None or not self.context.initialized:
            return local_payload
        gathered: list[Any] = [None for _ in range(self.context.world_size)]
        dist.all_gather_object(gathered, local_payload)
        out: dict[int, torch.Tensor] = {}
        for payload in gathered:
            if payload:
                out.update(payload)
        return out

    def apply_one_local(self, matrix: torch.Tensor, wire: int) -> None:
        wire = int(wire)
        if wire not in self.local_tensors:
            raise ValueError(f"Wire {wire} is not owned by rank {self.rank}.")
        self.local_tensors[wire] = _apply_one_mps_tensor(
            self.local_tensors[wire], matrix
        )

    def apply_two_local(
        self, matrix: torch.Tensor, left_wire: int, *, reverse: bool = False
    ) -> float:
        left_wire = int(left_wire)
        right_wire = left_wire + 1
        if left_wire not in self.local_tensors or right_wire not in self.local_tensors:
            raise ValueError(
                f"Wires {left_wire} and {right_wire} are not both owned by rank {self.rank}."
            )
        left, right, step_error = _apply_two_mps_tensors(
            self.local_tensors[left_wire],
            self.local_tensors[right_wire],
            matrix,
            self.config,
            reverse=reverse,
        )
        self.local_tensors[left_wire] = left
        self.local_tensors[right_wire] = right
        return step_error

    def to_mps(self) -> MPSState:
        tensors_by_wire = self.gather_tensors()
        missing = [wire for wire in range(self.n_wires) if wire not in tensors_by_wire]
        if missing:
            raise RuntimeError(
                f"Cannot reconstruct sharded MPS; missing wires {missing}."
            )
        device = (
            self.context.device
            if self.context
            else next(iter(tensors_by_wire.values())).device
        )
        tensors = [
            tensors_by_wire[wire].to(device=device) for wire in range(self.n_wires)
        ]
        return MPSState(tensors, config=self.config)

    def to_statevector(self) -> torch.Tensor:
        return self.to_mps().to_statevector()

    state = to_statevector

    def summary(self) -> dict[str, Any]:
        return {
            "state_mode": "sharded_mps",
            "rank": self.rank,
            "world_size": self.world_size,
            "n_wires": self.n_wires,
            "batch_size": self.bsz,
            "local_tensor_wires": tuple(sorted(self.local_tensors)),
            "local_tensor_count": len(self.local_tensors),
            "local_tensor_bytes": sum(
                _tensor_nbytes(tensor) for tensor in self.local_tensors.values()
            ),
        }


class DistributedTensorNetworkState:
    """Distributed tensor-network result facade with slice-task metadata."""

    def __init__(
        self,
        local_state: TensorNetworkState,
        *,
        world_size: int,
        tasks: Sequence[DistributedSliceTask],
        context: TorchDistributedContext | None = None,
        state_cache: torch.Tensor | None = None,
        backend_policy: DistributedBackendPolicy | None = None,
        local_simulation: bool = False,
        rank_partial_bytes: Mapping[int, int] | None = None,
        jax_distributed_plan: Mapping[str, Any] | None = None,
    ) -> None:
        self.local_state = local_state
        self.world_size = int(world_size)
        self.tasks = tuple(tasks)
        self.context = context
        self._state_cache = state_cache
        self.plan = local_state.plan
        self.backend_policy = backend_policy
        self.local_simulation = bool(local_simulation)
        self.rank_partial_bytes = {
            int(rank): int(value) for rank, value in (rank_partial_bytes or {}).items()
        }
        self.jax_distributed_plan = (
            dict(jax_distributed_plan) if jax_distributed_plan is not None else None
        )

    @property
    def n_wires(self) -> int:
        return self.local_state.n_wires

    @property
    def bsz(self) -> int:
        return self.local_state.bsz

    def state(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self._state_cache is not None and not kwargs.get("refresh", False):
            return self._state_cache
        return self.local_state.state(*args, **kwargs)

    to_statevector = state
    wavefunction = state

    def probabilities(self) -> torch.Tensor:
        return self.local_state.probabilities()

    def expectation_z(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.local_state.expectation_z(*args, **kwargs)

    def expectation_ps(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.local_state.expectation_ps(*args, **kwargs)

    def sample(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.local_state.sample(*args, **kwargs)

    def counts(self, *args: Any, **kwargs: Any) -> list[dict[str | int, int]]:
        return self.local_state.counts(*args, **kwargs)

    def summary(self) -> dict[str, Any]:
        summary = dict(self.local_state.summary())
        initialized = bool(self.context and self.context.initialized)
        has_slice_parallel_state = (
            initialized or self.local_simulation
        ) and self._state_cache is not None
        rank_placement = _rank_placement_summary(
            self.context, world_size=self.world_size
        )
        reduction_tensor_bytes = (
            _tensor_nbytes(self._state_cache) if self._state_cache is not None else 0
        )
        summary.update(
            {
                "state_mode": "distributed_tensor_network",
                "world_size": self.world_size,
                "local_world_size": rank_placement["local_world_size"],
                "node_count": rank_placement["node_count"],
                "executor": (
                    "torch_distributed"
                    if initialized
                    else (
                        "local_tensor_development_simulator"
                        if self.local_simulation
                        else "local"
                    )
                ),
                "distributed_backend_policy": (
                    self.backend_policy.summary() if self.backend_policy else None
                ),
                "distribution_semantics": (
                    "slice_parallel_state_with_local_facade"
                    if has_slice_parallel_state
                    else "replicated_single_rank"
                ),
                "scalability_claim_allowed": False,
                "scalability_blockers": tuple(
                    blocker
                    for blocker, active in (
                        ("single_process_development_simulator", self.local_simulation),
                        ("full_local_tensor_network_facade", has_slice_parallel_state),
                        (
                            "expectation_methods_use_local_state",
                            has_slice_parallel_state,
                        ),
                    )
                    if active
                ),
                "rank": self.context.rank if self.context else 0,
                "rank_placement": rank_placement,
                "communication_tiers": {
                    "model": (
                        "local_simulated_all_reduce"
                        if self.local_simulation
                        else "collective_topology_dependent"
                    ),
                    "collective": (
                        "all_reduce_sum" if has_slice_parallel_state else None
                    ),
                    "reduction_tensor_bytes": reduction_tensor_bytes,
                    "inter_node_collective_possible": bool(
                        rank_placement["node_count"] > 1 and self.world_size > 1
                    ),
                    "note": "Exact intra-node/inter-node bytes depend on torch.distributed/NCCL collective algorithm.",
                },
                "slice_tasks": len(self.tasks),
                "tasks_by_rank": {
                    rank: sum(1 for task in self.tasks if task.rank == rank)
                    for rank in range(self.world_size)
                },
                "rank_partial_bytes_by_rank": {
                    rank: self.rank_partial_bytes.get(rank, 0)
                    for rank in range(self.world_size)
                },
                "local_memory_bytes_by_rank": (
                    tuple(
                        self.rank_partial_bytes.get(rank, 0)
                        for rank in range(self.world_size)
                    )
                    if self.rank_partial_bytes
                    else None
                ),
                "jax_distributed_plan": self.jax_distributed_plan,
            }
        )
        return summary
