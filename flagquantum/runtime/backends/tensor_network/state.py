"""Result objects for distributed tensor-network execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from ....simulation.mps.rank_local import tensor_nbytes as _tensor_nbytes
from ....simulation.tensor_network.state import TensorNetworkState
from ...distributed.backend_policy import DistributedBackendPolicy
from ...distributed.context import TorchDistributedContext, _rank_placement_summary
from .sliced_tasks import DistributedTNSliceTask


def _tasks_by_rank(
    tasks: Sequence[DistributedTNSliceTask],
) -> dict[int, int]:
    counts: dict[int, int] = {}
    for task in tasks:
        counts[task.owner_rank] = counts.get(task.owner_rank, 0) + 1
    return counts


class DistributedTensorNetworkState:
    """Distributed tensor-network result with slice-task metadata."""

    def __init__(
        self,
        local_state: TensorNetworkState,
        *,
        world_size: int,
        tasks: Sequence[DistributedTNSliceTask],
        context: TorchDistributedContext | None = None,
        state_cache: torch.Tensor | None = None,
        backend_policy: DistributedBackendPolicy | None = None,
        local_simulation: bool = False,
        rank_partial_bytes: Mapping[int, int] | None = None,
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
                        ("expectation_methods_use_local_state", has_slice_parallel_state),
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
                    rank: sum(1 for task in self.tasks if task.owner_rank == rank)
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
            }
        )
        return summary


@dataclass(frozen=True)
class DistributedTensorNetworkAmplitude:
    """A scalar-output distributed TN result with auditable slice ownership."""

    value: torch.Tensor
    world_size: int
    tasks: tuple[DistributedTNSliceTask, ...]
    rank_partial_bytes: Mapping[int, int]
    distribution_semantics: str
    working_set_preflight: Mapping[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "state_mode": "distributed_tensor_network_amplitude",
            "output_target": "single_amplitude",
            "world_size": self.world_size,
            "slice_tasks": len(self.tasks),
            "tasks_by_rank": _tasks_by_rank(self.tasks),
            "rank_partial_bytes_by_rank": dict(self.rank_partial_bytes),
            "reduction_payload_bytes": _tensor_nbytes(self.value),
            "distribution_semantics": self.distribution_semantics,
            "full_state_materialized": False,
            "working_set_preflight": dict(self.working_set_preflight),
        }


@dataclass(frozen=True)
class DistributedTensorNetworkExpectation:
    """A scalar Pauli expectation computed by distributed TN slicing."""

    value: torch.Tensor
    world_size: int
    tasks: tuple[DistributedTNSliceTask, ...]
    rank_partial_bytes: Mapping[int, int]
    observable_wires: tuple[int, ...]
    distribution_semantics: str
    working_set_preflight: Mapping[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "state_mode": "distributed_tensor_network_expectation",
            "output_target": "local_observables",
            "observable_wires": self.observable_wires,
            "world_size": self.world_size,
            "slice_tasks": len(self.tasks),
            "tasks_by_rank": _tasks_by_rank(self.tasks),
            "rank_partial_bytes_by_rank": dict(self.rank_partial_bytes),
            "reduction_payload_bytes": _tensor_nbytes(self.value),
            "distribution_semantics": self.distribution_semantics,
            "full_state_materialized": False,
            "working_set_preflight": dict(self.working_set_preflight),
        }


@dataclass(frozen=True)
class DistributedTensorNetworkAmplitudes:
    """A small shared-contraction amplitude batch."""

    values: torch.Tensor
    world_size: int
    tasks: tuple[DistributedTNSliceTask, ...]
    rank_partial_bytes: Mapping[int, int]
    target_count: int
    distribution_semantics: str
    working_set_preflight: Mapping[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "state_mode": "distributed_tensor_network_amplitudes",
            "output_target": "few_amplitudes",
            "target_count": self.target_count,
            "world_size": self.world_size,
            "slice_tasks": len(self.tasks),
            "tasks_by_rank": _tasks_by_rank(self.tasks),
            "rank_partial_bytes_by_rank": dict(self.rank_partial_bytes),
            "reduction_payload_bytes": _tensor_nbytes(self.values),
            "distribution_semantics": self.distribution_semantics,
            "shared_contraction": True,
            "full_state_materialized": False,
            "working_set_preflight": dict(self.working_set_preflight),
        }


@dataclass(frozen=True)
class DistributedTensorNetworkExpectations:
    """A shared-contraction batch of Pauli-product expectations."""

    values: torch.Tensor
    world_size: int
    tasks: tuple[DistributedTNSliceTask, ...]
    rank_partial_bytes: Mapping[int, int]
    observable_count: int
    distribution_semantics: str
    working_set_preflight: Mapping[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "state_mode": "distributed_tensor_network_expectations",
            "output_target": "local_observables",
            "observable_count": self.observable_count,
            "world_size": self.world_size,
            "slice_tasks": len(self.tasks),
            "tasks_by_rank": _tasks_by_rank(self.tasks),
            "rank_partial_bytes_by_rank": dict(self.rank_partial_bytes),
            "reduction_payload_bytes": _tensor_nbytes(self.values),
            "distribution_semantics": self.distribution_semantics,
            "shared_contraction": True,
            "full_state_materialized": False,
            "working_set_preflight": dict(self.working_set_preflight),
        }


__all__ = [
    "DistributedTensorNetworkState",
    "DistributedTensorNetworkAmplitude",
    "DistributedTensorNetworkAmplitudes",
    "DistributedTensorNetworkExpectation",
    "DistributedTensorNetworkExpectations",
]
