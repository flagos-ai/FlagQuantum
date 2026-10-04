"""Result objects for distributed tensor-network execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ....simulation.mps.rank_local import tensor_nbytes as _tensor_nbytes
from ....simulation.tensor_network.state import TensorNetworkState
from ...distributed.backend_policy import DistributedBackendPolicy
from ...distributed.context import TorchDistributedContext, _rank_placement_summary
from .sliced_tasks import DistributedTNSliceTask


def _sliced_labels(tasks: Sequence[DistributedTNSliceTask]) -> tuple[int, ...]:
    """The internal labels the task plan actually split the contraction over.

    Recording them turns "the workload was sliced" into a fact a reader can
    check against the plan, rather than something only the caller that passed
    the labels knows.
    """

    return tuple(sorted({label for task in tasks for label, _ in task.assignments}))


def _tasks_by_rank(
    tasks: Sequence[DistributedTNSliceTask],
) -> dict[int, int]:
    counts: dict[int, int] = {}
    for task in tasks:
        counts[task.owner_rank] = counts.get(task.owner_rank, 0) + 1
    return counts


def _communication_tiers(
    *,
    reduction_tensor_bytes: int,
    rank_placement: Mapping[str, Any],
    model: str,
    collective: str | None,
) -> dict[str, Any]:
    """Report which reduction crossed a node boundary and how much of it did.

    Slice parallelism reduces one sparse output per rank, so the reduction
    payload is the whole inter-rank contribution. A single-rank-per-node
    placement crosses a node boundary on every leg, a single-node placement on
    none, and a mixed placement cannot be attributed without the collective
    algorithm, so it reports no attributed bytes rather than a guess.
    """

    node_count = int(rank_placement.get("node_count", 1) or 1)
    local_world_size = int(rank_placement.get("local_world_size", 1) or 1)
    size = int(reduction_tensor_bytes) if collective is not None else 0
    pure_inter_node = node_count > 1 and local_world_size == 1
    pure_intra_node = node_count == 1
    return {
        "model": model,
        "collective": collective,
        "reduction_tensor_bytes": int(reduction_tensor_bytes),
        "inter_node_collective_bytes": size if pure_inter_node else 0,
        "intra_node_collective_bytes": size if pure_intra_node else 0,
        "unattributed_collective_bytes": (
            0 if (pure_inter_node or pure_intra_node) else size
        ),
        "inter_node_collective_possible": bool(node_count > 1),
        "note": "Exact intra-node/inter-node bytes depend on torch.distributed/NCCL collective algorithm.",
    }


def _collective_for_semantics(distribution_semantics: str) -> tuple[str, str | None]:
    """Name the reduction that ran and how it reduced, per execution semantics."""

    if distribution_semantics == "sharded_across_ranks":
        return "collective_topology_dependent", "all_reduce_sum"
    if (
        distribution_semantics
        == "local_simulated_slice_parallel_sparse_output_reduction"
    ):
        return "local_simulated_all_reduce", "local_simulated_all_reduce"
    return "single_device", None


def _claim_evidence_type(*, backend: str | None, initialized: bool) -> str:
    """Classify what a run may be cited for, from what actually executed.

    The values come from the audit vocabulary, so a reader -- or the distributed
    evidence contract -- can tell an accelerator-backed run from a simulated
    one. Only a real process group on `nccl` or `flagos` executed on the
    accelerators, and only that run is runtime evidence for them; a `gloo`
    group, a single process, and the local slice simulator are all development
    executions, whichever topology they declare.
    """

    if initialized and str(backend) in {"nccl", "flagos"}:
        return "production_runtime"
    return "development_smoke"


def _context_claim_evidence_type(context: TorchDistributedContext | None) -> str:
    """The classification for a context a caller may or may not have built."""

    return _claim_evidence_type(
        backend=None if context is None else str(context.backend),
        initialized=bool(context is not None and context.initialized),
    )


def _distributed_output_evidence(
    *,
    world_size: int,
    rank_partial_bytes: Mapping[int, int],
    rank_placement: Mapping[str, Any],
    distribution_semantics: str,
    claim_evidence_type: str,
    scalability_blockers: Sequence[str],
    reduction_tensor_bytes: int,
) -> dict[str, Any]:
    """Placement, memory, communication, semantics, and blockers every output reports.

    The five distributed tensor-network entry points answer different questions
    but place their work identically, so they must answer "where did this run,
    what did it communicate, and what may be claimed from it" identically too.
    """

    model, collective = _collective_for_semantics(distribution_semantics)
    return {
        "world_size": world_size,
        "local_world_size": rank_placement["local_world_size"],
        "node_count": rank_placement["node_count"],
        "rank_placement": dict(rank_placement),
        "distribution_semantics": distribution_semantics,
        "claim_evidence_type": claim_evidence_type,
        "scalability_claim_allowed": False,
        "scalability_blockers": tuple(scalability_blockers),
        "communication_tiers": _communication_tiers(
            reduction_tensor_bytes=reduction_tensor_bytes,
            rank_placement=rank_placement,
            model=model,
            collective=collective,
        ),
        "rank_partial_bytes_by_rank": {
            rank: int(rank_partial_bytes.get(rank, 0)) for rank in range(world_size)
        },
        "local_memory_bytes_by_rank": tuple(
            int(rank_partial_bytes.get(rank, 0)) for rank in range(world_size)
        ),
    }


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
        rank_placement: Mapping[str, Any] | None = None,
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
        self.rank_placement = dict(rank_placement) if rank_placement else None

    @property
    def n_qubits(self) -> int:
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
        rank_placement = (
            dict(self.rank_placement)
            if self.rank_placement is not None
            else _rank_placement_summary(self.context, world_size=self.world_size)
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
                # The slices are contracted on rank owners, but the returned
                # state is the reduced full state that every rank materializes,
                # so the forward work and this object's distribution differ and
                # both are reported.
                "forward_distribution_semantics": (
                    "sharded_across_ranks"
                    if has_slice_parallel_state
                    else "single_device"
                ),
                "state_distribution_semantics": (
                    "replicated_full_state_after_all_reduce"
                    if has_slice_parallel_state
                    else "replicated_single_rank"
                ),
                "full_state_materialized": self._state_cache is not None,
                "claim_evidence_type": _context_claim_evidence_type(self.context),
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
                "communication_tiers": _communication_tiers(
                    reduction_tensor_bytes=reduction_tensor_bytes,
                    rank_placement=rank_placement,
                    model=(
                        "local_simulated_all_reduce"
                        if self.local_simulation
                        else "collective_topology_dependent"
                    ),
                    collective=("all_reduce_sum" if has_slice_parallel_state else None),
                ),
                "slice_tasks": len(self.tasks),
                "slice_labels": _sliced_labels(self.tasks),
                "tasks_by_rank": {
                    rank: sum(1 for task in self.tasks if task.owner_rank == rank)
                    for rank in range(self.world_size)
                },
                "rank_partial_bytes_by_rank": {
                    rank: self.rank_partial_bytes.get(rank, 0)
                    for rank in range(self.world_size)
                },
                "local_memory_bytes_by_rank": tuple(
                    self.rank_partial_bytes.get(rank, 0)
                    for rank in range(self.world_size)
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
    rank_placement: Mapping[str, Any]
    claim_evidence_type: str
    scalability_blockers: Sequence[str] = ()

    def summary(self) -> dict[str, Any]:
        reduction_tensor_bytes = _tensor_nbytes(self.value)
        return {
            "state_mode": "distributed_tensor_network_amplitude",
            "output_target": "single_amplitude",
            "slice_tasks": len(self.tasks),
            "slice_labels": _sliced_labels(self.tasks),
            "tasks_by_rank": _tasks_by_rank(self.tasks),
            "reduction_payload_bytes": reduction_tensor_bytes,
            "full_state_materialized": False,
            "working_set_preflight": dict(self.working_set_preflight),
            **_distributed_output_evidence(
                world_size=self.world_size,
                rank_partial_bytes=self.rank_partial_bytes,
                rank_placement=self.rank_placement,
                distribution_semantics=self.distribution_semantics,
                claim_evidence_type=self.claim_evidence_type,
                scalability_blockers=self.scalability_blockers,
                reduction_tensor_bytes=reduction_tensor_bytes,
            ),
        }


@dataclass(frozen=True)
class DistributedTensorNetworkExpectation:
    """A scalar Pauli expectation computed by distributed TN slicing."""

    value: torch.Tensor
    world_size: int
    tasks: tuple[DistributedTNSliceTask, ...]
    rank_partial_bytes: Mapping[int, int]
    observable_qubits: tuple[int, ...]
    distribution_semantics: str
    working_set_preflight: Mapping[str, Any]
    rank_placement: Mapping[str, Any]
    claim_evidence_type: str
    scalability_blockers: Sequence[str] = ()

    def summary(self) -> dict[str, Any]:
        reduction_tensor_bytes = _tensor_nbytes(self.value)
        return {
            "state_mode": "distributed_tensor_network_expectation",
            "output_target": "local_observables",
            "observable_wires": self.observable_qubits,
            "slice_tasks": len(self.tasks),
            "slice_labels": _sliced_labels(self.tasks),
            "tasks_by_rank": _tasks_by_rank(self.tasks),
            "reduction_payload_bytes": reduction_tensor_bytes,
            "full_state_materialized": False,
            "working_set_preflight": dict(self.working_set_preflight),
            **_distributed_output_evidence(
                world_size=self.world_size,
                rank_partial_bytes=self.rank_partial_bytes,
                rank_placement=self.rank_placement,
                distribution_semantics=self.distribution_semantics,
                claim_evidence_type=self.claim_evidence_type,
                scalability_blockers=self.scalability_blockers,
                reduction_tensor_bytes=reduction_tensor_bytes,
            ),
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
    rank_placement: Mapping[str, Any]
    claim_evidence_type: str
    scalability_blockers: Sequence[str] = ()

    def summary(self) -> dict[str, Any]:
        reduction_tensor_bytes = _tensor_nbytes(self.values)
        return {
            "state_mode": "distributed_tensor_network_amplitudes",
            "output_target": "few_amplitudes",
            "target_count": self.target_count,
            "slice_tasks": len(self.tasks),
            "slice_labels": _sliced_labels(self.tasks),
            "tasks_by_rank": _tasks_by_rank(self.tasks),
            "reduction_payload_bytes": reduction_tensor_bytes,
            "shared_contraction": True,
            "full_state_materialized": False,
            "working_set_preflight": dict(self.working_set_preflight),
            **_distributed_output_evidence(
                world_size=self.world_size,
                rank_partial_bytes=self.rank_partial_bytes,
                rank_placement=self.rank_placement,
                distribution_semantics=self.distribution_semantics,
                claim_evidence_type=self.claim_evidence_type,
                scalability_blockers=self.scalability_blockers,
                reduction_tensor_bytes=reduction_tensor_bytes,
            ),
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
    rank_placement: Mapping[str, Any]
    claim_evidence_type: str
    scalability_blockers: Sequence[str] = ()

    def summary(self) -> dict[str, Any]:
        reduction_tensor_bytes = _tensor_nbytes(self.values)
        return {
            "state_mode": "distributed_tensor_network_expectations",
            "output_target": "local_observables",
            "observable_count": self.observable_count,
            "slice_tasks": len(self.tasks),
            "slice_labels": _sliced_labels(self.tasks),
            "tasks_by_rank": _tasks_by_rank(self.tasks),
            "reduction_payload_bytes": reduction_tensor_bytes,
            "shared_contraction": True,
            "full_state_materialized": False,
            "working_set_preflight": dict(self.working_set_preflight),
            **_distributed_output_evidence(
                world_size=self.world_size,
                rank_partial_bytes=self.rank_partial_bytes,
                rank_placement=self.rank_placement,
                distribution_semantics=self.distribution_semantics,
                claim_evidence_type=self.claim_evidence_type,
                scalability_blockers=self.scalability_blockers,
                reduction_tensor_bytes=reduction_tensor_bytes,
            ),
        }


__all__ = [
    "DistributedTensorNetworkState",
    "DistributedTensorNetworkAmplitude",
    "DistributedTensorNetworkAmplitudes",
    "DistributedTensorNetworkExpectation",
    "DistributedTensorNetworkExpectations",
]
