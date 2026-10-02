"""Explicit full-state gather for rank-local distributed statevector results.

The production executor keeps one amplitude shard per rank and refuses to
reconstruct the global state, because a reconstruction whose cost grows with the
whole state is not a distributed result. A workload can still need the global
amplitude vector as its own product: an export, a full-distribution readout, or a
comparison against a single-device reference. This module is that path, and it is
kept out of the execution loop so the cost has to be asked for by name.

The gather is a collective over the whole state, so it moves
``(world_size - 1) / world_size`` of the state through every rank and leaves a
full copy resident on each of them. That is the opposite of what sharding buys,
which is why the result carries the gather's byte counts, reports
``distribution_semantics = "replicated_per_rank"``, and keeps a blocker of its
own. Nothing here can be read as a scaling measurement, and the rank-local
executor result stays the scaling claim surface.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from .errors import FullStateMaterializationError
from .forward import TorchDistributedStatevectorResult
from .models import DistributedStatevectorPlan

#: A gather trades the shard's memory and its communication volume for one
#: resident copy of the whole state on every rank. That is a cost a scaling
#: result cannot carry, so the gather states it rather than leaving a reader to
#: infer it from the byte counts.
BLOCKER_FULL_STATE_GATHER_IS_NOT_A_SCALING_RESULT = (
    "full_state_gather_is_not_a_scaling_result"
)

#: Layouts that publish a complete logical-to-physical wire permutation, which is
#: what turns the internal physical basis order back into the logical one.
_GATHERABLE_WIRE_LAYOUTS = frozenset({"canonical", "communication_aware", "persistent"})

#: Distributions whose rank ownership follows from the plan alone. A distribution
#: that places shards by an index vector the ranks do not publish cannot be placed
#: back into one vector by a collective that moves amplitudes only.
_GATHERABLE_DISTRIBUTIONS = frozenset(
    {
        "qubit_address_sharded",
        "contiguous_amplitude_range",
        "replicated_single_rank",
    }
)


def _published_logical_to_physical_wires(
    result: TorchDistributedStatevectorResult,
) -> tuple[int, ...]:
    """Return the result's wire permutation, or refuse to reconstruct without it.

    The amplitudes a rank holds live in the executor's internal basis order. That
    order is only recoverable from ``logical_to_physical_wires``; returning the
    amplitudes without applying it would hand back a permuted vector under a
    canonical name.
    """

    n_wires = int(result.plan.n_wires)
    if result.wire_layout not in _GATHERABLE_WIRE_LAYOUTS:
        raise FullStateMaterializationError(
            f"wire layout {result.wire_layout!r} publishes no logical-to-physical "
            "permutation; the gathered amplitudes cannot be returned in logical "
            "basis order"
        )
    mapping = tuple(int(wire) for wire in result.logical_to_physical_wires)
    if sorted(mapping) != list(range(n_wires)):
        raise FullStateMaterializationError(
            f"logical_to_physical_wires is not a permutation of {n_wires} wires: "
            f"{mapping}"
        )
    return mapping


def _expected_global_indices(
    plan: DistributedStatevectorPlan, *, rank: int
) -> torch.Tensor:
    """Return the global amplitude indices the plan assigns to one rank."""

    offsets = torch.arange(int(plan.shards[rank].local_amplitudes), dtype=torch.long)
    if plan.distribution == "qubit_address_sharded":
        return (offsets << len(plan.sharded_wires)) | int(rank)
    return offsets + int(plan.shards[rank].amplitude_start)


def _check_published_indices(
    plan: DistributedStatevectorPlan,
    *,
    rank: int,
    global_indices: torch.Tensor,
) -> None:
    """Cross-check the plan's ownership against an explicitly published index map.

    The executor leaves this tensor empty when it represents ownership compactly,
    so an empty map is not a disagreement. A non-empty map that contradicts the
    plan is: the two disagree about which amplitudes this rank owns, and placing
    the block would silently write them at the wrong positions.
    """

    if not global_indices.numel():
        return
    expected = _expected_global_indices(plan, rank=rank)
    published = global_indices.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    if published.numel() != expected.numel() or not bool(
        torch.equal(published, expected)
    ):
        raise FullStateMaterializationError(
            f"rank {rank} publishes global indices that disagree with the plan's "
            "rank ownership; refusing to place the gathered amplitudes"
        )


def _place_rank_blocks(
    blocks: Sequence[torch.Tensor],
    plan: DistributedStatevectorPlan,
) -> torch.Tensor:
    """Place every rank's local amplitudes at the global indices the plan assigns."""

    world_size = len(blocks)
    local_amplitudes = int(plan.shards[0].local_amplitudes)
    internal = torch.empty(
        (int(plan.bsz), int(plan.total_amplitudes)),
        dtype=blocks[0].dtype,
        device=blocks[0].device,
    )
    if plan.distribution == "qubit_address_sharded":
        # The rank address occupies the low bits of the global index, so rank `r`
        # owns every world_size-th amplitude starting at `r`.
        for rank, block in enumerate(blocks):
            internal[:, rank::world_size] = block
        return internal
    for rank, block in enumerate(blocks):
        start = int(plan.shards[rank].amplitude_start)
        internal[:, start : start + local_amplitudes] = block
    return internal


def _canonical_basis_order(
    internal: torch.Tensor,
    mapping: tuple[int, ...],
) -> torch.Tensor:
    """Reorder a vector from the internal physical basis to the logical basis.

    ``mapping[logical_wire]`` is the physical wire that carries it, and both
    indexes are read most-significant-wire-first, which is the convention the
    probe's own validation readout uses.
    """

    n_wires = len(mapping)
    if mapping == tuple(range(n_wires)):
        return internal
    logical = torch.arange(internal.shape[1], dtype=torch.long)
    physical = torch.zeros_like(logical)
    for logical_wire, physical_wire in enumerate(mapping):
        bit = (logical >> (n_wires - logical_wire - 1)) & 1
        physical = physical | (bit << (n_wires - physical_wire - 1))
    return internal.index_select(1, physical.to(internal.device))


def _rank_global_index_map(
    plan: DistributedStatevectorPlan,
) -> tuple[dict[str, Any], ...]:
    """Publish which global amplitudes each rank owns."""

    if plan.distribution == "qubit_address_sharded":
        return tuple(
            {
                "rank": int(shard.rank),
                "placement": "strided_rank_address_bit",
                "global_offset": int(shard.rank),
                "global_stride": int(plan.world_size),
                "local_amplitudes": int(shard.local_amplitudes),
            }
            for shard in plan.shards
        )
    return tuple(
        {
            "rank": int(shard.rank),
            "placement": "contiguous_amplitude_range",
            "global_start": int(shard.amplitude_start),
            "global_end": int(shard.amplitude_end),
            "local_amplitudes": int(shard.local_amplitudes),
        }
        for shard in plan.shards
    )


def _validate_gatherable(
    result: TorchDistributedStatevectorResult,
    mapping: tuple[int, ...],
) -> torch.Tensor:
    """Refuse a result the collective cannot place back into one vector."""

    plan = result.plan
    if plan.distribution not in _GATHERABLE_DISTRIBUTIONS:
        raise FullStateMaterializationError(
            f"distribution {plan.distribution!r} does not place shards by rank "
            "address or contiguous range, so the gathered amplitudes cannot be "
            "placed back into one vector"
        )
    local = result.shard_state.amplitudes
    if local.element_size() != int(plan.complex_bytes):
        raise FullStateMaterializationError(
            f"the shard holds {local.element_size()}-byte amplitudes but the plan "
            f"was built for {plan.complex_bytes}-byte ones"
        )
    local_amplitudes = int(plan.shards[0].local_amplitudes)
    uneven_ranks = [
        int(shard.rank)
        for shard in plan.shards
        if int(shard.local_amplitudes) != local_amplitudes
    ]
    if uneven_ranks or local_amplitudes * int(plan.world_size) != int(
        plan.total_amplitudes
    ):
        raise FullStateMaterializationError(
            "the plan gives ranks uneven amplitude counts; the gather moves one "
            "equal-sized block per rank"
        )
    if int(local.shape[0]) != int(plan.bsz) or int(local.shape[1]) != local_amplitudes:
        raise FullStateMaterializationError(
            f"the local shard has shape {tuple(local.shape)} but the plan assigns "
            f"this rank {plan.bsz} batch row(s) of {local_amplitudes} amplitudes"
        )
    _check_published_indices(
        plan,
        rank=int(result.shard_state.rank),
        global_indices=result.shard_state.global_indices,
    )
    if len(mapping) != int(plan.n_wires):
        raise FullStateMaterializationError(
            f"the wire permutation covers {len(mapping)} wires but the plan was "
            f"built for {plan.n_wires}"
        )
    return local


@dataclass(frozen=True)
class DistributedStatevectorGatherResult:
    """One gathered full amplitude vector plus the cost of having gathered it."""

    state: torch.Tensor
    plan: DistributedStatevectorPlan
    wire_layout: str
    logical_to_physical_wires: tuple[int, ...]
    rank_global_index_map: tuple[dict[str, Any], ...]
    local_block_bytes: int
    full_state_bytes: int
    gather_bytes_per_rank: int
    total_gather_bytes: int
    peak_resident_bytes_per_rank: int

    def summary(self) -> dict[str, Any]:
        """Describe the gathered state and the gather's cost, never a speedup."""

        return {
            "executor": "distributed_statevector_full_state_gather_v1",
            "operation": "all_gather",
            "operation_semantics": "explicit_full_state_gather",
            "gather_cost_semantics": (
                "every_rank_receives_every_other_ranks_shard_and_keeps_the_whole_state"
            ),
            "claim_evidence_type": "production_runtime",
            "distribution_semantics": "replicated_per_rank",
            "scalability_claim_allowed": False,
            "release_gate_allowed": False,
            "full_state_materialization": True,
            "amplitude_basis_order": "canonical_logical",
            "wire_layout": self.wire_layout,
            "logical_to_physical_wires": self.logical_to_physical_wires,
            "n_wires": int(self.plan.n_wires),
            "bsz": int(self.plan.bsz),
            "world_size": int(self.plan.world_size),
            "local_world_size": int(self.plan.local_world_size),
            "node_count": int(self.plan.node_count),
            "state_partition": self.plan.distribution,
            "total_amplitudes": int(self.plan.total_amplitudes),
            "rank_global_index_map": self.rank_global_index_map,
            "local_block_bytes": self.local_block_bytes,
            "full_state_bytes": self.full_state_bytes,
            "gather_bytes_per_rank": self.gather_bytes_per_rank,
            "total_gather_bytes": self.total_gather_bytes,
            "peak_resident_bytes_per_rank": self.peak_resident_bytes_per_rank,
            "blockers": (BLOCKER_FULL_STATE_GATHER_IS_NOT_A_SCALING_RESULT,),
        }


def gather_distributed_statevector(
    result: TorchDistributedStatevectorResult,
    *,
    process_group: Any | None = None,
) -> DistributedStatevectorGatherResult:
    """Gather rank-local shards into one amplitude vector in logical basis order.

    Every rank that takes part receives the whole state, so the returned vector is
    identical on all of them and each one holds a full copy. Pass the process
    group the result was produced on; its world size has to match the plan.

    Parameters
    ----------
    result:
        A rank-local result from ``execute_torch_distributed_statevector``.
    process_group:
        The group the result was produced on. ``None`` uses the default group.

    Returns
    -------
    DistributedStatevectorGatherResult
        The gathered vector, the rank ownership map, and the gather's bytes.

    Raises
    ------
    FullStateMaterializationError
        If the result publishes no wire permutation, if the plan gives ranks
        uneven amplitude counts, if the published global indices disagree with
        the plan, or if the process group's world size differs from the plan's.

    Examples
    --------
    >>> import torch, flagquantum as fq
    >>> from flagquantum.runtime.executors.statevector import (
    ...     execute_torch_distributed_statevector,
    ...     gather_distributed_statevector,
    ... )
    >>> circuit = fq.Circuit(3).h(0).cx(0, 2)
    >>> result = execute_torch_distributed_statevector(circuit, device="cpu")
    >>> gathered = gather_distributed_statevector(result)
    >>> bool(torch.allclose(gathered.state[0], circuit.state().reshape(-1)))
    True
    """

    mapping = _published_logical_to_physical_wires(result)
    local = _validate_gatherable(result, mapping)
    world_size = dist.get_world_size(process_group) if dist.is_initialized() else 1
    if world_size != int(result.plan.world_size):
        raise FullStateMaterializationError(
            f"the process group holds {world_size} ranks but the plan was built for "
            f"{result.plan.world_size}; gather on the group the result came from"
        )

    if world_size > 1:
        blocks: list[torch.Tensor] = [
            torch.empty_like(local) for _ in range(world_size)
        ]
        dist.all_gather(blocks, local.contiguous(), group=process_group)
    else:
        blocks = [local]
    state = _canonical_basis_order(_place_rank_blocks(blocks, result.plan), mapping)

    local_block_bytes = int(local.numel() * local.element_size())
    full_state_bytes = (
        int(result.plan.total_amplitudes) * int(result.plan.bsz) * local.element_size()
    )
    gather_bytes_per_rank = (world_size - 1) * local_block_bytes
    return DistributedStatevectorGatherResult(
        state=state,
        plan=result.plan,
        wire_layout=str(result.wire_layout),
        logical_to_physical_wires=mapping,
        rank_global_index_map=_rank_global_index_map(result.plan),
        local_block_bytes=local_block_bytes,
        full_state_bytes=full_state_bytes,
        gather_bytes_per_rank=gather_bytes_per_rank,
        total_gather_bytes=world_size * gather_bytes_per_rank,
        peak_resident_bytes_per_rank=full_state_bytes + local_block_bytes,
    )


__all__ = (
    "BLOCKER_FULL_STATE_GATHER_IS_NOT_A_SCALING_RESULT",
    "DistributedStatevectorGatherResult",
    "gather_distributed_statevector",
)
