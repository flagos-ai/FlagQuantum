"""Explicit full-state export for rank-owned distributed matrix product states.

The production MPS executor keeps a contiguous block of sites per rank and
refuses to reconstruct the global state, because a reconstruction whose cost
grows with the whole state is not a distributed result. A workload can still need
the global state as its own product: an export, a full-distribution readout, or a
comparison against a single-device reference. This module is that path, and it is
kept out of the execution loop so the cost has to be asked for by name.

The sites are exchanged in one all-gather rather than in a chain of
point-to-point transfers. A rank that had finished sending its own sites would
otherwise be free to join the next collective while another rank was still
receiving, and two phases would then drive one communicator at once; an
all-gather posts every rank's contribution before any of them returns, so there is
no such window.

Every rank keeps the whole state when this returns, and the reconstruction costs
one exchange of every site. That is the opposite of what sharding buys, which is
why the result carries the export's byte counts, reports
``distribution_semantics = "replicated_per_rank"``, and keeps a blocker of its
own. Nothing here can be read as a scaling measurement, and the rank-local
executor result stays the scaling claim surface.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from ....simulation.mps.rank_local import tensor_nbytes as _tensor_nbytes
from ....simulation.mps.state import MPSState
from ...distributed.context import resolve_node_count
from .errors import MPSFullMaterializationError
from .metadata_transport import all_gather_json
from .records import TorchDistributedMPSForwardResult
from .state import RankOwnedMPSState

#: An export trades the shard's memory and its communication volume for one
#: resident copy of the whole state on every rank. That is a cost a scaling
#: result cannot carry, so the export states it rather than leaving a reader to
#: infer it from the byte counts.
BLOCKER_FULL_MPS_GATHER_IS_NOT_A_SCALING_RESULT = (
    "full_mps_gather_is_not_a_scaling_result"
)

#: The site order an export publishes. The rank-owned state indexes its tensors by
#: logical wire and its ownership map partitions exactly those wires, so placing
#: each gathered site at its own wire index reproduces the logical order. A state
#: whose tensors were indexed some other way could not be exported under this
#: name, which is why the name is published rather than assumed.
SITE_ORDER_CANONICAL_LOGICAL = "canonical_logical"


def _owned_wires(state: RankOwnedMPSState) -> tuple[int, ...]:
    """Return this rank's sites in wire order, refusing an empty or odd shard."""

    wires = tuple(sorted(int(wire) for wire in state.local_tensors))
    if not wires:
        raise MPSFullMaterializationError(
            f"rank {state.rank} owns no MPS site to export"
        )
    local = {wire: state.local_tensors[wire] for wire in wires}
    reference = local[wires[0]]
    if any(tensor.dtype != reference.dtype for tensor in local.values()):
        raise MPSFullMaterializationError(
            "the owned MPS sites do not share one dtype: "
            f"{[(wire, str(local[wire].dtype)) for wire in wires]}"
        )
    if any(tensor.device != reference.device for tensor in local.values()):
        raise MPSFullMaterializationError(
            "the owned MPS sites do not share one device: "
            f"{[(wire, str(local[wire].device)) for wire in wires]}"
        )
    return wires


def _validated_ownership(state: RankOwnedMPSState) -> tuple[tuple[int, ...], ...]:
    """Return the rank ownership map, or refuse to reconstruct without one.

    The gathered sites can only be returned as one state if the ownership map
    says which rank holds which wire: a state that published tensors without a
    map would leave each rank's block unplaceable, and placing it by assumption
    would hand back a state under a canonical name that is not the canonical one.
    """

    ownership = tuple(tuple(int(wire) for wire in wires) for wires in state.ownership)
    if len(ownership) != int(state.world_size):
        raise MPSFullMaterializationError(
            f"the ownership map names {len(ownership)} ranks but the state was "
            f"built for {state.world_size}"
        )
    if any(not wires for wires in ownership):
        raise MPSFullMaterializationError(
            "the ownership map gives at least one rank no site, so that rank "
            "contributes nothing to the gather and cannot receive the state"
        )
    flattened = sorted(wire for wires in ownership for wire in wires)
    expected = list(range(int(state.n_wires)))
    if flattened != expected:
        raise MPSFullMaterializationError(
            "the ownership map is not a partition of the state's wires: "
            f"covers={flattened} expected={expected}"
        )
    owned = tuple(sorted(int(wire) for wire in state.local_tensors))
    if owned != tuple(sorted(ownership[int(state.rank)])):
        raise MPSFullMaterializationError(
            f"rank {state.rank} holds sites {list(owned)} but the ownership map "
            f"gives it {list(ownership[int(state.rank)])}"
        )
    return ownership


def _site_table(
    wires: Sequence[int], tensors: Mapping[int, torch.Tensor]
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    return tuple(
        (int(wire), tuple(int(dim) for dim in tensors[wire].shape)) for wire in wires
    )


def _place_gathered_sites(
    tables: Sequence[Sequence[tuple[int, tuple[int, ...]]]],
    buffers: Sequence[torch.Tensor],
    *,
    ownership: tuple[tuple[int, ...], ...],
) -> tuple[torch.Tensor, ...]:
    """Place every rank's block at the wire indices the ownership map assigns.

    Each rank's contribution is padded to the widest block so one all-gather can
    carry them all, so the placement reads each rank's own length table rather
    than a fixed site count: a rank whose sites are smaller than the widest block
    would otherwise have the next rank's padding read as one of its own sites.
    """

    placed: dict[int, torch.Tensor] = {}
    for rank, (table, buffer) in enumerate(zip(tables, buffers, strict=True)):
        expected = sorted(ownership[rank])
        if sorted(wire for wire, _ in table) != expected:
            raise MPSFullMaterializationError(
                f"rank {rank} contributed sites {[wire for wire, _ in table]} but "
                f"the ownership map gives it {expected}"
            )
        offset = 0
        for wire, shape in table:
            count = int(math.prod(shape))
            placed[int(wire)] = buffer[offset : offset + count].reshape(shape)
            offset += count
    expected_wires = list(range(sum(len(wires) for wires in ownership)))
    if sorted(placed) != expected_wires:
        raise MPSFullMaterializationError(
            "the exported MPS is missing sites: "
            f"present={sorted(placed)} expected={expected_wires}"
        )
    return tuple(placed[wire] for wire in expected_wires)


@dataclass(frozen=True)
class DistributedMPSExportResult:
    """One exported full matrix product state plus the cost of having exported it."""

    state: MPSState
    ownership: tuple[tuple[int, ...], ...]
    local_site_count: int
    local_site_bytes: int
    padded_block_bytes: int
    full_state_bytes: int
    gather_bytes_per_rank: int
    total_gather_bytes: int
    peak_resident_bytes_per_rank: int
    local_world_size: int
    node_count: int

    def summary(self) -> dict[str, Any]:
        """Describe the exported state and the export's cost, never a speedup."""

        return {
            "executor": "distributed_mps_full_state_export_v1",
            "operation": "all_gather",
            "operation_semantics": "explicit_full_mps_gather",
            "gather_cost_semantics": (
                "every_rank_receives_every_other_ranks_sites_and_keeps_the_whole_state"
            ),
            "claim_evidence_type": "production_runtime",
            "distribution_semantics": "replicated_per_rank",
            "scalability_claim_allowed": False,
            "release_gate_allowed": False,
            "full_state_materialization": True,
            "site_order": SITE_ORDER_CANONICAL_LOGICAL,
            "n_wires": int(self.state.n_wires),
            "bsz": int(self.state.bsz),
            "max_bond": int(self.state.max_bond),
            "site_parameter_count": int(self.state.parameter_count),
            "world_size": len(self.ownership),
            "local_world_size": int(self.local_world_size),
            "node_count": int(self.node_count),
            "state_partition": "contiguous_site_block_per_rank",
            "site_ownership": self.ownership,
            "local_site_count": int(self.local_site_count),
            "local_site_bytes": int(self.local_site_bytes),
            "padded_block_bytes": int(self.padded_block_bytes),
            "full_state_bytes": int(self.full_state_bytes),
            "gather_bytes_per_rank": int(self.gather_bytes_per_rank),
            "total_gather_bytes": int(self.total_gather_bytes),
            "peak_resident_bytes_per_rank": int(self.peak_resident_bytes_per_rank),
            "blockers": (BLOCKER_FULL_MPS_GATHER_IS_NOT_A_SCALING_RESULT,),
        }


def export_distributed_mps(
    result: TorchDistributedMPSForwardResult,
    *,
    process_group: Any | None = None,
) -> DistributedMPSExportResult:
    """Gather rank-owned sites into one matrix product state in logical site order.

    Every rank that takes part receives the whole state, so the returned state is
    identical on all of them and each one holds a full copy. Pass the process
    group the result was produced on; its world size has to match the state's.

    Parameters
    ----------
    result:
        A rank-local result from ``execute_torch_distributed_mps_forward``.
    process_group:
        The group the result was produced on. ``None`` uses the default group.

    Returns
    -------
    DistributedMPSExportResult
        The exported state, the site ownership map, and the export's bytes.

    Raises
    ------
    MPSFullMaterializationError
        If the ownership map is not a partition of the state's wires, if this rank
        does not hold the sites the map gives it, if the owned sites disagree
        about their dtype or device, or if the process group's world size differs
        from the state's.
    """

    state = result.shard_state
    ownership = _validated_ownership(state)
    wires = _owned_wires(state)
    tensors = {wire: state.local_tensors[wire] for wire in wires}
    world_size = dist.get_world_size(process_group) if dist.is_initialized() else 1
    if world_size != int(state.world_size):
        raise MPSFullMaterializationError(
            f"the process group holds {world_size} ranks but the state was built for "
            f"{state.world_size}; export on the group the result came from"
        )

    table = _site_table(wires, tensors)
    tables = all_gather_json(table) if world_size > 1 else (table,)
    # One all-gather carries every rank's sites, so each block is padded to the
    # widest of them and the padding is counted as part of what moved.
    elements_per_rank = max(
        (sum(int(math.prod(shape)) for _, shape in item) for item in tables), default=0
    )
    reference = tensors[wires[0]]
    flat = torch.cat([tensors[wire].reshape(-1) for wire in wires])
    block = flat.new_zeros(elements_per_rank)
    block[: flat.numel()] = flat
    if world_size > 1:
        buffers = [torch.empty_like(block) for _ in range(world_size)]
        dist.all_gather(buffers, block.contiguous(), group=process_group)
    else:
        buffers = [block]
    placed = _place_gathered_sites(tables, buffers, ownership=ownership)
    exported = MPSState(list(placed), config=state.config)

    element_bytes = int(reference.element_size())
    local_site_bytes = sum(_tensor_nbytes(tensor) for tensor in tensors.values())
    padded_block_bytes = elements_per_rank * element_bytes
    # The exported state is the resident copy each rank keeps, so its bytes are
    # the whole state's own parameter bytes rather than the padded exchange.
    full_state_bytes = int(exported.parameter_count) * element_bytes
    gather_bytes_per_rank = (world_size - 1) * padded_block_bytes
    return DistributedMPSExportResult(
        state=exported,
        ownership=ownership,
        local_site_count=len(wires),
        local_site_bytes=local_site_bytes,
        padded_block_bytes=padded_block_bytes,
        full_state_bytes=full_state_bytes,
        gather_bytes_per_rank=gather_bytes_per_rank,
        total_gather_bytes=world_size * gather_bytes_per_rank,
        peak_resident_bytes_per_rank=full_state_bytes + local_site_bytes,
        local_world_size=int(result.local_world_size),
        node_count=resolve_node_count(world_size, int(result.local_world_size)),
    )


__all__ = (
    "BLOCKER_FULL_MPS_GATHER_IS_NOT_A_SCALING_RESULT",
    "SITE_ORDER_CANONICAL_LOGICAL",
    "DistributedMPSExportResult",
    "export_distributed_mps",
)
