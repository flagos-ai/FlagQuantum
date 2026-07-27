"""Rank-owned MPS state and partition contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ....core.ir import CircuitIR
from ....simulation.mps import MPSConfig
from .communication import _tensor_nbytes
from .errors import MPSFullMaterializationError, MPSReverseContractError


@dataclass(frozen=True)
class MPSPartition:
    rank: int
    wires: tuple[int, ...]
    tensor_bytes: int


@dataclass
class RankOwnedMPSState:
    n_wires: int
    bsz: int
    rank: int
    world_size: int
    config: MPSConfig
    local_tensors: dict[int, torch.Tensor]
    ownership: tuple[tuple[int, ...], ...]

    def owner(self, wire: int) -> int:
        for rank, wires in enumerate(self.ownership):
            if int(wire) in wires:
                return rank
        raise KeyError(f"wire {wire} has no MPS owner")

    def full_state(self) -> torch.Tensor:
        raise MPSFullMaterializationError(
            "rank-owned MPS state forbids production full-state materialization"
        )

    def summary(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "local_tensor_wires": tuple(sorted(self.local_tensors)),
            "local_tensor_count": len(self.local_tensors),
            "local_tensor_bytes": sum(
                _tensor_nbytes(tensor) for tensor in self.local_tensors.values()
            ),
            "ownership": self.ownership,
        }


def initial_mps_ownership(n_wires: int, world_size: int) -> tuple[tuple[int, ...], ...]:
    """Build the default contiguous site ownership for all ranks."""
    if world_size < 1 or world_size > n_wires:
        raise ValueError("rank-owned MPS requires 1 <= world_size <= n_wires")
    return tuple(
        tuple(range(rank * n_wires // world_size, (rank + 1) * n_wires // world_size))
        for rank in range(world_size)
    )


def validate_mps_ownership(
    ownership: Sequence[Sequence[int]], n_wires: int, world_size: int
) -> tuple[tuple[int, ...], ...]:
    """Validate a contiguous, non-empty, complete site partition."""
    normalized = tuple(tuple(int(wire) for wire in wires) for wires in ownership)
    if len(normalized) != world_size:
        raise ValueError("MPS ownership must contain exactly one shard per rank")
    if any(not wires for wires in normalized):
        raise ValueError("MPS ownership requires at least one site per rank")
    flattened = tuple(wire for wires in normalized for wire in wires)
    if flattened != tuple(range(n_wires)):
        raise ValueError(
            "MPS ownership must be ordered, contiguous, non-overlapping, and cover all sites"
        )
    return normalized


def normalize_rank_owned_initial_tensors(
    initial_tensors: Mapping[int, torch.Tensor],
    *,
    ownership: tuple[tuple[int, ...], ...],
    rank: int,
    world: int,
    n_wires: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[int, torch.Tensor], int, dict[int, tuple[int, int, int, int]]]:
    """Validate, clone, and globally reconcile a rank-owned initial MPS."""
    del world
    expected = set(ownership[rank])
    actual = {int(wire) for wire in initial_tensors}
    if actual != expected:
        raise MPSReverseContractError(
            f"rank {rank} initial MPS wires differ: expected {sorted(expected)}, "
            f"received {sorted(actual)}"
        )
    local: dict[int, torch.Tensor] = {}
    shapes: dict[int, tuple[int, int, int, int]] = {}
    batch_size: int | None = None
    for wire in sorted(expected):
        tensor = initial_tensors[wire]
        if tensor.ndim != 4 or int(tensor.shape[2]) != 2:
            raise MPSReverseContractError(
                f"initial MPS tensor {wire} must have shape [batch,left,2,right]"
            )
        if tensor.device != device or tensor.dtype != dtype:
            raise MPSReverseContractError(
                f"initial MPS tensor {wire} must use {device}/{dtype}, "
                f"received {tensor.device}/{tensor.dtype}"
            )
        if batch_size is None:
            batch_size = int(tensor.shape[0])
        elif int(tensor.shape[0]) != batch_size:
            raise MPSReverseContractError("initial MPS batch dimensions differ")
        shapes[wire] = tuple(int(value) for value in tensor.shape)
        local[wire] = tensor.detach().clone()
    shape_table = torch.zeros((n_wires, 4), dtype=torch.int64, device=device)
    for wire, shape in shapes.items():
        shape_table[wire] = torch.tensor(shape, dtype=torch.int64, device=device)
    dist.all_reduce(shape_table, op=dist.ReduceOp.SUM)
    global_shapes = {
        wire: tuple(int(value) for value in shape_table[wire].cpu().tolist())
        for wire in range(n_wires)
    }
    if set(global_shapes) != set(range(n_wires)):
        raise MPSReverseContractError(
            "initial MPS does not cover every wire exactly once"
        )
    batches = {shape[0] for shape in global_shapes.values()}
    if len(batches) != 1:
        raise MPSReverseContractError(
            "initial MPS batch dimensions differ across ranks"
        )
    if global_shapes[0][1] != 1 or global_shapes[n_wires - 1][3] != 1:
        raise MPSReverseContractError(
            "open-boundary MPS edge bond dimensions must be one"
        )
    for wire in range(n_wires - 1):
        if global_shapes[wire][3] != global_shapes[wire + 1][1]:
            raise MPSReverseContractError(
                f"initial MPS bond {wire} differs: "
                f"{global_shapes[wire][3]} != {global_shapes[wire + 1][1]}"
            )
    return local, batches.pop(), global_shapes


def mps_factorization_site_costs(
    bond_dimensions: Sequence[int],
) -> tuple[int, ...]:
    """Return per-site dense two-site factorization cost proxies."""
    bonds = tuple(int(value) for value in bond_dimensions)
    if len(bonds) < 2 or any(value < 1 for value in bonds):
        raise ValueError("bond_dimensions must contain positive boundary dimensions")
    n_wires = len(bonds) - 1
    split_costs = []
    for bond in range(max(0, n_wires - 1)):
        rows, columns = 2 * bonds[bond], 2 * bonds[bond + 2]
        split_costs.append(rows * columns * min(rows, columns))
    costs = []
    for wire in range(n_wires):
        left = split_costs[wire - 1] if wire > 0 else 0
        right = split_costs[wire] if wire < len(split_costs) else 0
        costs.append(max(1, left + right))
    return tuple(costs)


def cost_aware_mps_ownership(
    bond_dimensions: Sequence[int], world_size: int
) -> tuple[tuple[int, ...], ...]:
    """Partition sites using a dense two-site factorization cost proxy."""
    n_wires = len(tuple(bond_dimensions)) - 1
    if not 1 <= world_size <= n_wires:
        raise ValueError("cost-aware MPS ownership requires 1 <= world_size <= n_wires")
    weights = mps_factorization_site_costs(bond_dimensions)
    return validate_mps_ownership(
        _weighted_ownership(weights, world_size), n_wires, world_size
    )


def gate_aligned_cost_aware_mps_ownership(
    bond_dimensions: Sequence[int], world_size: int, *, alignment: int = 2
) -> tuple[tuple[int, ...], ...]:
    """Balance factorization work while keeping shard cuts gate-layer aligned."""
    n_wires = len(tuple(bond_dimensions)) - 1
    if not 1 <= world_size <= n_wires:
        raise ValueError("gate-aligned ownership requires 1 <= world_size <= n_wires")
    if alignment < 1:
        raise ValueError("alignment must be positive")
    if world_size == 1:
        return (tuple(range(n_wires)),)
    candidates = list(range(alignment, n_wires, alignment))
    if len(candidates) < world_size - 1:
        raise ValueError("not enough aligned cuts for the requested world size")
    weights = mps_factorization_site_costs(bond_dimensions)
    prefix = [0]
    for weight in weights:
        prefix.append(prefix[-1] + int(weight))
    boundaries = [0]
    first_candidate = 0
    for rank in range(1, world_size):
        remaining_cuts = world_size - rank - 1
        last_candidate = len(candidates) - remaining_cuts
        target = prefix[-1] * rank / world_size
        window = candidates[first_candidate:last_candidate]
        cut = min(window, key=lambda value: (abs(prefix[value] - target), value))
        boundaries.append(cut)
        first_candidate = candidates.index(cut) + 1
    boundaries.append(n_wires)
    ownership = tuple(
        tuple(range(boundaries[rank], boundaries[rank + 1]))
        for rank in range(world_size)
    )
    return validate_mps_ownership(ownership, n_wires, world_size)


def _owner(ownership: Sequence[Sequence[int]], wire: int) -> int:
    for rank, wires in enumerate(ownership):
        if int(wire) in wires:
            return rank
    raise KeyError(wire)


def _weighted_ownership(
    weights: Sequence[int], world_size: int
) -> tuple[tuple[int, ...], ...]:
    total = max(1, sum(int(value) for value in weights))
    boundaries = [0]
    cumulative = 0
    cursor = 0
    for rank in range(world_size - 1):
        target = total * (rank + 1) / world_size
        max_cursor = len(weights) - (world_size - rank - 1)
        while cursor < max_cursor and cumulative + weights[cursor] <= target:
            cumulative += int(weights[cursor])
            cursor += 1
        cursor = max(cursor, boundaries[-1] + 1)
        boundaries.append(cursor)
    boundaries.append(len(weights))
    return tuple(
        tuple(range(boundaries[rank], boundaries[rank + 1]))
        for rank in range(world_size)
    )


@dataclass(frozen=True)
class ReverseMPSInitialization:
    state: RankOwnedMPSState
    ownership: tuple[tuple[int, ...], ...]
    global_shapes: dict[int, tuple[int, ...]]
    batch_size: int
    device: torch.device
    dtype: torch.dtype


def initialize_reverse_mps_state(
    ir: CircuitIR,
    *,
    rank: int,
    world_size: int,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
    max_bond: int | None,
    cutoff: float,
    svd_driver: str | None,
    initial_bond_dimension: int,
    initial_mps_tensors: Mapping[int, torch.Tensor] | None,
    site_ownership: Sequence[Sequence[int]] | None,
) -> ReverseMPSInitialization:
    """Resolve ownership and construct the rank-local initial MPS."""

    ownership = (
        initial_mps_ownership(ir.n_wires, world_size)
        if site_ownership is None
        else validate_mps_ownership(site_ownership, ir.n_wires, world_size)
    )
    resolved_device = torch.device(device or "cpu")
    resolved_dtype = dtype or getattr(torch, ir.dtype)
    if initial_mps_tensors is not None:
        local_tensors, batch_size, global_shapes = normalize_rank_owned_initial_tensors(
            initial_mps_tensors,
            ownership=ownership,
            rank=rank,
            world=world_size,
            n_wires=ir.n_wires,
            device=resolved_device,
            dtype=resolved_dtype,
        )
        declared_batch = int(ir.metadata.get("batch_size", batch_size))
        if declared_batch != batch_size:
            raise MPSReverseContractError(
                f"circuit batch size {declared_batch} differs from initial MPS "
                f"{batch_size}"
            )
    else:
        batch_size = int(ir.metadata.get("batch_size", 1))
        global_shapes = {}
        for wire in range(ir.n_wires):
            left_dim = 1 if wire == 0 else initial_bond_dimension
            right_dim = 1 if wire == ir.n_wires - 1 else initial_bond_dimension
            global_shapes[wire] = (batch_size, left_dim, 2, right_dim)
        local_tensors = {}
        for wire in ownership[rank]:
            _, left_dim, _, right_dim = global_shapes[wire]
            tensor = torch.zeros(
                (batch_size, left_dim, 2, right_dim),
                dtype=resolved_dtype,
                device=resolved_device,
            )
            tensor[:, 0, 0, 0] = 1
            if initial_bond_dimension > 1:
                tensor[:, left_dim - 1, 1, right_dim - 1] = 1
                if wire == 0:
                    tensor.mul_(2**-0.5)
            local_tensors[wire] = tensor
    state = RankOwnedMPSState(
        ir.n_wires,
        batch_size,
        rank,
        world_size,
        MPSConfig(max_bond=max_bond, cutoff=cutoff, svd_driver=svd_driver),
        local_tensors,
        ownership,
    )
    return ReverseMPSInitialization(
        state,
        ownership,
        global_shapes,
        batch_size,
        resolved_device,
        resolved_dtype,
    )


# Private compatibility alias for historical internal callers.
_initial_ownership = initial_mps_ownership

__all__ = (
    "MPSPartition",
    "RankOwnedMPSState",
    "ReverseMPSInitialization",
    "cost_aware_mps_ownership",
    "gate_aligned_cost_aware_mps_ownership",
    "initial_mps_ownership",
    "initialize_reverse_mps_state",
    "mps_factorization_site_costs",
    "normalize_rank_owned_initial_tensors",
    "validate_mps_ownership",
)
