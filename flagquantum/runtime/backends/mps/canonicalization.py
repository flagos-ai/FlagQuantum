"""Cross-rank canonicalization for rank-owned PyTorch MPS states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from ....simulation.mps.rank_local import tensor_nbytes
from ....simulation.mps_canonicalization import (
    absorb_left_canonical_transfer,
    absorb_right_canonical_transfer,
    factor_left_canonical_site,
    factor_right_canonical_site,
    local_mixed_canonical_residual,
    mps_center_norms,
)
from .communication import _recv_tensor_p2p, _send_tensor_p2p


@dataclass(frozen=True)
class MPSCanonicalizationMetrics:
    center: int
    residual: float
    state_norms: tuple[float, ...]
    messages_by_rank: tuple[int, ...]
    bytes_by_rank: tuple[int, ...]
    temporary_bytes_by_rank: tuple[int, ...]


def _owner(state: Any, wire: int) -> int:
    return int(state.owner(wire))


def _reference(state: Any) -> torch.Tensor:
    return next(iter(state.local_tensors.values()))


def _left_step(state: Any, wire: int) -> tuple[int, int, int]:
    left_owner, right_owner = _owner(state, wire), _owner(state, wire + 1)
    messages = communicated = temporary = 0
    if state.rank == left_owner:
        tensor = state.local_tensors[wire]
        canonical, transfer = factor_left_canonical_site(tensor)
        state.local_tensors[wire] = canonical
        temporary = tensor_nbytes(canonical) + tensor_nbytes(transfer)
        if left_owner == right_owner:
            right = state.local_tensors[wire + 1]
            updated = absorb_left_canonical_transfer(transfer, right)
            state.local_tensors[wire + 1] = updated
            temporary += tensor_nbytes(updated)
        else:
            _send_tensor_p2p(transfer.unsqueeze(-1), dst=right_owner)
            messages = 1
            communicated = tensor_nbytes(transfer)
    elif state.rank == right_owner:
        right = state.local_tensors[wire + 1]
        transfer = _recv_tensor_p2p(src=left_owner, reference=right).squeeze(-1)
        updated = absorb_left_canonical_transfer(transfer, right)
        state.local_tensors[wire + 1] = updated
        messages = 1
        communicated = tensor_nbytes(transfer)
        temporary = tensor_nbytes(transfer) + tensor_nbytes(updated)
    return messages, communicated, temporary


def _right_step(state: Any, wire: int) -> tuple[int, int, int]:
    left_owner, right_owner = _owner(state, wire - 1), _owner(state, wire)
    messages = communicated = temporary = 0
    if state.rank == right_owner:
        tensor = state.local_tensors[wire]
        transfer, canonical = factor_right_canonical_site(tensor)
        state.local_tensors[wire] = canonical
        temporary = tensor_nbytes(canonical) + tensor_nbytes(transfer)
        if left_owner == right_owner:
            left = state.local_tensors[wire - 1]
            updated = absorb_right_canonical_transfer(left, transfer)
            state.local_tensors[wire - 1] = updated
            temporary += tensor_nbytes(updated)
        else:
            _send_tensor_p2p(transfer.unsqueeze(-1), dst=left_owner)
            messages = 1
            communicated = tensor_nbytes(transfer)
    elif state.rank == left_owner:
        left = state.local_tensors[wire - 1]
        transfer = _recv_tensor_p2p(src=right_owner, reference=left).squeeze(-1)
        updated = absorb_right_canonical_transfer(left, transfer)
        state.local_tensors[wire - 1] = updated
        messages = 1
        communicated = tensor_nbytes(transfer)
        temporary = tensor_nbytes(transfer) + tensor_nbytes(updated)
    return messages, communicated, temporary


def _local_mixed_residual(state: Any, center: int) -> float:
    residual = local_mixed_canonical_residual(state.local_tensors, center)
    dist.all_reduce(residual, op=dist.ReduceOp.MAX)
    return float(residual.cpu())


def canonicalize_rank_owned_mps(
    state: Any, *, center: int | None = None
) -> MPSCanonicalizationMetrics:
    """Move a sharded MPS center using local QR and boundary transfer matrices."""

    target = state.n_wires - 1 if center is None else int(center)
    if not 0 <= target < state.n_wires:
        raise ValueError(f"canonical center must be in [0, {state.n_wires - 1}]")
    local_messages = local_bytes = local_temporary = 0
    for wire in range(state.n_wires - 1):
        messages, byte_count, temporary = _left_step(state, wire)
        local_messages += messages
        local_bytes += byte_count
        local_temporary = max(local_temporary, temporary)
    for wire in range(state.n_wires - 1, target, -1):
        messages, byte_count, temporary = _right_step(state, wire)
        local_messages += messages
        local_bytes += byte_count
        local_temporary = max(local_temporary, temporary)

    # Account for the metrics all-gather, norm broadcast and residual all-reduce.
    local_messages += 3
    local_bytes += 3 * torch.empty((), dtype=torch.int64).element_size()
    local_bytes += state.bsz * torch.empty((), dtype=torch.float64).element_size()
    local_bytes += torch.empty((), dtype=torch.float64).element_size()
    device = _reference(state).device
    counters = torch.tensor(
        [local_messages, local_bytes, local_temporary], dtype=torch.int64, device=device
    )
    gathered = [torch.zeros_like(counters) for _ in range(state.world_size)]
    dist.all_gather(gathered, counters)
    center_owner = _owner(state, target)
    if state.rank == center_owner:
        norms = mps_center_norms(state.local_tensors[target])
    else:
        norms = torch.empty(state.bsz, dtype=torch.float64, device=device)
    norms = norms.to(dtype=torch.float64)
    dist.broadcast(norms, src=center_owner)
    return MPSCanonicalizationMetrics(
        center=target,
        residual=_local_mixed_residual(state, target),
        state_norms=tuple(float(value) for value in norms.cpu()),
        messages_by_rank=tuple(int(value[0]) for value in gathered),
        bytes_by_rank=tuple(int(value[1]) for value in gathered),
        temporary_bytes_by_rank=tuple(int(value[2]) for value in gathered),
    )


__all__ = [
    "MPSCanonicalizationMetrics",
    "canonicalize_rank_owned_mps",
]
