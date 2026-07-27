"""Cross-rank canonicalization for rank-owned PyTorch MPS states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from .communication import _recv_tensor_p2p, _send_tensor_p2p, _tensor_nbytes


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


def _deterministic_qr(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """QR with a non-negative real diagonal convention for rank-stable gauges."""

    q, r = torch.linalg.qr(matrix, mode="reduced")
    diagonal = torch.diagonal(r, dim1=-2, dim2=-1)
    magnitude = diagonal.abs()
    phase = torch.where(magnitude > 0, diagonal / magnitude, torch.ones_like(diagonal))
    q = q * phase.unsqueeze(-2)
    r = phase.conj().unsqueeze(-1) * r
    return q, r


def _left_step(state: Any, wire: int) -> tuple[int, int, int]:
    left_owner, right_owner = _owner(state, wire), _owner(state, wire + 1)
    messages = communicated = temporary = 0
    if state.rank == left_owner:
        tensor = state.local_tensors[wire]
        bsz, left_dim, physical_dim, right_dim = tensor.shape
        matrix = tensor.reshape(bsz, left_dim * physical_dim, right_dim)
        q, transfer = _deterministic_qr(matrix)
        state.local_tensors[wire] = q.reshape(bsz, left_dim, physical_dim, q.shape[-1])
        temporary = _tensor_nbytes(q) + _tensor_nbytes(transfer)
        if left_owner == right_owner:
            right = state.local_tensors[wire + 1]
            updated = torch.einsum("bij,bjsk->bisk", transfer, right)
            state.local_tensors[wire + 1] = updated
            temporary += _tensor_nbytes(updated)
        else:
            _send_tensor_p2p(transfer.unsqueeze(-1), dst=right_owner)
            messages = 1
            communicated = _tensor_nbytes(transfer)
    elif state.rank == right_owner:
        right = state.local_tensors[wire + 1]
        transfer = _recv_tensor_p2p(src=left_owner, reference=right).squeeze(-1)
        updated = torch.einsum("bij,bjsk->bisk", transfer, right)
        state.local_tensors[wire + 1] = updated
        messages = 1
        communicated = _tensor_nbytes(transfer)
        temporary = _tensor_nbytes(transfer) + _tensor_nbytes(updated)
    return messages, communicated, temporary


def _right_step(state: Any, wire: int) -> tuple[int, int, int]:
    left_owner, right_owner = _owner(state, wire - 1), _owner(state, wire)
    messages = communicated = temporary = 0
    if state.rank == right_owner:
        tensor = state.local_tensors[wire]
        bsz, left_dim, physical_dim, right_dim = tensor.shape
        matrix_t = tensor.reshape(bsz, left_dim, physical_dim * right_dim).transpose(
            -2, -1
        )
        q, r = _deterministic_qr(matrix_t)
        right = q.transpose(-2, -1)
        transfer = r.transpose(-2, -1)
        state.local_tensors[wire] = right.reshape(
            bsz, right.shape[1], physical_dim, right_dim
        )
        temporary = _tensor_nbytes(q) + _tensor_nbytes(r)
        if left_owner == right_owner:
            left = state.local_tensors[wire - 1]
            updated = torch.einsum("blpa,bac->blpc", left, transfer)
            state.local_tensors[wire - 1] = updated
            temporary += _tensor_nbytes(updated)
        else:
            _send_tensor_p2p(transfer.unsqueeze(-1), dst=left_owner)
            messages = 1
            communicated = _tensor_nbytes(transfer)
    elif state.rank == left_owner:
        left = state.local_tensors[wire - 1]
        transfer = _recv_tensor_p2p(src=right_owner, reference=left).squeeze(-1)
        updated = torch.einsum("blpa,bac->blpc", left, transfer)
        state.local_tensors[wire - 1] = updated
        messages = 1
        communicated = _tensor_nbytes(transfer)
        temporary = _tensor_nbytes(transfer) + _tensor_nbytes(updated)
    return messages, communicated, temporary


def _local_mixed_residual(state: Any, center: int) -> float:
    residual = torch.zeros((), dtype=torch.float64, device=_reference(state).device)
    for wire, tensor in state.local_tensors.items():
        if wire == center:
            continue
        for batch_tensor in tensor:
            if wire < center:
                matrix = batch_tensor.reshape(-1, batch_tensor.shape[-1])
                gram = matrix.mH @ matrix
            else:
                matrix = batch_tensor.reshape(batch_tensor.shape[0], -1)
                gram = matrix @ matrix.mH
            eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
            residual = torch.maximum(
                residual, torch.max(torch.abs(gram - eye)).double()
            )
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
        center_tensor = state.local_tensors[target]
        norms = torch.sum(
            torch.abs(center_tensor.reshape(center_tensor.shape[0], -1)) ** 2, dim=1
        )
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
    "_deterministic_qr",
    "canonicalize_rank_owned_mps",
]
