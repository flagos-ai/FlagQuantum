"""Cross-rank state exchange, collection, and MPS ownership rebalancing."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from ....core.ir import Instruction
from ....simulation.mps.rank_local import (
    apply_two_mps_tensors_with_info as _apply_two_mps_tensors_with_info,
)
from ....simulation.mps.rank_local import (
    tensor_nbytes as _tensor_nbytes,
)
from .state import RankOwnedMPSState, _owner, _weighted_ownership
from .transport import (
    _recv_tensor_p2p,
    _send_tensor_p2p,
)


def apply_rank_boundary_gate(
    state: RankOwnedMPSState,
    instruction: Instruction,
    matrix: torch.Tensor,
) -> tuple[int, int, dict[str, Any] | None]:
    """Execute one adjacent two-site gate whose tensors have different owners."""
    first, second = (int(wire) for wire in instruction.wires)
    left_wire, right_wire = sorted((first, second))
    left_rank = state.owner(left_wire)
    right_rank = state.owner(right_wire)
    reference = next(iter(state.local_tensors.values()))
    sent_bytes = 0
    record = None
    if state.rank == right_rank:
        right = state.local_tensors[right_wire]
        sent_bytes += _tensor_nbytes(right)
        _send_tensor_p2p(right, dst=left_rank)
        updated = _recv_tensor_p2p(src=left_rank, reference=right)
        state.local_tensors[right_wire] = updated
        sent_bytes += _tensor_nbytes(updated)
    elif state.rank == left_rank:
        left = state.local_tensors[left_wire]
        right = _recv_tensor_p2p(src=right_rank, reference=reference)
        received_bytes = _tensor_nbytes(right)
        left, right, split_info = _apply_two_mps_tensors_with_info(
            left,
            right,
            matrix,
            state.config,
            reverse=first > second,
        )
        state.local_tensors[left_wire] = left
        record = {
            **dict(split_info),
            "bond": left_wire,
            "source": "rank_boundary_gate",
        }
        _send_tensor_p2p(right, dst=right_rank)
        sent_bytes += received_bytes + _tensor_nbytes(right)
    messages = 2 if state.rank in {left_rank, right_rank} else 0
    return messages, sent_bytes, record


def _device_footprint(state: RankOwnedMPSState, qubit: int) -> torch.Tensor:
    """One site's byte footprint as a device scalar, on the state's device.

    The footprint is `numel * element_size`, which the host already knows from
    the tensor's shape and dtype, so stating it reads nothing back from the
    device. Writing that integer into a device vector would instead lift a fresh
    device scalar per site, which a profiler reports as one host-to-device copy
    per site inside whatever region asked for the footprint. Multiplying a
    device scalar by the same integer passes it as a kernel argument, so the
    vector assembles on the device and the collective below is the only
    interchange this accounting performs.
    """

    reference = next(iter(state.local_tensors.values()))
    tensor = state.local_tensors.get(qubit)
    if tensor is None:
        return torch.zeros((), dtype=torch.int64, device=reference.device)
    one = torch.ones((), dtype=torch.int64, device=reference.device)
    return one * _tensor_nbytes(tensor)


def global_mps_tensor_bytes(state: RankOwnedMPSState) -> tuple[int, ...]:
    """Collect the current per-site tensor footprint on every rank."""
    sizes = torch.stack(
        [_device_footprint(state, qubit) for qubit in range(state.n_qubits)]
    )
    dist.all_reduce(sizes)
    return tuple(int(value) for value in sizes.tolist())


def global_mps_bond_dimensions(state: RankOwnedMPSState) -> tuple[int, ...]:
    """Collect the current internal bond dimensions on every rank."""
    reference = next(iter(state.local_tensors.values()))
    entries: list[torch.Tensor] = []
    for qubit in range(max(0, state.n_qubits - 1)):
        tensor = state.local_tensors.get(qubit)
        if tensor is None:
            entries.append(torch.zeros((), dtype=torch.int64, device=reference.device))
        else:
            one = torch.ones((), dtype=torch.int64, device=reference.device)
            entries.append(one * int(tensor.shape[3]))
    dimensions = torch.stack(entries)
    dist.all_reduce(dimensions)
    return tuple(int(value) for value in dimensions.tolist())


def migrate_mps_partitions(
    state: RankOwnedMPSState, ownership: tuple[tuple[int, ...], ...]
) -> tuple[int, int]:
    """Move site tensors to a new validated ownership plan."""
    old = state.ownership
    reference = next(iter(state.local_tensors.values()))
    messages = byte_count = 0
    for wire in range(state.n_qubits):
        source = _owner(old, wire)
        target = _owner(ownership, wire)
        if source == target:
            continue
        if state.rank == source:
            tensor = state.local_tensors.pop(wire)
            byte_count += _tensor_nbytes(tensor)
            messages += 1
            _send_tensor_p2p(tensor, dst=target)
        elif state.rank == target:
            tensor = _recv_tensor_p2p(src=source, reference=reference)
            state.local_tensors[wire] = tensor
            byte_count += _tensor_nbytes(tensor)
            messages += 1
    state.ownership = ownership
    return messages, byte_count


def rebalance_mps_if_needed(
    state: RankOwnedMPSState, threshold: float
) -> tuple[bool, int, int]:
    """Rebalance rank ownership when the measured byte ratio crosses a threshold."""
    if threshold == float("inf"):
        # No ratio can cross an infinite threshold, so the footprint collective
        # below could only ever return "do not migrate". The compiled site
        # kernel path requires exactly this threshold, because its buckets are
        # sized for the plan and must not move between ranks mid-layer; paying
        # for a collective per two-site gate to re-confirm that would be work
        # the configuration cannot act on.
        return False, 0, 0
    sizes = global_mps_tensor_bytes(state)
    rank_bytes = [sum(sizes[wire] for wire in wires) for wires in state.ownership]
    positive = [value for value in rank_bytes if value > 0]
    if not positive or max(positive) / min(positive) < threshold:
        return False, 0, 0
    ownership = _weighted_ownership(sizes, state.world_size)
    if ownership == state.ownership:
        return False, 0, 0
    messages, byte_count = migrate_mps_partitions(state, ownership)
    return True, messages, byte_count


__all__ = (
    "apply_rank_boundary_gate",
    "global_mps_bond_dimensions",
    "global_mps_tensor_bytes",
    "migrate_mps_partitions",
    "rebalance_mps_if_needed",
)
