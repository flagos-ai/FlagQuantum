"""Workspace-aware planning for owner-local MPS factorizations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping

import torch
from torch.profiler import record_function

from ....core.ir import Instruction
from .communication import (
    _apply_two_mps_tensors_with_info,
    _instruction_matrix_for_mps,
)
from .records import MPSReverseContractError
from .state import RankOwnedMPSState


class MPSFactorizationMemoryError(RuntimeError):
    """A factorization cannot satisfy its declared device-memory contract."""


@dataclass(frozen=True)
class FactorizationMemorySnapshot:
    free_bytes: int
    total_bytes: int
    allocated_bytes: int
    reserved_bytes: int

    def __post_init__(self) -> None:
        if (
            min(
                self.free_bytes,
                self.total_bytes,
                self.allocated_bytes,
                self.reserved_bytes,
            )
            < 0
        ):
            raise ValueError(
                "factorization memory snapshot values must be non-negative"
            )


@dataclass(frozen=True)
class FactorizationWorkspacePolicy:
    """Fail-closed runtime budget, independent from the compile-cache policy."""

    memory_budget_bytes: int | None = None
    minimum_headroom_bytes: int = 2 * (1 << 30)
    nccl_reserve_bytes: int = 256 * (1 << 20)
    allocator_safety_margin_bytes: int = 512 * (1 << 20)
    solver_workspace_pair_multiplier: float = 4.0
    autograd_save_pair_multiplier: float = 2.0
    maximum_chunk_size: int = 8
    high_bond_threshold: int = 1024
    high_bond_maximum_chunk_size: int = 1
    svd_driver: str | None = "gesvd"
    policy_id: str = "mps_factorization_workspace_v1"

    def __post_init__(self) -> None:
        byte_fields = (
            self.minimum_headroom_bytes,
            self.nccl_reserve_bytes,
            self.allocator_safety_margin_bytes,
        )
        if any(value < 0 for value in byte_fields):
            raise ValueError("factorization memory reserves must be non-negative")
        if self.memory_budget_bytes is not None and self.memory_budget_bytes <= 0:
            raise ValueError("factorization memory budget must be positive")
        if self.solver_workspace_pair_multiplier < 0:
            raise ValueError("solver workspace multiplier must be non-negative")
        if self.autograd_save_pair_multiplier < 0:
            raise ValueError("autograd save multiplier must be non-negative")
        if (
            min(
                self.maximum_chunk_size,
                self.high_bond_threshold,
                self.high_bond_maximum_chunk_size,
            )
            <= 0
        ):
            raise ValueError("factorization chunk and bond limits must be positive")
        if self.svd_driver not in {None, "gesvd", "gesvdj", "gesvda"}:
            raise ValueError("unsupported CUDA SVD driver")


@dataclass(frozen=True)
class FactorizationWorkingSet:
    input_bytes_per_item: int
    pair_output_bytes_per_item: int
    split_output_bytes_per_item: int
    singular_value_bytes_per_item: int
    solver_workspace_bytes_per_item: int
    autograd_save_bytes_per_item: int
    fixed_reserve_bytes: int

    @property
    def item_bytes(self) -> int:
        return (
            self.input_bytes_per_item
            + self.pair_output_bytes_per_item
            + self.split_output_bytes_per_item
            + self.singular_value_bytes_per_item
            + self.solver_workspace_bytes_per_item
            + self.autograd_save_bytes_per_item
        )

    def bytes_for_chunk(self, chunk_size: int) -> int:
        return self.fixed_reserve_bytes + self.item_bytes * int(chunk_size)

    def as_dict(self) -> dict[str, int]:
        return {**asdict(self), "item_bytes": self.item_bytes}


@dataclass(frozen=True)
class FactorizationMicrobatchDecision:
    requested_chunk_size: int
    selected_chunk_size: int
    shape: tuple[tuple[int, ...], ...]
    dtype: str
    device: str
    requires_grad: bool
    maximum_by_policy: int
    maximum_by_memory: int
    available_working_bytes: int
    requested_working_set_bytes: int
    selected_working_set_bytes: int
    minimum_headroom_bytes: int
    downshifted: bool
    policy_id: str
    working_set: FactorizationWorkingSet
    memory_snapshot: FactorizationMemorySnapshot

    def as_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "working_set": self.working_set.as_dict(),
            "memory_snapshot": asdict(self.memory_snapshot),
        }


MemoryProvider = Callable[[torch.device], FactorizationMemorySnapshot]


def cuda_factorization_memory_snapshot(
    device: torch.device,
) -> FactorizationMemorySnapshot:
    """Measure allocator and device headroom without retaining CUDA tensors."""

    if device.type != "cuda":
        maximum = (1 << 63) - 1
        return FactorizationMemorySnapshot(
            free_bytes=maximum,
            total_bytes=maximum,
            allocated_bytes=0,
            reserved_bytes=0,
        )
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return FactorizationMemorySnapshot(
        free_bytes=int(free_bytes),
        total_bytes=int(total_bytes),
        allocated_bytes=int(torch.cuda.memory_allocated(device)),
        reserved_bytes=int(torch.cuda.memory_reserved(device)),
    )


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def estimate_rxx_factorization_working_set(
    left: torch.Tensor,
    right: torch.Tensor,
    matrix: torch.Tensor,
    *,
    policy: FactorizationWorkspacePolicy,
    requires_grad: bool,
) -> FactorizationWorkingSet:
    """Conservatively account for one contraction and QR/SVD split."""

    if left.ndim != 4 or right.ndim != 4:
        raise ValueError("RXX factorization samples must be rank-4 MPS tensors")
    if int(left.shape[0]) != int(right.shape[0]):
        raise ValueError("RXX factorization samples must share a batch size")
    if int(left.shape[-1]) != int(right.shape[1]):
        raise ValueError("RXX factorization samples must share their middle bond")
    batch = int(left.shape[0])
    rows = int(left.shape[1]) * int(left.shape[2])
    columns = int(right.shape[2]) * int(right.shape[3])
    retained = min(rows, columns)
    element_bytes = left.element_size()
    real_bytes = element_bytes // 2 if left.is_complex() else element_bytes
    pair_bytes = batch * rows * columns * element_bytes
    split_bytes = batch * (rows * retained + retained * columns) * element_bytes
    singular_bytes = batch * retained * real_bytes
    autograd_bytes = (
        int(pair_bytes * policy.autograd_save_pair_multiplier) if requires_grad else 0
    )
    return FactorizationWorkingSet(
        input_bytes_per_item=_tensor_bytes(left)
        + _tensor_bytes(right)
        + _tensor_bytes(matrix),
        pair_output_bytes_per_item=pair_bytes,
        split_output_bytes_per_item=split_bytes,
        singular_value_bytes_per_item=singular_bytes,
        solver_workspace_bytes_per_item=int(
            pair_bytes * policy.solver_workspace_pair_multiplier
        ),
        autograd_save_bytes_per_item=autograd_bytes,
        fixed_reserve_bytes=(
            policy.nccl_reserve_bytes + policy.allocator_safety_margin_bytes
        ),
    )


def plan_rxx_factorization_microbatch(
    left: torch.Tensor,
    right: torch.Tensor,
    matrix: torch.Tensor,
    *,
    requested_chunk_size: int,
    policy: FactorizationWorkspacePolicy | None = None,
    memory_provider: MemoryProvider = cuda_factorization_memory_snapshot,
) -> FactorizationMicrobatchDecision:
    """Select a deterministic chunk that preserves emergency headroom."""

    if requested_chunk_size <= 0:
        raise ValueError("requested factorization chunk size must be positive")
    resolved = policy or FactorizationWorkspacePolicy()
    device = left.device
    if right.device != device or matrix.device != device:
        raise ValueError("factorization samples must share one device")
    snapshot = memory_provider(device)
    budget_remaining = (
        snapshot.free_bytes
        if resolved.memory_budget_bytes is None
        else max(0, resolved.memory_budget_bytes - snapshot.allocated_bytes)
    )
    available = max(
        0,
        min(snapshot.free_bytes, budget_remaining) - resolved.minimum_headroom_bytes,
    )
    requires_grad = bool(
        torch.is_grad_enabled()
        and any(item.requires_grad for item in (left, right, matrix))
    )
    working_set = estimate_rxx_factorization_working_set(
        left,
        right,
        matrix,
        policy=resolved,
        requires_grad=requires_grad,
    )
    maximum_by_memory = max(
        0,
        (available - working_set.fixed_reserve_bytes) // working_set.item_bytes,
    )
    maximum_by_policy = resolved.maximum_chunk_size
    maximum_bond = max(
        int(left.shape[1]),
        int(left.shape[-1]),
        int(right.shape[1]),
        int(right.shape[-1]),
    )
    if maximum_bond >= resolved.high_bond_threshold:
        maximum_by_policy = min(
            maximum_by_policy, resolved.high_bond_maximum_chunk_size
        )
    selected = min(requested_chunk_size, maximum_by_policy, maximum_by_memory)
    if selected < 1:
        required = working_set.bytes_for_chunk(1)
        raise MPSFactorizationMemoryError(
            "RXX factorization memory plan rejected before allocation: "
            f"requested_bytes={required}, available_working_bytes={available}, "
            f"headroom_bytes={resolved.minimum_headroom_bytes}, "
            f"shape={(tuple(left.shape), tuple(right.shape), tuple(matrix.shape))}, "
            f"dtype={left.dtype}, device={device}, policy={resolved.policy_id}"
        )
    return FactorizationMicrobatchDecision(
        requested_chunk_size=requested_chunk_size,
        selected_chunk_size=int(selected),
        shape=(tuple(left.shape), tuple(right.shape), tuple(matrix.shape)),
        dtype=str(left.dtype),
        device=str(device),
        requires_grad=requires_grad,
        maximum_by_policy=int(maximum_by_policy),
        maximum_by_memory=int(maximum_by_memory),
        available_working_bytes=int(available),
        requested_working_set_bytes=working_set.bytes_for_chunk(requested_chunk_size),
        selected_working_set_bytes=working_set.bytes_for_chunk(selected),
        minimum_headroom_bytes=resolved.minimum_headroom_bytes,
        downshifted=selected < requested_chunk_size,
        policy_id=resolved.policy_id,
        working_set=working_set,
        memory_snapshot=snapshot,
    )


def apply_mps_pair_forward(
    left: torch.Tensor,
    right: torch.Tensor,
    instruction: Instruction,
    state: RankOwnedMPSState,
) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]:
    """Apply and split one owner-local adjacent two-site instruction."""
    with record_function("flagquantum::mps::two_site_split"):
        matrix = _instruction_matrix_for_mps(
            instruction, bsz=state.bsz, device=left.device, dtype=left.dtype
        )
        output_left, output_right, split_info = _apply_two_mps_tensors_with_info(
            left,
            right,
            matrix,
            state.config,
            reverse=int(instruction.wires[0]) > int(instruction.wires[1]),
        )
    return output_left, output_right, dict(split_info)


def mps_qr_forward(
    left: torch.Tensor, right: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the thin QR update with a vectorized tiny-column fast path."""
    bsz, left_dim, physical, right_dim = left.shape
    matrix = left.reshape(bsz, left_dim * physical, right_dim)
    if right_dim <= 2:
        first = matrix[..., 0]
        r00 = torch.linalg.vector_norm(first, dim=-1, keepdim=True)
        tiny = torch.finfo(first.real.dtype).tiny
        threshold = torch.finfo(first.real.dtype).eps * max(1, matrix.shape[-2])
        if torch.any(r00 <= threshold):
            raise MPSReverseContractError(
                "thin QR pullback is rank deficient in its first column"
            )
        q0 = first / r00.clamp_min(tiny)
        columns = [q0]
        rows = [r00]
        if right_dim == 2:
            second = matrix[..., 1]
            r01 = torch.sum(q0.conj() * second, dim=-1, keepdim=True)
            residual = second - q0 * r01
            r11 = torch.linalg.vector_norm(residual, dim=-1, keepdim=True)
            scale = torch.maximum(
                r00, torch.linalg.vector_norm(second, dim=-1, keepdim=True)
            )
            if torch.any(r11 <= threshold * scale.clamp_min(1.0)):
                raise MPSReverseContractError(
                    "thin QR pullback is rank deficient or near-degenerate"
                )
            q1 = residual / r11.clamp_min(tiny)
            columns.append(q1)
            rows = [
                torch.cat((r00, r01), dim=-1),
                torch.cat((torch.zeros_like(r11), r11), dim=-1),
            ]
        q = torch.stack(columns, dim=-1)
        transfer = (
            rows[0].unsqueeze(-2) if right_dim == 1 else torch.stack(rows, dim=-2)
        )
    else:
        q, transfer = torch.linalg.qr(matrix, mode="reduced")
    return q.reshape(bsz, left_dim, physical, q.shape[-1]), torch.einsum(
        "bij,bjsk->bisk", transfer, right
    )


def apply_mps_qr_forward(
    left: torch.Tensor, right: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one profiled owner-local QR canonicalization step."""
    with record_function("flagquantum::mps::qr"):
        return mps_qr_forward(left, right)


# Historical explicit-import compatibility. The canonical public name intentionally
# omits the development-era implementation suffix.
mps_qr_forward_impl = mps_qr_forward


__all__ = [
    "FactorizationMemorySnapshot",
    "FactorizationMicrobatchDecision",
    "FactorizationWorkingSet",
    "FactorizationWorkspacePolicy",
    "MPSFactorizationMemoryError",
    "cuda_factorization_memory_snapshot",
    "estimate_rxx_factorization_working_set",
    "plan_rxx_factorization_microbatch",
    "apply_mps_pair_forward",
    "apply_mps_qr_forward",
    "mps_qr_forward",
]
