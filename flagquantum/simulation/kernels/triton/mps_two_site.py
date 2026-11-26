"""Fused CUDA kernel for an MPS two-site gate contraction."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from ._jit import autotune, jit
else:
    from triton import autotune, jit


class _TwoSiteContext(Protocol):
    @property
    def saved_tensors(self) -> tuple[torch.Tensor, ...]: ...

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


@autotune(
    configs=[
        triton.Config(
            {"block_rows": 16, "block_columns": 16},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"block_rows": 16, "block_columns": 16},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"block_rows": 32, "block_columns": 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"block_rows": 32, "block_columns": 32},
            num_warps=8,
            num_stages=2,
        ),
    ],
    key=["left_dim", "bond_dim", "right_dim", "batched_gate"],
)
@jit
def _two_site_forward_kernel(
    left_parts: tl.tensor,
    gate_parts: tl.tensor,
    right_parts: tl.tensor,
    output_parts: tl.tensor,
    left_dim: tl.constexpr,
    bond_dim: tl.constexpr,
    right_dim: tl.constexpr,
    batched_gate: tl.constexpr,
    block_rows: tl.constexpr,
    block_columns: tl.constexpr,
) -> None:
    batch = tl.program_id(2)
    rows = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    columns = tl.program_id(1) * block_columns + tl.arange(0, block_columns)
    row_mask = rows < left_dim * 2
    column_mask = columns < 2 * right_dim
    left_index = rows // 2
    output_left_physical = rows % 2
    output_right_physical = columns // right_dim
    right_index = columns % right_dim
    real = tl.zeros((block_rows, block_columns), dtype=tl.float32)
    imag = tl.zeros((block_rows, block_columns), dtype=tl.float32)
    for middle in range(bond_dim):
        for left_physical in range(2):
            left_offset = (
                (batch * left_dim + left_index) * 2 + left_physical
            ) * bond_dim + middle
            left_real = tl.load(left_parts + 2 * left_offset, mask=row_mask, other=0.0)[
                :, None
            ]
            left_imag = tl.load(
                left_parts + 2 * left_offset + 1, mask=row_mask, other=0.0
            )[:, None]
            for right_physical in range(2):
                right_offset = (
                    (batch * bond_dim + middle) * 2 + right_physical
                ) * right_dim + right_index
                right_real = tl.load(
                    right_parts + 2 * right_offset, mask=column_mask, other=0.0
                )[None, :]
                right_imag = tl.load(
                    right_parts + 2 * right_offset + 1,
                    mask=column_mask,
                    other=0.0,
                )[None, :]
                gate_row = (
                    2 * output_left_physical[:, None] + output_right_physical[None, :]
                )
                gate_column = 2 * left_physical + right_physical
                gate_offset = gate_row * 4 + gate_column
                if batched_gate:
                    gate_offset += batch * 16
                gate_real = tl.load(
                    gate_parts + 2 * gate_offset,
                    mask=row_mask[:, None] & column_mask[None, :],
                    other=0.0,
                )
                gate_imag = tl.load(
                    gate_parts + 2 * gate_offset + 1,
                    mask=row_mask[:, None] & column_mask[None, :],
                    other=0.0,
                )
                pair_real = left_real * right_real - left_imag * right_imag
                pair_imag = left_real * right_imag + left_imag * right_real
                real += pair_real * gate_real - pair_imag * gate_imag
                imag += pair_real * gate_imag + pair_imag * gate_real
    output_offset = (
        batch * left_dim * 4 * right_dim
        + rows[:, None] * (2 * right_dim)
        + columns[None, :]
    )
    mask = row_mask[:, None] & column_mask[None, :]
    tl.store(output_parts + 2 * output_offset, real, mask=mask)
    tl.store(output_parts + 2 * output_offset + 1, imag, mask=mask)


@autotune(
    configs=[
        triton.Config(
            {
                "block_rows": 16,
                "block_rank": 16,
                "block_bond": 16,
                "block_right": 16,
            },
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {
                "block_rows": 16,
                "block_rank": 32,
                "block_bond": 16,
                "block_right": 32,
            },
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {
                "block_rows": 32,
                "block_rank": 16,
                "block_bond": 16,
                "block_right": 32,
            },
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {
                "block_rows": 32,
                "block_rank": 32,
                "block_bond": 16,
                "block_right": 32,
            },
            num_warps=8,
            num_stages=2,
        ),
    ],
    key=["left_dim", "bond_dim", "right_dim", "rank", "batched_gate"],
)
@jit
def _two_site_range_kernel(
    left_parts: tl.tensor,
    gate_parts: tl.tensor,
    right_parts: tl.tensor,
    projection_parts: tl.tensor,
    output_parts: tl.tensor,
    left_dim: tl.constexpr,
    bond_dim: tl.constexpr,
    right_dim: tl.constexpr,
    rank: tl.constexpr,
    batched_gate: tl.constexpr,
    block_rows: tl.constexpr,
    block_rank: tl.constexpr,
    block_bond: tl.constexpr,
    block_right: tl.constexpr,
) -> None:
    batch = tl.program_id(2)
    rows = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    ranks = tl.program_id(1) * block_rank + tl.arange(0, block_rank)
    row_mask = rows < left_dim * 2
    rank_mask = ranks < rank
    left_index = rows // 2
    output_left_physical = rows % 2
    real = tl.zeros((block_rows, block_rank), dtype=tl.float32)
    imag = tl.zeros((block_rows, block_rank), dtype=tl.float32)
    bond_offsets = tl.arange(0, block_bond)
    right_offsets = tl.arange(0, block_right)
    for bond_block in range(tl.cdiv(bond_dim, block_bond)):
        middles = bond_block * block_bond + bond_offsets
        bond_mask = middles < bond_dim
        for right_physical in range(2):
            for output_right_physical in range(2):
                projected_real = tl.zeros((block_bond, block_rank), dtype=tl.float32)
                projected_imag = tl.zeros((block_bond, block_rank), dtype=tl.float32)
                for right_block in range(tl.cdiv(right_dim, block_right)):
                    right_indices = right_block * block_right + right_offsets
                    right_mask = right_indices < right_dim
                    right_offset = (
                        (batch * bond_dim + middles[:, None]) * 2 + right_physical
                    ) * right_dim + right_indices[None, :]
                    projection_offset = (
                        output_right_physical * right_dim + right_indices[:, None]
                    ) * rank + ranks[None, :]
                    rr = tl.load(
                        right_parts + 2 * right_offset,
                        mask=bond_mask[:, None] & right_mask[None, :],
                        other=0.0,
                    )
                    ri = tl.load(
                        right_parts + 2 * right_offset + 1,
                        mask=bond_mask[:, None] & right_mask[None, :],
                        other=0.0,
                    )
                    projection_mask = right_mask[:, None] & rank_mask[None, :]
                    pr = tl.load(
                        projection_parts + 2 * projection_offset,
                        mask=projection_mask,
                        other=0.0,
                    )
                    pi = tl.load(
                        projection_parts + 2 * projection_offset + 1,
                        mask=projection_mask,
                        other=0.0,
                    )
                    projected_real += tl.dot(rr, pr, allow_tf32=False) - tl.dot(
                        ri, pi, allow_tf32=False
                    )
                    projected_imag += tl.dot(rr, pi, allow_tf32=False) + tl.dot(
                        ri, pr, allow_tf32=False
                    )
                for left_physical in range(2):
                    left_offset = (
                        (batch * left_dim + left_index[:, None]) * 2 + left_physical
                    ) * bond_dim + middles[None, :]
                    left_mask = row_mask[:, None] & bond_mask[None, :]
                    lr = tl.load(
                        left_parts + 2 * left_offset,
                        mask=left_mask,
                        other=0.0,
                    )
                    li = tl.load(
                        left_parts + 2 * left_offset + 1,
                        mask=left_mask,
                        other=0.0,
                    )
                    contracted_real = tl.dot(
                        lr, projected_real, allow_tf32=False
                    ) - tl.dot(li, projected_imag, allow_tf32=False)
                    contracted_imag = tl.dot(
                        lr, projected_imag, allow_tf32=False
                    ) + tl.dot(li, projected_real, allow_tf32=False)
                    gate_row = 2 * output_left_physical + output_right_physical
                    gate_column = 2 * left_physical + right_physical
                    gate_offset = gate_row * 4 + gate_column
                    if batched_gate:
                        gate_offset += batch * 16
                    gr = tl.load(
                        gate_parts + 2 * gate_offset,
                        mask=row_mask,
                        other=0.0,
                    )[:, None]
                    gi = tl.load(
                        gate_parts + 2 * gate_offset + 1,
                        mask=row_mask,
                        other=0.0,
                    )[:, None]
                    real += contracted_real * gr - contracted_imag * gi
                    imag += contracted_real * gi + contracted_imag * gr
    offset = batch * left_dim * 2 * rank + rows[:, None] * rank + ranks[None, :]
    mask = row_mask[:, None] & rank_mask[None, :]
    tl.store(output_parts + 2 * offset, real, mask=mask)
    tl.store(output_parts + 2 * offset + 1, imag, mask=mask)


def _launch(
    left: torch.Tensor, gate: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    batch, left_dim, _, bond_dim = left.shape
    right_batch, right_bond, _, right_dim = right.shape
    if batch != right_batch or bond_dim != right_bond:
        raise ValueError("MPS two-site dimensions do not align")
    batched_gate = gate.ndim == 3
    if batched_gate and int(gate.shape[0]) != batch:
        raise ValueError("batched two-site gate does not match MPS batch")
    left = left.contiguous()
    gate = gate.contiguous()
    right = right.contiguous()
    output_parts = torch.empty(
        batch, left_dim * 2, 2 * right_dim, 2, dtype=torch.float32, device=left.device
    )

    def grid(meta: dict[str, int]) -> tuple[int, int, int]:
        return (
            triton.cdiv(left_dim * 2, meta["block_rows"]),
            triton.cdiv(2 * right_dim, meta["block_columns"]),
            batch,
        )

    _two_site_forward_kernel[grid](
        torch.view_as_real(left.resolve_conj().resolve_neg()),
        torch.view_as_real(gate.resolve_conj().resolve_neg()),
        torch.view_as_real(right.resolve_conj().resolve_neg()),
        output_parts,
        left_dim=left_dim,
        bond_dim=bond_dim,
        right_dim=right_dim,
        batched_gate=batched_gate,
    )
    return torch.view_as_complex(output_parts)


class _FusedMPSTwoSite(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: _TwoSiteContext,
        left: torch.Tensor,
        gate: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(left, gate, right)
        return _launch(left, gate, right)

    @staticmethod
    def backward(
        ctx: _TwoSiteContext, gradient: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        left, gate, right = ctx.saved_tensors
        batch, left_dim, _, _ = left.shape
        right_dim = right.shape[3]
        grad = gradient.reshape(batch, left_dim, 4, right_dim)
        theta = torch.einsum("blsm,bmtr->blstr", left, right).reshape(
            batch, left_dim, 4, right_dim
        )
        if gate.ndim == 3:
            theta_gradient = torch.einsum("bij,blir->bljr", torch.conj(gate), grad)
            gate_gradient = torch.einsum("blir,bljr->bij", grad, torch.conj(theta))
        else:
            theta_gradient = torch.einsum("ij,blir->bljr", torch.conj(gate), grad)
            gate_gradient = torch.einsum("blir,bljr->ij", grad, torch.conj(theta))
        theta_gradient = theta_gradient.reshape(batch, left_dim, 2, 2, right_dim)
        left_gradient = torch.einsum(
            "blstr,bmtr->blsm", theta_gradient, torch.conj(right)
        )
        right_gradient = torch.einsum(
            "blstr,blsm->bmtr", theta_gradient, torch.conj(left)
        )
        return left_gradient, gate_gradient, right_gradient


def fused_mps_two_site(
    left: torch.Tensor, gate: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    """Return the post-gate two-site matrix `[B, left*2, 2*right]`."""

    if (
        not left.is_cuda
        or left.dtype != torch.complex64
        or gate.dtype != torch.complex64
        or right.dtype != torch.complex64
    ):
        theta = torch.einsum("blsm,bmtr->blstr", left, right).reshape(
            left.shape[0], left.shape[1], 4, right.shape[3]
        )
        equation = "ij,bljr->blir" if gate.ndim == 2 else "bij,bljr->blir"
        return torch.einsum(equation, gate, theta).reshape(
            left.shape[0], left.shape[1] * 2, 2 * right.shape[3]
        )
    apply: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], object] = (
        _FusedMPSTwoSite.apply
    )
    result = apply(left, gate, right)
    if not isinstance(result, torch.Tensor):
        raise TypeError("MPS two-site autograd must return a tensor")
    return result


def fused_mps_range_projection(
    left: torch.Tensor,
    gate: torch.Tensor,
    right: torch.Tensor,
    projection: torch.Tensor,
) -> torch.Tensor:
    """Compute ``(left + gate + right) @ projection`` without forming the matrix."""

    batch, left_dim, _, bond_dim = left.shape
    right_dim = int(right.shape[-1])
    rank = int(projection.shape[-1])
    if projection.shape != (2 * right_dim, rank):
        raise ValueError("projection shape does not match the two-site columns")
    if not left.is_cuda or left.dtype != torch.complex64:
        theta = torch.einsum("blsm,bmtr->blstr", left, right).reshape(
            batch, left_dim, 4, right_dim
        )
        equation = "ij,bljr->blir" if gate.ndim == 2 else "bij,bljr->blir"
        matrix = torch.einsum(equation, gate, theta).reshape(
            batch, left_dim * 2, 2 * right_dim
        )
        return matrix @ projection
    output_parts = torch.empty(
        batch, left_dim * 2, rank, 2, dtype=torch.float32, device=left.device
    )

    def grid(meta: dict[str, int]) -> tuple[int, int, int]:
        return (
            triton.cdiv(left_dim * 2, meta["block_rows"]),
            triton.cdiv(rank, meta["block_rank"]),
            batch,
        )

    _two_site_range_kernel[grid](
        torch.view_as_real(left.contiguous()),
        torch.view_as_real(gate.contiguous()),
        torch.view_as_real(right.contiguous()),
        torch.view_as_real(projection.contiguous()),
        output_parts,
        left_dim=left_dim,
        bond_dim=bond_dim,
        right_dim=right_dim,
        rank=rank,
        batched_gate=gate.ndim == 3,
    )
    return torch.view_as_complex(output_parts)


__all__ = ["fused_mps_range_projection", "fused_mps_two_site"]
