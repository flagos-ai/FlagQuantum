"""Autograd-enabled fused complex batched matrix multiplication for CUDA."""

from __future__ import annotations

from math import prod
from typing import Callable, Protocol

import torch
import triton
import triton.language as tl


class _BMMContext(Protocol):
    @property
    def saved_tensors(self) -> tuple[torch.Tensor, ...]: ...

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


class _LayoutBMMContext(_BMMContext, Protocol):
    left_permutation: tuple[int, ...]
    right_permutation: tuple[int, ...]
    shapes: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]


@triton.jit
def _flattened_offset(
    index: tl.tensor,
    packed_layout: tl.constexpr,
) -> int | tl.tensor:
    offset = 0
    remaining = index
    if len(packed_layout) > 0:
        packed_axis: tl.constexpr = packed_layout[0]
        offset += (remaining % (packed_axis & 0xFFFFFFFF)) * (packed_axis >> 32)
        remaining = remaining // (packed_axis & 0xFFFFFFFF)
    if len(packed_layout) > 1:
        packed_axis_1: tl.constexpr = packed_layout[1]
        offset += (remaining % (packed_axis_1 & 0xFFFFFFFF)) * (packed_axis_1 >> 32)
        remaining = remaining // (packed_axis_1 & 0xFFFFFFFF)
    if len(packed_layout) > 2:
        packed_axis_2: tl.constexpr = packed_layout[2]
        offset += (remaining % (packed_axis_2 & 0xFFFFFFFF)) * (packed_axis_2 >> 32)
        remaining = remaining // (packed_axis_2 & 0xFFFFFFFF)
    if len(packed_layout) > 3:
        packed_axis_3: tl.constexpr = packed_layout[3]
        offset += (remaining % (packed_axis_3 & 0xFFFFFFFF)) * (packed_axis_3 >> 32)
        remaining = remaining // (packed_axis_3 & 0xFFFFFFFF)
    if len(packed_layout) > 4:
        packed_axis_4: tl.constexpr = packed_layout[4]
        offset += (remaining % (packed_axis_4 & 0xFFFFFFFF)) * (packed_axis_4 >> 32)
        remaining = remaining // (packed_axis_4 & 0xFFFFFFFF)
    if len(packed_layout) > 5:
        packed_axis_5: tl.constexpr = packed_layout[5]
        offset += (remaining % (packed_axis_5 & 0xFFFFFFFF)) * (packed_axis_5 >> 32)
        remaining = remaining // (packed_axis_5 & 0xFFFFFFFF)
    if len(packed_layout) > 6:
        packed_axis_6: tl.constexpr = packed_layout[6]
        offset += (remaining % (packed_axis_6 & 0xFFFFFFFF)) * (packed_axis_6 >> 32)
        remaining = remaining // (packed_axis_6 & 0xFFFFFFFF)
    if len(packed_layout) > 7:
        packed_axis_7: tl.constexpr = packed_layout[7]
        offset += (remaining % (packed_axis_7 & 0xFFFFFFFF)) * (packed_axis_7 >> 32)
    return offset


@triton.jit
def _complex_layout_bmm_kernel(
    left_parts: tl.tensor,
    right_parts: tl.tensor,
    output_parts: tl.tensor,
    m_size: tl.constexpr,
    n_size: tl.constexpr,
    k_size: tl.constexpr,
    left_batch_strides: tl.constexpr,
    left_row_strides: tl.constexpr,
    left_reduction_strides: tl.constexpr,
    right_batch_strides: tl.constexpr,
    right_reduction_strides: tl.constexpr,
    right_column_strides: tl.constexpr,
    conjugate_left: tl.constexpr,
    conjugate_right: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
) -> None:
    batch = tl.program_id(2)
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    columns = tl.program_id(1) * block_n + tl.arange(0, block_n)
    reduction = tl.arange(0, block_k)
    left_batch = _flattened_offset(batch, left_batch_strides)
    right_batch = _flattened_offset(batch, right_batch_strides)
    left_rows = _flattened_offset(rows, left_row_strides)
    right_columns = _flattened_offset(columns, right_column_strides)
    real = tl.zeros((block_m, block_n), dtype=tl.float32)
    imag = tl.zeros((block_m, block_n), dtype=tl.float32)
    for block_offset in range(0, tl.cdiv(k_size, block_k)):
        indices = block_offset * block_k + reduction
        left_reduction = _flattened_offset(indices, left_reduction_strides)
        right_reduction = _flattened_offset(indices, right_reduction_strides)
        left_offsets = left_batch + left_rows[:, None] + left_reduction[None, :]
        right_offsets = right_batch + right_reduction[:, None] + right_columns[None, :]
        left_mask = (rows[:, None] < m_size) & (indices[None, :] < k_size)
        right_mask = (indices[:, None] < k_size) & (columns[None, :] < n_size)
        ar = tl.load(left_parts + 2 * left_offsets, mask=left_mask, other=0.0)
        ai = tl.load(left_parts + 2 * left_offsets + 1, mask=left_mask, other=0.0)
        br = tl.load(right_parts + 2 * right_offsets, mask=right_mask, other=0.0)
        bi = tl.load(right_parts + 2 * right_offsets + 1, mask=right_mask, other=0.0)
        if conjugate_left:
            ai = -ai
        if conjugate_right:
            bi = -bi
        real += tl.dot(ar, br, allow_tf32=False) - tl.dot(ai, bi, allow_tf32=False)
        imag += tl.dot(ar, bi, allow_tf32=False) + tl.dot(ai, br, allow_tf32=False)
    output_offsets = batch * m_size * n_size + rows[:, None] * n_size + columns[None, :]
    output_mask = (rows[:, None] < m_size) & (columns[None, :] < n_size)
    tl.store(output_parts + 2 * output_offsets, real, mask=output_mask)
    tl.store(output_parts + 2 * output_offsets + 1, imag, mask=output_mask)


@triton.jit
def _complex_bmm_kernel(
    left_parts: tl.tensor,
    right_parts: tl.tensor,
    output_parts: tl.tensor,
    m_size: tl.constexpr,
    n_size: tl.constexpr,
    k_size: tl.constexpr,
    left_batch_stride: tl.constexpr,
    left_row_stride: tl.constexpr,
    left_reduction_stride: tl.constexpr,
    right_batch_stride: tl.constexpr,
    right_reduction_stride: tl.constexpr,
    right_column_stride: tl.constexpr,
    conjugate_left: tl.constexpr,
    conjugate_right: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
) -> None:
    batch = tl.program_id(2)
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    columns = tl.program_id(1) * block_n + tl.arange(0, block_n)
    reduction = tl.arange(0, block_k)
    left_base = batch * left_batch_stride
    right_base = batch * right_batch_stride
    real = tl.zeros((block_m, block_n), dtype=tl.float32)
    imag = tl.zeros((block_m, block_n), dtype=tl.float32)
    for offset in range(0, tl.cdiv(k_size, block_k)):
        indices = offset * block_k + reduction
        left_offsets = (
            left_base
            + rows[:, None] * left_row_stride
            + indices[None, :] * left_reduction_stride
        )
        right_offsets = (
            right_base
            + indices[:, None] * right_reduction_stride
            + columns[None, :] * right_column_stride
        )
        left_mask = (rows[:, None] < m_size) & (indices[None, :] < k_size)
        right_mask = (indices[:, None] < k_size) & (columns[None, :] < n_size)
        ar = tl.load(left_parts + 2 * left_offsets, mask=left_mask, other=0.0)
        ai = tl.load(left_parts + 2 * left_offsets + 1, mask=left_mask, other=0.0)
        br = tl.load(right_parts + 2 * right_offsets, mask=right_mask, other=0.0)
        bi = tl.load(right_parts + 2 * right_offsets + 1, mask=right_mask, other=0.0)
        if conjugate_left:
            ai = -ai
        if conjugate_right:
            bi = -bi
        real += tl.dot(ar, br, allow_tf32=False) - tl.dot(ai, bi, allow_tf32=False)
        imag += tl.dot(ar, bi, allow_tf32=False) + tl.dot(ai, br, allow_tf32=False)
    output_offsets = batch * m_size * n_size + rows[:, None] * n_size + columns[None, :]
    output_mask = (rows[:, None] < m_size) & (columns[None, :] < n_size)
    tl.store(output_parts + 2 * output_offsets, real, mask=output_mask)
    tl.store(output_parts + 2 * output_offsets + 1, imag, mask=output_mask)


def _launch(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    conjugate_left: bool = False,
    conjugate_right: bool = False,
) -> torch.Tensor:
    if left.ndim != 3 or right.ndim != 3:
        raise ValueError("fused complex BMM requires rank-3 tensors")
    batch, rows, reduction = left.shape
    right_batch, right_reduction, columns = right.shape
    if batch != right_batch or reduction != right_reduction:
        raise ValueError("fused complex BMM dimensions do not align")
    if left.is_conj() or right.is_conj() or left.is_neg() or right.is_neg():
        raise ValueError("fused complex BMM expects physical, non-conjugate inputs")
    left_parts = torch.view_as_real(left)
    right_parts = torch.view_as_real(right)
    output_parts = torch.empty(
        batch, rows, columns, 2, dtype=torch.float32, device=left.device
    )
    block_m = 16 if rows <= 16 else 32
    block_n = 16 if columns <= 16 else 32
    block_k = 16 if reduction <= 16 else 32
    grid = (triton.cdiv(rows, block_m), triton.cdiv(columns, block_n), batch)
    _complex_bmm_kernel[grid](
        left_parts,
        right_parts,
        output_parts,
        m_size=rows,
        n_size=columns,
        k_size=reduction,
        left_batch_stride=left.stride(0),
        left_row_stride=left.stride(1),
        left_reduction_stride=left.stride(2),
        right_batch_stride=right.stride(0),
        right_reduction_stride=right.stride(1),
        right_column_stride=right.stride(2),
        conjugate_left=conjugate_left,
        conjugate_right=conjugate_right,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=4,
        num_stages=3,
    )
    return torch.view_as_complex(output_parts)


def _launch_layout(
    left: torch.Tensor,
    right: torch.Tensor,
    left_permutation: tuple[int, ...],
    right_permutation: tuple[int, ...],
    batch_shape: tuple[int, ...],
    left_shape: tuple[int, ...],
    contracted_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    *,
    conjugate_left: bool = False,
    conjugate_right: bool = False,
) -> torch.Tensor:
    batch_rank = len(batch_shape)
    left_rank = len(left_shape)
    contracted_rank = len(contracted_shape)
    ordered_left_strides = tuple(left.stride(axis) for axis in left_permutation)
    ordered_right_strides = tuple(right.stride(axis) for axis in right_permutation)
    left_batch_strides = ordered_left_strides[:batch_rank] or (0,)
    left_row_strides = ordered_left_strides[batch_rank : batch_rank + left_rank] or (0,)
    left_reduction_strides = ordered_left_strides[batch_rank + left_rank :] or (0,)
    right_batch_strides = ordered_right_strides[:batch_rank] or (0,)
    right_reduction_strides = ordered_right_strides[
        batch_rank : batch_rank + contracted_rank
    ] or (0,)
    right_column_strides = ordered_right_strides[batch_rank + contracted_rank :] or (0,)

    def packed_layout(
        shape: tuple[int, ...], strides: tuple[int, ...]
    ) -> tuple[int, ...]:
        if len(shape) > 8:
            raise ValueError("layout-aware fused BMM supports at most 8 axes per group")
        return tuple(
            (int(stride) << 32) | int(dimension)
            for dimension, stride in reversed(tuple(zip(shape or (1,), strides)))
        )

    batch = prod(batch_shape) or 1
    rows = prod(left_shape) or 1
    reduction = prod(contracted_shape) or 1
    columns = prod(right_shape) or 1
    output_parts = torch.empty(
        batch, rows, columns, 2, dtype=torch.float32, device=left.device
    )
    block_m = 16 if rows <= 16 else 32
    block_n = 16 if columns <= 16 else 32
    block_k = 16 if reduction <= 16 else 32
    grid = (triton.cdiv(rows, block_m), triton.cdiv(columns, block_n), batch)
    _complex_layout_bmm_kernel[grid](
        torch.view_as_real(left),
        torch.view_as_real(right),
        output_parts,
        m_size=rows,
        n_size=columns,
        k_size=reduction,
        left_batch_strides=packed_layout(batch_shape, left_batch_strides),
        left_row_strides=packed_layout(left_shape, left_row_strides),
        left_reduction_strides=packed_layout(contracted_shape, left_reduction_strides),
        right_batch_strides=packed_layout(batch_shape, right_batch_strides),
        right_reduction_strides=packed_layout(
            contracted_shape, right_reduction_strides
        ),
        right_column_strides=packed_layout(right_shape, right_column_strides),
        conjugate_left=conjugate_left,
        conjugate_right=conjugate_right,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=4,
        num_stages=3,
    )
    return torch.view_as_complex(output_parts)


class _FusedComplexLayoutBMM(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: _LayoutBMMContext,
        left: torch.Tensor,
        right: torch.Tensor,
        left_permutation: tuple[int, ...],
        right_permutation: tuple[int, ...],
        shapes: tuple[
            tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]
        ],
    ) -> torch.Tensor:
        ctx.save_for_backward(left, right)
        ctx.left_permutation = left_permutation
        ctx.right_permutation = right_permutation
        ctx.shapes = shapes
        return _launch_layout(left, right, left_permutation, right_permutation, *shapes)

    @staticmethod
    def backward(
        ctx: _LayoutBMMContext, gradient: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None, None]:
        left, right = ctx.saved_tensors
        batch_shape, left_shape, contracted_shape, right_shape = ctx.shapes
        batch_rank = len(batch_shape)
        gradient = gradient.reshape((*batch_shape, *left_shape, *right_shape))
        gradient_permutation = tuple(range(gradient.ndim))
        right_transpose_permutation = (
            *ctx.right_permutation[:batch_rank],
            *ctx.right_permutation[batch_rank + len(contracted_shape) :],
            *ctx.right_permutation[batch_rank : batch_rank + len(contracted_shape)],
        )
        left_transpose_permutation = (
            *ctx.left_permutation[:batch_rank],
            *ctx.left_permutation[batch_rank + len(left_shape) :],
            *ctx.left_permutation[batch_rank : batch_rank + len(left_shape)],
        )
        left_gradient = _launch_layout(
            gradient,
            right,
            gradient_permutation,
            right_transpose_permutation,
            batch_shape,
            left_shape,
            right_shape,
            contracted_shape,
            conjugate_right=True,
        )
        right_gradient = _launch_layout(
            left,
            gradient,
            left_transpose_permutation,
            gradient_permutation,
            batch_shape,
            contracted_shape,
            left_shape,
            right_shape,
            conjugate_left=True,
        )
        inverse_left = tuple(
            ctx.left_permutation.index(axis) for axis in range(left.ndim)
        )
        inverse_right = tuple(
            ctx.right_permutation.index(axis) for axis in range(right.ndim)
        )
        left_gradient = left_gradient.reshape(
            (*batch_shape, *left_shape, *contracted_shape)
        ).permute(inverse_left)
        right_gradient = right_gradient.reshape(
            (*batch_shape, *contracted_shape, *right_shape)
        ).permute(inverse_right)
        return left_gradient, right_gradient, None, None, None


def fused_complex_layout_bmm(
    left: torch.Tensor,
    right: torch.Tensor,
    left_permutation: tuple[int, ...],
    right_permutation: tuple[int, ...],
    shapes: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]],
) -> torch.Tensor:
    """BMM-lower high-rank tensors without materializing permuted inputs."""

    if (
        not left.is_cuda
        or left.dtype != torch.complex64
        or right.dtype != torch.complex64
    ):
        batch_shape, left_shape, contracted_shape, right_shape = shapes
        batch = prod(batch_shape) or 1
        rows = prod(left_shape) or 1
        reduction = prod(contracted_shape) or 1
        columns = prod(right_shape) or 1
        return torch.bmm(
            left.permute(left_permutation).reshape(batch, rows, reduction),
            right.permute(right_permutation).reshape(batch, reduction, columns),
        )
    apply: Callable[..., object] = _FusedComplexLayoutBMM.apply
    result = apply(left, right, left_permutation, right_permutation, shapes)
    if not isinstance(result, torch.Tensor):
        raise TypeError("Complex layout BMM autograd must return a tensor")
    return result


class _FusedComplexBMM(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: _BMMContext, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        ctx.save_for_backward(left, right)
        return _launch(left, right)

    @staticmethod
    def backward(
        ctx: _BMMContext, gradient: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left, right = ctx.saved_tensors
        left_gradient = _launch(
            gradient,
            right.transpose(-2, -1),
            conjugate_right=True,
        )
        right_gradient = _launch(
            left.transpose(-2, -1),
            gradient,
            conjugate_left=True,
        )
        return left_gradient, right_gradient


def fused_complex_bmm(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return ``left @ right`` using one fused real/imag Triton kernel."""

    if (
        not left.is_cuda
        or left.dtype != torch.complex64
        or right.dtype != torch.complex64
    ):
        return torch.bmm(left, right)
    apply: Callable[[torch.Tensor, torch.Tensor], object] = _FusedComplexBMM.apply
    result = apply(left, right)
    if not isinstance(result, torch.Tensor):
        raise TypeError("Complex BMM autograd must return a tensor")
    return result


__all__ = ["fused_complex_bmm", "fused_complex_layout_bmm"]
