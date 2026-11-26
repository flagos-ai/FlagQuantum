"""Fused CUDA VJP for one Hermitian MPS observable contribution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from ._jit import jit
else:
    from triton import jit

_MAX_BOND = 64
_MAX_CONTRACTION_WORK = 1 << 25


@jit
def _mps_hermitian_observable_adjoint_kernel(
    left_parts: tl.tensor,
    operator_parts: tl.tensor,
    tensor_parts: tl.tensor,
    right_parts: tl.tensor,
    weights: tl.tensor,
    output_parts: tl.tensor,
    left_dim: tl.constexpr,
    right_dim: tl.constexpr,
    block_output: tl.constexpr,
    block_reduction: tl.constexpr,
) -> None:
    batch = tl.program_id(1)
    output = tl.program_id(0) * block_output + tl.arange(0, block_output)
    output_left = output // (2 * right_dim)
    output_physical = (output // right_dim) % 2
    output_right = output % right_dim
    total_real = tl.zeros((block_output,), dtype=tl.float32)
    total_imag = tl.zeros((block_output,), dtype=tl.float32)
    reduction_size: tl.constexpr = left_dim * 2 * right_dim

    for block_offset in range(tl.cdiv(reduction_size, block_reduction)):
        reduction = block_offset * block_reduction + tl.arange(0, block_reduction)
        inner_left = reduction // (2 * right_dim)
        inner_physical = (reduction // right_dim) % 2
        inner_right = reduction % right_dim
        output_mask = output < left_dim * 2 * right_dim
        reduction_mask = reduction < reduction_size
        mask = output_mask[:, None] & reduction_mask[None, :]

        left_offset = (
            batch * left_dim * left_dim
            + output_left[:, None] * left_dim
            + inner_left[None, :]
        )
        operator_offset = output_physical[:, None] * 2 + inner_physical[None, :]
        tensor_offset = (
            batch * left_dim * 2 * right_dim
            + inner_left[None, :] * 2 * right_dim
            + inner_physical[None, :] * right_dim
            + inner_right[None, :]
        )
        right_offset = (
            batch * right_dim * right_dim
            + output_right[:, None] * right_dim
            + inner_right[None, :]
        )

        left_real = tl.load(left_parts + 2 * left_offset, mask=mask, other=0.0)
        left_imag = tl.load(left_parts + 2 * left_offset + 1, mask=mask, other=0.0)
        operator_real = tl.load(
            operator_parts + 2 * operator_offset, mask=mask, other=0.0
        )
        operator_imag = tl.load(
            operator_parts + 2 * operator_offset + 1, mask=mask, other=0.0
        )
        tensor_real = tl.load(tensor_parts + 2 * tensor_offset, mask=mask, other=0.0)
        tensor_imag = tl.load(
            tensor_parts + 2 * tensor_offset + 1, mask=mask, other=0.0
        )
        right_real = tl.load(right_parts + 2 * right_offset, mask=mask, other=0.0)
        right_imag = tl.load(right_parts + 2 * right_offset + 1, mask=mask, other=0.0)

        left_operator_real = left_real * operator_real - left_imag * operator_imag
        left_operator_imag = left_real * operator_imag + left_imag * operator_real
        tensor_right_real = tensor_real * right_real - tensor_imag * right_imag
        tensor_right_imag = tensor_real * right_imag + tensor_imag * right_real
        total_real += tl.sum(
            left_operator_real * tensor_right_real
            - left_operator_imag * tensor_right_imag,
            axis=1,
        )
        total_imag += tl.sum(
            left_operator_real * tensor_right_imag
            + left_operator_imag * tensor_right_real,
            axis=1,
        )

    scale = 2.0 * tl.load(weights + batch)
    output_offset = batch * left_dim * 2 * right_dim + output
    output_mask = output < left_dim * 2 * right_dim
    tl.store(output_parts + 2 * output_offset, scale * total_real, mask=output_mask)
    tl.store(
        output_parts + 2 * output_offset + 1,
        scale * total_imag,
        mask=output_mask,
    )


def _validate(
    tensor: torch.Tensor,
    left_environment: torch.Tensor,
    right_environment: torch.Tensor,
    operator: torch.Tensor,
    weights: torch.Tensor,
) -> None:
    if tensor.ndim != 4 or int(tensor.shape[2]) != 2:
        raise ValueError("MPS tensor must have shape [batch,left,2,right]")
    batch, left_dim, _, right_dim = tensor.shape
    if min(batch, left_dim, right_dim) <= 0:
        raise ValueError("MPS observable-adjoint dimensions must be positive")
    if left_environment.shape != (batch, left_dim, left_dim):
        raise ValueError("left environment must have shape [batch,left,left]")
    if right_environment.shape != (batch, right_dim, right_dim):
        raise ValueError("right environment must have shape [batch,right,right]")
    if operator.shape != (2, 2):
        raise ValueError("local observable operator must have shape [2,2]")
    if weights.shape != (batch,):
        raise ValueError("observable weights must have shape [batch]")
    if tensor.dtype not in (torch.complex64, torch.complex128):
        raise ValueError("MPS observable-adjoint tensors must use a complex dtype")
    complex_inputs = (left_environment, right_environment, operator)
    if any(item.device != tensor.device for item in (*complex_inputs, weights)):
        raise ValueError("MPS observable-adjoint inputs must share a device")
    if any(item.dtype != tensor.dtype for item in complex_inputs):
        raise ValueError("MPS observable-adjoint complex inputs must share a dtype")
    expected_weight_dtype = (
        torch.float32 if tensor.dtype == torch.complex64 else torch.float64
    )
    if weights.dtype != expected_weight_dtype:
        raise ValueError("observable weights must use the matching real dtype")


def _reference(
    tensor: torch.Tensor,
    left_environment: torch.Tensor,
    right_environment: torch.Tensor,
    operator: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    # For Hermitian L, O, and R, the real quadratic form's two Wirtinger
    # contributions are equal. The VJP is therefore twice one contraction.
    return (
        2
        * weights[:, None, None, None]
        * torch.einsum(
            "bij,pq,bjqs,brs->bipr",
            left_environment,
            operator,
            tensor,
            right_environment,
        )
    ).detach()


def _supported_shape(
    tensor: torch.Tensor,
    left_environment: torch.Tensor,
    right_environment: torch.Tensor,
    operator: torch.Tensor,
    weights: torch.Tensor,
) -> bool:
    batch, left_dim, _, right_dim = tensor.shape
    local_elements = left_dim * 2 * right_dim
    contraction_work = batch * local_elements * local_elements
    inputs = (tensor, left_environment, right_environment, operator, weights)
    return bool(
        all(item.is_contiguous() for item in inputs)
        and all(not item.is_conj() and not item.is_neg() for item in inputs)
        and left_dim <= _MAX_BOND
        and right_dim <= _MAX_BOND
        and contraction_work <= _MAX_CONTRACTION_WORK
    )


def _launch(
    tensor: torch.Tensor,
    left_environment: torch.Tensor,
    right_environment: torch.Tensor,
    operator: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    batch, left_dim, _, right_dim = tensor.shape
    output_parts = torch.empty(
        batch,
        left_dim,
        2,
        right_dim,
        2,
        dtype=torch.float32,
        device=tensor.device,
    )
    block_output = 8
    block_reduction = 64
    grid = (triton.cdiv(left_dim * 2 * right_dim, block_output), batch)
    _mps_hermitian_observable_adjoint_kernel[grid](
        torch.view_as_real(left_environment),
        torch.view_as_real(operator),
        torch.view_as_real(tensor),
        torch.view_as_real(right_environment),
        weights,
        output_parts,
        left_dim=left_dim,
        right_dim=right_dim,
        block_output=block_output,
        block_reduction=block_reduction,
        num_warps=4,
    )
    return torch.view_as_complex(output_parts)


def fused_mps_hermitian_observable_adjoint(
    tensor: torch.Tensor,
    left_environment: torch.Tensor,
    right_environment: torch.Tensor,
    operator: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Return the local tensor VJP for a Hermitian MPS observable term.

    The left environment, right environment, and local operator are required to
    be Hermitian by the semantic contract. Runtime dispatch establishes that
    invariant without adding device synchronization to this kernel wrapper.
    """

    _validate(tensor, left_environment, right_environment, operator, weights)
    if (
        not tensor.is_cuda
        or tensor.dtype != torch.complex64
        or not _supported_shape(
            tensor,
            left_environment,
            right_environment,
            operator,
            weights,
        )
    ):
        return _reference(
            tensor,
            left_environment,
            right_environment,
            operator,
            weights,
        )
    return _launch(
        tensor,
        left_environment,
        right_environment,
        operator,
        weights,
    )


__all__ = ["fused_mps_hermitian_observable_adjoint"]
