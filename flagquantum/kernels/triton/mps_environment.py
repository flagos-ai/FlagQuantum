"""Fused CUDA kernel for identity and Pauli-Z MPS environment transfer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from ._jit import jit
else:
    from triton import jit

_MAX_BOND = 32
_MAX_CONTRACTION_WORK = 1 << 22


@jit
def _mps_environment_transfer_kernel(
    environment_parts: tl.tensor,
    tensor_parts: tl.tensor,
    output_parts: tl.tensor,
    left_dim: tl.constexpr,
    right_dim: tl.constexpr,
    insert_z: tl.constexpr,
    block_output: tl.constexpr,
    block_reduction: tl.constexpr,
) -> None:
    batch = tl.program_id(1)
    output = tl.program_id(0) * block_output + tl.arange(0, block_output)
    output_row = output // right_dim
    output_column = output % right_dim
    real = tl.zeros((block_output,), dtype=tl.float32)
    imag = tl.zeros((block_output,), dtype=tl.float32)
    reduction_size: tl.constexpr = left_dim * left_dim * 2

    for block_offset in range(tl.cdiv(reduction_size, block_reduction)):
        reduction = block_offset * block_reduction + tl.arange(0, block_reduction)
        physical = reduction % 2
        pair = reduction // 2
        input_column = pair % left_dim
        input_row = pair // left_dim
        reduction_mask = reduction < reduction_size
        environment_offset = (
            batch * left_dim * left_dim + input_row * left_dim + input_column
        )
        bra_offset = (
            batch * left_dim * 2 * right_dim
            + input_row * 2 * right_dim
            + physical * right_dim
            + output_row[:, None]
        )
        ket_offset = (
            batch * left_dim * 2 * right_dim
            + input_column * 2 * right_dim
            + physical * right_dim
            + output_column[:, None]
        )
        environment_real = tl.load(
            environment_parts + 2 * environment_offset[None, :],
            mask=reduction_mask[None, :],
            other=0.0,
        )
        environment_imag = tl.load(
            environment_parts + 2 * environment_offset[None, :] + 1,
            mask=reduction_mask[None, :],
            other=0.0,
        )
        bra_mask = (output_row[:, None] < right_dim) & reduction_mask[None, :]
        ket_mask = (output_column[:, None] < right_dim) & reduction_mask[None, :]
        bra_real = tl.load(tensor_parts + 2 * bra_offset, mask=bra_mask, other=0.0)
        bra_imag = tl.load(tensor_parts + 2 * bra_offset + 1, mask=bra_mask, other=0.0)
        ket_real = tl.load(tensor_parts + 2 * ket_offset, mask=ket_mask, other=0.0)
        ket_imag = tl.load(tensor_parts + 2 * ket_offset + 1, mask=ket_mask, other=0.0)

        partial_real = bra_real * environment_real + bra_imag * environment_imag
        partial_imag = bra_real * environment_imag - bra_imag * environment_real
        term_real = partial_real * ket_real - partial_imag * ket_imag
        term_imag = partial_real * ket_imag + partial_imag * ket_real
        if insert_z:
            sign = 1.0 - 2.0 * physical
            term_real *= sign[None, :]
            term_imag *= sign[None, :]
        real += tl.sum(term_real, axis=1)
        imag += tl.sum(term_imag, axis=1)

    output_offset = batch * right_dim * right_dim + output
    output_mask = output < right_dim * right_dim
    tl.store(output_parts + 2 * output_offset, real, mask=output_mask)
    tl.store(output_parts + 2 * output_offset + 1, imag, mask=output_mask)


def _validate(environment: torch.Tensor, tensor: torch.Tensor) -> None:
    if environment.ndim != 3:
        raise ValueError("MPS environment must have shape [batch,left,left]")
    if tensor.ndim != 4 or int(tensor.shape[2]) != 2:
        raise ValueError("MPS tensor must have shape [batch,left,2,right]")
    batch, left_dim, _, right_dim = tensor.shape
    if min(batch, left_dim, right_dim) <= 0:
        raise ValueError("MPS environment dimensions must be positive")
    if environment.shape != (batch, left_dim, left_dim):
        raise ValueError("MPS environment and tensor dimensions do not align")
    if environment.device != tensor.device:
        raise ValueError("MPS environment and tensor must share a device")
    if environment.dtype != tensor.dtype:
        raise ValueError("MPS environment and tensor must share a dtype")


def _reference(
    environment: torch.Tensor,
    tensor: torch.Tensor,
    *,
    insert_z: bool,
) -> torch.Tensor:
    signs = tensor.real.new_tensor((1.0, -1.0) if insert_z else (1.0, 1.0))
    return torch.einsum(
        "bij,bipr,bjps->brs",
        environment,
        tensor.conj(),
        tensor * signs.reshape(1, 1, 2, 1),
    )


def _supported_shape(environment: torch.Tensor, tensor: torch.Tensor) -> bool:
    batch, left_dim, _, right_dim = tensor.shape
    contraction_work = batch * left_dim * left_dim * right_dim * right_dim
    return bool(
        environment.is_contiguous()
        and tensor.is_contiguous()
        and not environment.is_conj()
        and not environment.is_neg()
        and not tensor.is_conj()
        and not tensor.is_neg()
        and left_dim <= _MAX_BOND
        and right_dim <= _MAX_BOND
        and contraction_work <= _MAX_CONTRACTION_WORK
    )


def _launch(
    environment: torch.Tensor,
    tensor: torch.Tensor,
    *,
    insert_z: bool,
) -> torch.Tensor:
    batch, left_dim, _, right_dim = tensor.shape
    output_parts = torch.empty(
        batch,
        right_dim,
        right_dim,
        2,
        dtype=torch.float32,
        device=tensor.device,
    )
    block_output = 8
    block_reduction = 64
    grid = (triton.cdiv(right_dim * right_dim, block_output), batch)
    _mps_environment_transfer_kernel[grid](
        torch.view_as_real(environment),
        torch.view_as_real(tensor),
        output_parts,
        left_dim=left_dim,
        right_dim=right_dim,
        insert_z=insert_z,
        block_output=block_output,
        block_reduction=block_reduction,
        num_warps=4,
    )
    return torch.view_as_complex(output_parts)


def fused_mps_environment_transfer(
    environment: torch.Tensor,
    tensor: torch.Tensor,
    *,
    insert_z: bool = False,
) -> torch.Tensor:
    """Transfer an MPS environment through identity or Pauli-Z at one site."""

    _validate(environment, tensor)
    if (
        not environment.is_cuda
        or environment.dtype != torch.complex64
        or environment.requires_grad
        or tensor.requires_grad
        or not _supported_shape(environment, tensor)
    ):
        return _reference(environment, tensor, insert_z=insert_z)
    return _launch(environment, tensor, insert_z=insert_z)


__all__ = ["fused_mps_environment_transfer"]
