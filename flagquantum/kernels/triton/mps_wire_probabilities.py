"""Fused MPS single-qubit probability reduction for CUDA tensors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton.language as tl

if TYPE_CHECKING:
    from ._jit import jit
else:
    from triton import jit

_MAX_SITE_ELEMENTS = 1 << 12
_BLOCK_SIZE = 256


@jit
def _mps_qubit_probabilities_kernel(
    tensor_parts: tl.tensor,
    output: tl.tensor,
    site_elements: tl.constexpr,
    right_dim: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    batch = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    probability_zero = tl.zeros((), dtype=tl.float32)
    probability_one = tl.zeros((), dtype=tl.float32)

    for block_offset in range(tl.cdiv(site_elements, block_size)):
        indices = block_offset * block_size + offsets
        mask = indices < site_elements
        left = indices // right_dim
        right = indices % right_dim
        batch_offset = batch * site_elements * 2
        zero_offset = batch_offset + left * 2 * right_dim + right
        one_offset = zero_offset + right_dim

        zero_real = tl.load(
            tensor_parts + 2 * zero_offset,
            mask=mask,
            other=0.0,
        )
        zero_imag = tl.load(
            tensor_parts + 2 * zero_offset + 1,
            mask=mask,
            other=0.0,
        )
        one_real = tl.load(
            tensor_parts + 2 * one_offset,
            mask=mask,
            other=0.0,
        )
        one_imag = tl.load(
            tensor_parts + 2 * one_offset + 1,
            mask=mask,
            other=0.0,
        )
        probability_zero += tl.sum(
            zero_real * zero_real + zero_imag * zero_imag,
            axis=0,
        )
        probability_one += tl.sum(
            one_real * one_real + one_imag * one_imag,
            axis=0,
        )

    normalizer = tl.maximum(probability_zero + probability_one, 1.0e-12)
    tl.store(output + batch * 2, probability_zero / normalizer)
    tl.store(output + batch * 2 + 1, probability_one / normalizer)


def _validate(tensor: torch.Tensor) -> None:
    if tensor.ndim != 4 or int(tensor.shape[2]) != 2:
        raise ValueError("MPS tensor must have shape [batch,left,2,right]")
    batch, left_dim, _, right_dim = tensor.shape
    if min(batch, left_dim, right_dim) <= 0:
        raise ValueError("MPS qubit-probability dimensions must be positive")
    if tensor.dtype not in (torch.complex64, torch.complex128):
        raise ValueError("MPS qubit probabilities require a complex dtype")


def _reference(tensor: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sum(torch.abs(tensor) ** 2, dim=(1, 3))
    probabilities = torch.clamp(probabilities, min=0)
    normalizer = probabilities.sum(dim=-1, keepdim=True)
    return probabilities / torch.clamp(normalizer, min=1.0e-12)


def _supported_shape(tensor: torch.Tensor) -> bool:
    _, left_dim, _, right_dim = tensor.shape
    return bool(
        tensor.is_contiguous()
        and not tensor.is_conj()
        and not tensor.is_neg()
        and not tensor.requires_grad
        and left_dim * right_dim <= _MAX_SITE_ELEMENTS
    )


def _launch(tensor: torch.Tensor) -> torch.Tensor:
    batch, left_dim, _, right_dim = tensor.shape
    output = torch.empty(batch, 2, dtype=torch.float32, device=tensor.device)
    _mps_qubit_probabilities_kernel[(batch,)](
        torch.view_as_real(tensor),
        output,
        site_elements=left_dim * right_dim,
        right_dim=right_dim,
        block_size=_BLOCK_SIZE,
        num_warps=4,
    )
    return output


def fused_mps_qubit_probabilities(tensor: torch.Tensor) -> torch.Tensor:
    """Return normalized probabilities for the physical index of one MPS site.

    The Triton path fuses complex magnitude, both bond reductions, and
    normalization. Unsupported inputs retain the exact PyTorch operation.
    """

    _validate(tensor)
    if (
        not tensor.is_cuda
        or tensor.dtype != torch.complex64
        or not _supported_shape(tensor)
    ):
        return _reference(tensor)
    return _launch(tensor)


__all__ = ["fused_mps_qubit_probabilities"]
