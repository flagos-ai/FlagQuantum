"""Fused sampled-wire collapse and boundary propagation for MPS states."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton.language as tl

if TYPE_CHECKING:
    from ._jit import jit
else:
    from triton import jit

_MAX_BOND = 64


@jit
def _mps_sampling_collapse_kernel(
    site_parts: tl.tensor,
    next_parts: tl.tensor,
    bits: tl.tensor,
    collapsed_parts: tl.tensor,
    propagated_parts: tl.tensor,
    norms: tl.tensor,
    right_dim: tl.constexpr,
    next_right_dim: tl.constexpr,
    block_right: tl.constexpr,
    block_next: tl.constexpr,
) -> None:
    batch = tl.program_id(0)
    right_offsets = tl.arange(0, block_right)
    next_offsets = tl.arange(0, block_next)
    right_mask = right_offsets < right_dim
    next_mask = next_offsets < next_right_dim
    bit = tl.load(bits + batch)

    boundary_offset = (batch * 2 + bit) * right_dim + right_offsets
    boundary_real = tl.load(
        site_parts + 2 * boundary_offset,
        mask=right_mask,
        other=0.0,
    )
    boundary_imag = tl.load(
        site_parts + 2 * boundary_offset + 1,
        mask=right_mask,
        other=0.0,
    )
    norm = tl.sqrt(
        tl.sum(boundary_real * boundary_real + boundary_imag * boundary_imag, axis=0)
    )
    tl.store(norms + batch, norm)
    inverse_norm = 1.0 / tl.maximum(norm, 1.0e-12)
    boundary_real *= inverse_norm
    boundary_imag *= inverse_norm

    collapsed_base = batch * 2
    tl.store(collapsed_parts + 2 * collapsed_base, tl.where(bit == 0, 1.0, 0.0))
    tl.store(collapsed_parts + 2 * collapsed_base + 1, 0.0)
    tl.store(
        collapsed_parts + 2 * (collapsed_base + 1),
        tl.where(bit == 1, 1.0, 0.0),
    )
    tl.store(collapsed_parts + 2 * (collapsed_base + 1) + 1, 0.0)

    for physical in range(2):
        next_element = (
            (batch * right_dim + right_offsets[:, None]) * 2 + physical
        ) * next_right_dim + next_offsets[None, :]
        matrix_mask = right_mask[:, None] & next_mask[None, :]
        next_real = tl.load(
            next_parts + 2 * next_element,
            mask=matrix_mask,
            other=0.0,
        )
        next_imag = tl.load(
            next_parts + 2 * next_element + 1,
            mask=matrix_mask,
            other=0.0,
        )
        output_real = tl.sum(
            boundary_real[:, None] * next_real - boundary_imag[:, None] * next_imag,
            axis=0,
        )
        output_imag = tl.sum(
            boundary_real[:, None] * next_imag + boundary_imag[:, None] * next_real,
            axis=0,
        )
        output_element = (batch * 2 + physical) * next_right_dim + next_offsets
        tl.store(
            propagated_parts + 2 * output_element,
            output_real,
            mask=next_mask,
        )
        tl.store(
            propagated_parts + 2 * output_element + 1,
            output_imag,
            mask=next_mask,
        )


def _validate(
    site: torch.Tensor,
    next_site: torch.Tensor,
    bits: torch.Tensor,
) -> None:
    if site.ndim != 4 or tuple(site.shape[1:3]) != (1, 2):
        raise ValueError("sampled MPS site must have shape [batch,1,2,right]")
    if next_site.ndim != 4 or int(next_site.shape[2]) != 2:
        raise ValueError("next MPS site must have shape [batch,left,2,right]")
    batch, _, _, right_dim = site.shape
    if batch <= 0 or right_dim <= 0 or int(next_site.shape[-1]) <= 0:
        raise ValueError("MPS sampling-collapse dimensions must be positive")
    if int(next_site.shape[0]) != batch or int(next_site.shape[1]) != right_dim:
        raise ValueError("sampled boundary and next MPS site bonds must match")
    if bits.shape != (batch,):
        raise ValueError("sampled bits must have shape [batch]")
    if site.dtype not in (torch.complex64, torch.complex128):
        raise ValueError("MPS sampling collapse requires a complex dtype")
    if next_site.dtype != site.dtype:
        raise ValueError("MPS sampling-collapse tensors must share one dtype")
    if bits.dtype not in (torch.int32, torch.int64):
        raise ValueError("sampled bits must use an integer dtype")
    if site.device != next_site.device or site.device != bits.device:
        raise ValueError("MPS sampling-collapse inputs must share one device")
    valid_bits = ((bits >= 0) & (bits <= 1)).all()
    if bits.is_cuda:
        torch._assert_async(valid_bits, "sampled bits must contain only zero or one")
    elif not bool(valid_bits):
        raise ValueError("sampled bits must contain only zero or one")


def _reference(
    site: torch.Tensor,
    next_site: torch.Tensor,
    bits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = int(site.shape[0])
    batches = torch.arange(batch, device=site.device)
    boundary = site[batches, 0, bits.to(torch.int64), :]
    norms = torch.linalg.vector_norm(boundary, dim=-1)
    if bool(torch.any(~torch.isfinite(norms))) or bool(torch.any(norms <= 1.0e-12)):
        raise RuntimeError("sampled MPS branch is not finite and positive")
    boundary = boundary / norms[:, None]
    collapsed = torch.zeros(
        batch,
        1,
        2,
        1,
        dtype=site.dtype,
        device=site.device,
    )
    collapsed[batches, 0, bits.to(torch.int64), 0] = 1
    propagated = torch.einsum("bl,blsr->bsr", boundary, next_site).unsqueeze(1)
    return collapsed, propagated


def _supported_shape(
    site: torch.Tensor,
    next_site: torch.Tensor,
    bits: torch.Tensor,
) -> bool:
    return bool(
        site.is_cuda
        and site.dtype == torch.complex64
        and not site.requires_grad
        and not next_site.requires_grad
        and site.is_contiguous()
        and next_site.is_contiguous()
        and bits.is_contiguous()
        and not site.is_conj()
        and not site.is_neg()
        and not next_site.is_conj()
        and not next_site.is_neg()
        and int(site.shape[-1]) <= _MAX_BOND
        and int(next_site.shape[-1]) <= _MAX_BOND
    )


def _launch(
    site: torch.Tensor,
    next_site: torch.Tensor,
    bits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = int(site.shape[0])
    right_dim = int(site.shape[-1])
    next_right_dim = int(next_site.shape[-1])
    collapsed = torch.empty(batch, 1, 2, 1, dtype=site.dtype, device=site.device)
    propagated = torch.empty(
        batch,
        1,
        2,
        next_right_dim,
        dtype=site.dtype,
        device=site.device,
    )
    norms = torch.empty(batch, dtype=torch.float32, device=site.device)
    block_right = 1 << (right_dim - 1).bit_length()
    block_next = 1 << (next_right_dim - 1).bit_length()
    _mps_sampling_collapse_kernel[(batch,)](
        torch.view_as_real(site),
        torch.view_as_real(next_site),
        bits,
        torch.view_as_real(collapsed),
        torch.view_as_real(propagated),
        norms,
        right_dim=right_dim,
        next_right_dim=next_right_dim,
        block_right=block_right,
        block_next=block_next,
        num_warps=4,
    )
    torch._assert_async(
        torch.isfinite(norms).all() & (norms > 1.0e-12).all(),
        "sampled MPS branch is not finite and positive",
    )
    return collapsed, propagated


def fused_mps_sampling_collapse(
    site: torch.Tensor,
    next_site: torch.Tensor,
    bits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse one sampled MPS wire and propagate its normalized boundary.

    The direct Triton implementation is forward-only and bounded to contiguous
    complex64 CUDA inputs with bond dimensions at most 64. Unsupported inputs
    retain the exact PyTorch operation.
    """

    _validate(site, next_site, bits)
    if not _supported_shape(site, next_site, bits):
        return _reference(site, next_site, bits)
    return _launch(site, next_site, bits)


__all__ = ["fused_mps_sampling_collapse"]
