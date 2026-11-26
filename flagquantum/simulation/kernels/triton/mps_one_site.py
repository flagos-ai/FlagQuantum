"""Fused CUDA kernel for an MPS one-site gate contraction."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from ._jit import jit
else:
    from triton import jit


class _OneSiteContext(Protocol):
    @property
    def saved_tensors(self) -> tuple[torch.Tensor, ...]: ...

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


@jit
def _one_site_forward_kernel(
    tensor_parts: tl.tensor,
    gate_parts: tl.tensor,
    output_parts: tl.tensor,
    left_dim: tl.constexpr,
    right_dim: tl.constexpr,
    logical_elements: tl.constexpr,
    batched_gate: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < logical_elements
    right_index = offsets % right_dim
    batch_left = offsets // right_dim
    left_index = batch_left % left_dim
    batch = batch_left // left_dim

    for output_physical in range(2):
        real = tl.zeros((block_size,), dtype=tl.float32)
        imag = tl.zeros((block_size,), dtype=tl.float32)
        for input_physical in range(2):
            tensor_offset = (
                (batch * left_dim + left_index) * 2 + input_physical
            ) * right_dim + right_index
            gate_offset = output_physical * 2 + input_physical
            tensor_real = tl.load(
                tensor_parts + 2 * tensor_offset, mask=mask, other=0.0
            )
            tensor_imag = tl.load(
                tensor_parts + 2 * tensor_offset + 1, mask=mask, other=0.0
            )
            if batched_gate:
                gate_offset += batch * 4
                gate_real = tl.load(gate_parts + 2 * gate_offset, mask=mask, other=0.0)
                gate_imag = tl.load(
                    gate_parts + 2 * gate_offset + 1, mask=mask, other=0.0
                )
            else:
                gate_real = tl.load(gate_parts + 2 * gate_offset)
                gate_imag = tl.load(gate_parts + 2 * gate_offset + 1)
            real += gate_real * tensor_real - gate_imag * tensor_imag
            imag += gate_real * tensor_imag + gate_imag * tensor_real

        output_offset = (
            (batch * left_dim + left_index) * 2 + output_physical
        ) * right_dim + right_index
        tl.store(output_parts + 2 * output_offset, real, mask=mask)
        tl.store(output_parts + 2 * output_offset + 1, imag, mask=mask)


def _reference(tensor: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    equation = "pq,blqr->blpr" if gate.ndim == 2 else "bpq,blqr->blpr"
    return torch.einsum(equation, gate, tensor)


def _launch(tensor: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 4 or int(tensor.shape[2]) != 2:
        raise ValueError("MPS one-site tensor must have shape [batch,left,2,right]")
    batch, left_dim, _, right_dim = tensor.shape
    if gate.shape not in {(2, 2), (batch, 2, 2)}:
        raise ValueError("one-site gate must have shape [2,2] or [batch,2,2]")
    if gate.device != tensor.device:
        raise ValueError("MPS tensor and one-site gate must share a device")

    tensor = tensor.contiguous()
    gate = gate.contiguous()
    output_parts = torch.empty(
        batch,
        left_dim,
        2,
        right_dim,
        2,
        dtype=torch.float32,
        device=tensor.device,
    )
    logical_elements = batch * left_dim * right_dim
    block_size = 256
    _one_site_forward_kernel[(triton.cdiv(logical_elements, block_size),)](
        torch.view_as_real(tensor.resolve_conj().resolve_neg()),
        torch.view_as_real(gate.resolve_conj().resolve_neg()),
        output_parts,
        left_dim=left_dim,
        right_dim=right_dim,
        logical_elements=logical_elements,
        batched_gate=gate.ndim == 3,
        block_size=block_size,
        num_warps=4,
    )
    return torch.view_as_complex(output_parts)


class _FusedMPSOneSite(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: _OneSiteContext,
        tensor: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(tensor, gate)
        return _launch(tensor, gate)

    @staticmethod
    def backward(
        ctx: _OneSiteContext,
        gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tensor, gate = ctx.saved_tensors
        if gate.ndim == 3:
            tensor_gradient = torch.einsum("bpq,blpr->blqr", torch.conj(gate), gradient)
            gate_gradient = torch.einsum("blpr,blqr->bpq", gradient, torch.conj(tensor))
        else:
            tensor_gradient = torch.einsum("pq,blpr->blqr", torch.conj(gate), gradient)
            gate_gradient = torch.einsum("blpr,blqr->pq", gradient, torch.conj(tensor))
        return tensor_gradient, gate_gradient


def fused_mps_one_site(tensor: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Apply a one-site gate to ``[batch,left,2,right]`` MPS tensors."""

    if (
        not tensor.is_cuda
        or tensor.dtype != torch.complex64
        or gate.dtype != torch.complex64
    ):
        return _reference(tensor, gate)
    apply: Callable[[torch.Tensor, torch.Tensor], object] = _FusedMPSOneSite.apply
    result = apply(tensor, gate)
    if not isinstance(result, torch.Tensor):
        raise TypeError("MPS one-site autograd must return a tensor")
    return result


__all__ = ["fused_mps_one_site"]
