"""Autograd-enabled statevector probability generation for CUDA tensors."""

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


class _ProbabilityContext(Protocol):
    @property
    def saved_tensors(self) -> tuple[torch.Tensor, ...]: ...

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


@jit
def _statevector_probabilities_kernel(
    state_parts: tl.tensor,
    probabilities: tl.tensor,
    element_count: tl.tensor,
    block_size: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < element_count
    real = tl.load(state_parts + 2 * offsets, mask=mask, other=0.0)
    imag = tl.load(state_parts + 2 * offsets + 1, mask=mask, other=0.0)
    tl.store(probabilities + offsets, real * real + imag * imag, mask=mask)


@jit
def _statevector_probabilities_backward_kernel(
    state_parts: tl.tensor,
    probability_gradient: tl.tensor,
    state_gradient_parts: tl.tensor,
    element_count: tl.tensor,
    block_size: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < element_count
    real = tl.load(state_parts + 2 * offsets, mask=mask, other=0.0)
    imag = tl.load(state_parts + 2 * offsets + 1, mask=mask, other=0.0)
    gradient = 2.0 * tl.load(
        probability_gradient + offsets,
        mask=mask,
        other=0.0,
    )
    tl.store(state_gradient_parts + 2 * offsets, gradient * real, mask=mask)
    tl.store(state_gradient_parts + 2 * offsets + 1, gradient * imag, mask=mask)


def _reference(state: torch.Tensor) -> torch.Tensor:
    return torch.abs(state) ** 2


def _validate(state: torch.Tensor) -> None:
    if state.ndim != 2:
        raise ValueError("statevector probabilities require shape [batch, amplitudes]")
    if min(state.shape) <= 0:
        raise ValueError("statevector probability dimensions must be positive")
    if state.dtype not in (torch.complex64, torch.complex128):
        raise ValueError("statevector probabilities require a complex dtype")


def _supported(state: torch.Tensor) -> bool:
    return bool(
        state.is_cuda
        and state.dtype == torch.complex64
        and state.is_contiguous()
        and not state.is_conj()
        and not state.is_neg()
    )


def _launch_forward(state: torch.Tensor) -> torch.Tensor:
    probabilities = torch.empty(
        state.shape,
        dtype=torch.float32,
        device=state.device,
    )
    element_count = state.numel()
    block_size = 256
    _statevector_probabilities_kernel[(triton.cdiv(element_count, block_size),)](
        torch.view_as_real(state),
        probabilities,
        element_count,
        block_size=block_size,
        num_warps=4,
        num_stages=2,
    )
    return probabilities


def _launch_backward(
    state: torch.Tensor,
    probability_gradient: torch.Tensor,
) -> torch.Tensor:
    probability_gradient = probability_gradient.contiguous()
    state_gradient = torch.empty_like(state)
    element_count = state.numel()
    block_size = 256
    _statevector_probabilities_backward_kernel[
        (triton.cdiv(element_count, block_size),)
    ](
        torch.view_as_real(state),
        probability_gradient,
        torch.view_as_real(state_gradient),
        element_count,
        block_size=block_size,
        num_warps=4,
        num_stages=2,
    )
    return state_gradient


class _StatevectorProbabilities(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _ProbabilityContext, state: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(state)
        return _launch_forward(state)

    @staticmethod
    def backward(
        ctx: _ProbabilityContext,
        probability_gradient: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        (state,) = ctx.saved_tensors
        return (_launch_backward(state, probability_gradient),)


def statevector_probabilities(state: torch.Tensor) -> torch.Tensor:
    """Return one probability per amplitude of a batched flat statevector.

    The CUDA complex64 path fuses real/imaginary magnitude work into one Triton
    pass and provides an explicit first-order backward kernel. Unsupported
    devices, dtypes, or layouts retain the exact PyTorch operation.
    """

    _validate(state)
    if not _supported(state):
        return _reference(state)
    apply: Callable[[torch.Tensor], object] = _StatevectorProbabilities.apply
    result = apply(state)
    if not isinstance(result, torch.Tensor):
        raise TypeError("Statevector probability autograd must return a tensor")
    return result


__all__ = ["statevector_probabilities"]
