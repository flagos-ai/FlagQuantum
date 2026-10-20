"""First-order sensitivity corrections when a Krylov subspace closes exactly.

A zero Arnoldi residual proves the forward value, not its derivative under an
operator or input perturbation. The owning kernels supply the missing adjoint
action using their existing matrix-free solver. No numerical method lives here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch


def linear_adjoint(
    apply: Callable[[torch.Tensor], torch.Tensor], reference: torch.Tensor
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Read the Hilbert-Schmidt adjoint from the linear Torch action itself."""
    with torch.enable_grad():
        probe = torch.zeros_like(reference, requires_grad=True)
        image = apply(probe)

    def adjoint(vector: torch.Tensor) -> torch.Tensor:
        if not image.requires_grad:
            return torch.zeros_like(vector)
        (gradient,) = torch.autograd.grad(
            image, probe, vector, retain_graph=True, allow_unused=True
        )
        return torch.zeros_like(vector) if gradient is None else gradient

    return adjoint


class _ClosedSubspaceGradient(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        value: torch.Tensor,
        residual: torch.Tensor,
        adjoint: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        ctx.adjoint = adjoint
        return value.clone()

    @staticmethod
    def backward(
        ctx: Any, gradient: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        if torch.is_grad_enabled():
            raise RuntimeError(
                "higher-order derivatives at a closed Krylov subspace are not supported"
            )
        with torch.no_grad():
            correction = ctx.adjoint(gradient)
        return gradient, correction, None


def closed_subspace_gradient(
    value: torch.Tensor,
    residual: torch.Tensor,
    adjoint: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Preserve forward bits while adding the missing residual sensitivity."""
    if not torch.is_grad_enabled() or not residual.requires_grad:
        return value
    return cast(torch.Tensor, _ClosedSubspaceGradient.apply(value, residual, adjoint))
