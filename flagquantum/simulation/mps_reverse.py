"""Rank-local numerical primitives for MPS reverse mode."""

from __future__ import annotations

from typing import Sequence

import torch


def project_mps_adjoint(
    value: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Project a stale truncated-bond adjoint onto the current tensor shape."""

    if tuple(value.shape) == tuple(target.shape):
        return value
    if value.ndim != target.ndim or value.shape[0] != target.shape[0]:
        raise ValueError(
            "MPS reverse adjoint rank/batch mismatch: "
            f"adjoint={tuple(value.shape)} output={tuple(target.shape)}"
        )
    projected = torch.zeros_like(target)
    slices = tuple(
        slice(0, min(int(source), int(destination)))
        for source, destination in zip(value.shape, target.shape)
    )
    projected[slices] = value[slices]
    return projected


def mps_vjp(
    outputs: Sequence[torch.Tensor],
    inputs: Sequence[torch.Tensor],
    parameters: Sequence[torch.Tensor],
    output_adjoints: Sequence[torch.Tensor],
) -> tuple[torch.Tensor | None, ...]:
    """Evaluate one rank-local MPS vector-Jacobian product."""

    if len(outputs) != len(output_adjoints):
        raise ValueError("MPS outputs and output adjoints must have equal arity")
    differentiable = tuple(
        (output, project_mps_adjoint(adjoint, output))
        for output, adjoint in zip(outputs, output_adjoints)
        if output.requires_grad
    )
    if not differentiable:
        raise ValueError("MPS VJP has no differentiable output")
    return torch.autograd.grad(
        tuple(item[0] for item in differentiable),
        tuple(inputs) + tuple(parameters),
        grad_outputs=tuple(item[1] for item in differentiable),
        allow_unused=True,
        retain_graph=True,
    )


__all__ = ("mps_vjp", "project_mps_adjoint")
