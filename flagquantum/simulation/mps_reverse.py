"""Rank-local numerical primitives for MPS reverse mode."""

from __future__ import annotations

from typing import Sequence

import torch

from .mps.factorization import _split_pair_matrix, _split_pair_matrix_bucket
from .mps.models import MPSConfig


def factor_mps_reverse_pair(
    pair_matrix: torch.Tensor,
    *,
    left_dim: int,
    right_dim: int,
    config: MPSConfig,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, float | int | str],
]:
    """Create the differentiable pair leaf and its MPS factorization."""

    with torch.enable_grad():
        pair_leaf = pair_matrix.detach().requires_grad_(True)
        left, right, info = _split_pair_matrix(
            pair_leaf,
            left_dim=left_dim,
            right_dim=right_dim,
            config=config,
        )
    return pair_leaf, left, right, dict(info)


def factor_mps_reverse_pair_bucket(
    pair_matrices: torch.Tensor,
    *,
    left_dim: int,
    right_dim: int,
    config: MPSConfig,
    batched_truncated_split: bool,
) -> tuple[
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, float | int | str],
    ],
    ...,
]:
    """Factor a bucket while preserving the selected reverse gradient path."""

    if pair_matrices.ndim != 4 or int(pair_matrices.shape[0]) < 1:
        raise ValueError(
            "reverse factorization bucket must have shape [pairs,batch,rows,columns]"
        )
    if not batched_truncated_split:
        return tuple(
            factor_mps_reverse_pair(
                pair,
                left_dim=left_dim,
                right_dim=right_dim,
                config=config,
            )
            for pair in pair_matrices
        )

    splits = _split_pair_matrix_bucket(
        pair_matrices.detach(),
        left_dim=left_dim,
        right_dim=right_dim,
        config=config,
    )
    outputs = []
    for pair, (batched_left, _, raw_info) in zip(pair_matrices, splits):
        after_left = batched_left.detach()
        retained_rank = int(after_left.shape[-1])
        retained_u = after_left.reshape(int(pair.shape[0]), left_dim * 2, retained_rank)
        with torch.enable_grad():
            pair_leaf = pair.detach().requires_grad_(True)
            after_right = torch.matmul(
                torch.conj(retained_u).transpose(-2, -1), pair_leaf
            ).reshape(int(pair.shape[0]), retained_rank, 2, right_dim)
        outputs.append(
            (
                pair_leaf,
                after_left,
                after_right,
                {**raw_info, "gradient_method": "projected_stop_subspace"},
            )
        )
    return tuple(outputs)


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


__all__ = (
    "factor_mps_reverse_pair",
    "factor_mps_reverse_pair_bucket",
    "mps_vjp",
    "project_mps_adjoint",
)
