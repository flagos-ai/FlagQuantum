"""Native matrix-product-state execution for FlagQuantum."""

from __future__ import annotations

from typing import Any

import torch
from torch.profiler import record_function

_DENSE_Z_SUM_WEIGHT_CACHE: dict[
    tuple[int, tuple[int, ...], str, torch.dtype], torch.Tensor
] = {}
_MPS_INSTRUCTION_SCHEDULE_CACHE: dict[tuple[Any, ...], tuple[tuple[int, ...], ...]] = {}
from .mps_models import (  # noqa: E402
    MPSConfig,
)


def _z_sum_dense_weights(
    n_wires: int,
    wires: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    key = (int(n_wires), tuple(int(wire) for wire in wires), str(device), dtype)
    cached = _DENSE_Z_SUM_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached
    indices = torch.arange(2 ** int(n_wires), device=device)
    weights = torch.zeros(2 ** int(n_wires), device=device, dtype=dtype)
    for wire in wires:
        bits = (indices >> (int(n_wires) - 1 - int(wire))) & 1
        weights = weights + (1 - 2 * bits.to(dtype))
    _DENSE_Z_SUM_WEIGHT_CACHE[key] = weights
    return weights


def _cuda_svd(
    matrix: torch.Tensor,
    *,
    driver: str | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the requested CUDA SVD driver with a correctness-preserving fallback."""
    try:
        return torch.linalg.svd(
            matrix,
            full_matrices=False,
            driver=driver if matrix.is_cuda else None,
        )
    except (RuntimeError, torch._C._LinAlgError):
        if not matrix.is_cuda or driver in (None, "gesvd"):
            raise
        return torch.linalg.svd(matrix, full_matrices=False, driver="gesvd")


def _split_pair_matrix(
    matrix: torch.Tensor,
    *,
    left_dim: int,
    right_dim: int,
    config: MPSConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int | str]]:
    """Split a two-site MPS matrix with an AD-friendly exact path."""

    full_rank = min(int(matrix.shape[-2]), int(matrix.shape[-1]))
    exact_split = config.cutoff <= 0 and (
        config.max_bond is None or int(config.max_bond) >= full_rank
    )
    if exact_split:
        with record_function("flagquantum::mps::qr"):
            if matrix.requires_grad:
                return _split_pair_matrix_exact_autograd(
                    matrix,
                    left_dim=left_dim,
                    right_dim=right_dim,
                    full_rank=full_rank,
                )
            left_parts = []
            right_parts = []
            for item in matrix:
                q, r = torch.linalg.qr(item, mode="reduced")
                rank = int(q.shape[-1])
                left_parts.append(q.reshape(left_dim, 2, rank))
                right_parts.append(r.reshape(rank, 2, right_dim))
            return (
                torch.stack(left_parts, dim=0),
                torch.stack(right_parts, dim=0),
                {
                    "method": "qr",
                    "rank": full_rank,
                    "original_rank": full_rank,
                    "discarded_weight": 0.0,
                },
            )

    with record_function("flagquantum::mps::svd"):
        svds = [
            _cuda_svd(matrix[b], driver=config.svd_driver)
            for b in range(matrix.shape[0])
        ]
    ranks = [_select_rank(s, config.max_bond, config.cutoff) for _, s, _ in svds]
    rank = min(ranks) if config.cutoff > 0 else ranks[0]
    original_rank = max(int(s.shape[0]) for _, s, _ in svds)
    singular_value_gap = (
        min(float(torch.abs(s[rank - 1] - s[rank]).detach().cpu()) for _, s, _ in svds)
        if rank < original_rank
        else None
    )
    left_parts = []
    right_parts = []
    step_error = 0.0
    for batch_index, (u, s, vh) in enumerate(svds):
        step_error = max(step_error, _discarded_weight(s, rank))
        u = u[:, :rank]
        if matrix.requires_grad:
            # Complex singular vectors have an undefined phase, and their exact
            # derivative is singular at repeated singular values.  Keep the
            # optimal forward subspace but stop its gradient; the projected
            # factor remains differentiable with respect to the pair matrix.
            # This is a stable approximate gradient, not exact SVD autograd.
            u = u.detach()
            projected = torch.matmul(
                torch.conj(u).transpose(-2, -1), matrix[batch_index]
            )
            left_parts.append(u.reshape(left_dim, 2, rank))
            right_parts.append(projected.reshape(rank, 2, right_dim))
        else:
            s = s[:rank]
            vh = vh[:rank, :]
            left_parts.append(u.reshape(left_dim, 2, rank))
            right_parts.append((s[:, None] * vh).reshape(rank, 2, right_dim))
    return (
        torch.stack(left_parts, dim=0),
        torch.stack(right_parts, dim=0),
        {
            "method": "svd",
            "rank": int(rank),
            "original_rank": original_rank,
            "discarded_weight": float(step_error),
            "gradient_method": (
                "projected_stop_subspace" if matrix.requires_grad else "none"
            ),
            **(
                {"singular_value_gap": float(singular_value_gap)}
                if singular_value_gap is not None
                else {}
            ),
        },
    )


def _split_pair_matrix_bucket(
    matrices: torch.Tensor,
    *,
    left_dim: int,
    right_dim: int,
    config: MPSConfig,
    svd_driver: str | None = None,
) -> tuple[tuple[torch.Tensor, torch.Tensor, dict[str, float | int | str]], ...]:
    """Split independent equal-shape pairs with one batched QR/SVD launch.

    The leading dimension identifies independent brickwork bonds and the next
    dimension is the circuit batch. Outputs remain per bond because cutoff
    policies may choose different retained ranks.
    """

    if matrices.ndim != 4 or int(matrices.shape[0]) < 1:
        raise ValueError(
            "factorization bucket must have shape [bonds,batch,rows,columns]"
        )
    if (
        matrices.requires_grad
        and config.cutoff <= 0
        and (
            config.max_bond is None
            or int(config.max_bond)
            >= min(int(matrices.shape[-2]), int(matrices.shape[-1]))
        )
    ):
        return tuple(
            _split_pair_matrix(
                matrix,
                left_dim=left_dim,
                right_dim=right_dim,
                config=config,
            )
            for matrix in matrices
        )

    full_rank = min(int(matrices.shape[-2]), int(matrices.shape[-1]))
    exact_split = config.cutoff <= 0 and (
        config.max_bond is None or int(config.max_bond) >= full_rank
    )
    if exact_split:
        with record_function("flagquantum::mps::batched_qr"):
            q, r = torch.linalg.qr(matrices, mode="reduced")
        rank = int(q.shape[-1])
        return tuple(
            (
                q[index].reshape(q.shape[1], left_dim, 2, rank),
                r[index].reshape(r.shape[1], rank, 2, right_dim),
                {
                    "method": "qr",
                    "rank": rank,
                    "original_rank": full_rank,
                    "discarded_weight": 0.0,
                },
            )
            for index in range(int(matrices.shape[0]))
        )

    with record_function("flagquantum::mps::batched_svd"):
        u, s, vh = _cuda_svd(
            matrices,
            driver=(svd_driver or config.svd_driver),
        )
    outputs = []
    for item in range(int(matrices.shape[0])):
        item_ranks = [
            _select_rank(s[item, batch], config.max_bond, config.cutoff)
            for batch in range(int(matrices.shape[1]))
        ]
        rank = min(item_ranks) if config.cutoff > 0 else item_ranks[0]
        singular_value_gap = (
            torch.min(torch.abs(s[item, :, rank - 1] - s[item, :, rank]))
            if rank < int(s.shape[-1])
            else None
        )
        step_error = max(
            _discarded_weight(s[item, batch], rank)
            for batch in range(int(matrices.shape[1]))
        )
        retained_u = u[item, :, :, :rank]
        if matrices.requires_grad:
            retained_u = retained_u.detach()
            projected = torch.matmul(
                torch.conj(retained_u).transpose(-2, -1), matrices[item]
            )
            right = projected.reshape(matrices.shape[1], rank, 2, right_dim)
        else:
            right = (s[item, :, :rank, None] * vh[item, :, :rank, :]).reshape(
                matrices.shape[1], rank, 2, right_dim
            )
        left = retained_u.reshape(matrices.shape[1], left_dim, 2, rank)
        outputs.append(
            (
                left,
                right,
                {
                    "method": "svd",
                    "rank": int(rank),
                    "original_rank": int(s.shape[-1]),
                    "discarded_weight": float(step_error),
                    "gradient_method": (
                        "projected_stop_subspace" if matrices.requires_grad else "none"
                    ),
                    **(
                        {"singular_value_gap": float(singular_value_gap.detach().cpu())}
                        if singular_value_gap is not None
                        else {}
                    ),
                },
            )
        )
    return tuple(outputs)


def _split_pair_matrix_exact_autograd(
    matrix: torch.Tensor,
    *,
    left_dim: int,
    right_dim: int,
    full_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int | str]]:
    bsz = int(matrix.shape[0])
    rows = int(matrix.shape[-2])
    cols = int(matrix.shape[-1])
    if rows <= cols:
        eye = torch.eye(rows, dtype=matrix.dtype, device=matrix.device).expand(
            bsz, rows, rows
        )
        left_tensor = eye.reshape(bsz, left_dim, 2, rows)
        right_tensor = matrix.reshape(bsz, rows, 2, right_dim)
        rank = rows
    else:
        left_tensor = matrix.reshape(bsz, left_dim, 2, cols)
        eye = torch.eye(cols, dtype=matrix.dtype, device=matrix.device).expand(
            bsz, cols, cols
        )
        right_tensor = eye.reshape(bsz, cols, 2, right_dim)
        rank = cols
    return (
        left_tensor,
        right_tensor,
        {
            "method": "exact_autograd",
            "rank": int(rank),
            "original_rank": int(full_rank),
            "discarded_weight": 0.0,
        },
    )


def _select_rank(values: torch.Tensor, max_bond: int | None, cutoff: float) -> int:
    rank = int(values.shape[0])
    if cutoff > 0:
        rank = int(torch.count_nonzero(values > cutoff).item())
        rank = max(rank, 1)
    if max_bond is not None:
        rank = min(rank, int(max_bond))
    return rank


def _discarded_weight(values: torch.Tensor, rank: int) -> float:
    if rank >= values.shape[0]:
        return 0.0
    discarded = values[rank:]
    return float(
        torch.sum(torch.real(discarded * torch.conj(discarded))).detach().cpu()
    )
