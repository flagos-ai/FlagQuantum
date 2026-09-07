"""Batched JAX MPS tensor kernels."""

from __future__ import annotations

from typing import Any


def jax_mps_apply_one_batched(tensor: Any, matrix: Any) -> Any:
    import jax.numpy as jnp

    matrix = jnp.asarray(matrix, dtype=tensor.dtype)
    if matrix.ndim == 2:
        return jnp.einsum("pq,blqr->blpr", matrix, tensor, precision="highest")
    return jnp.einsum("bpq,blqr->blpr", matrix, tensor, precision="highest")


def _select_rank(values: Any, *, max_bond: int | None, cutoff: float) -> int:
    import numpy as np

    values_np = np.asarray(values)
    if values_np.ndim == 1:
        values_np = values_np.reshape(1, -1)
    rank = int(values_np.shape[-1])
    if float(cutoff) > 0:
        counts = (values_np > float(cutoff)).sum(axis=-1)
        rank = max(1, int(counts.min()))
    if max_bond is not None:
        rank = min(rank, int(max_bond))
    return max(1, rank)


def _discarded_weight(values: Any, rank: int) -> float:
    import numpy as np

    values_np = np.asarray(values)
    if int(rank) >= int(values_np.shape[-1]):
        return 0.0
    discarded = values_np[..., int(rank) :]
    return float(np.max(np.sum(np.real(discarded * np.conj(discarded)), axis=-1)))


def jax_mps_split_pair_batched(
    matrix: Any,
    *,
    left_dim: int,
    right_dim: int,
    max_bond: int | None,
    cutoff: float,
) -> tuple[Any, Any, dict[str, Any]]:
    import jax.numpy as jnp

    bsz = int(matrix.shape[0])
    rows = int(matrix.shape[-2])
    cols = int(matrix.shape[-1])
    full_rank = min(rows, cols)
    exact_split = float(cutoff) <= 0 and (
        max_bond is None or int(max_bond) >= full_rank
    )
    if exact_split:
        if rows <= cols:
            rank = rows
            eye = jnp.eye(rank, dtype=matrix.dtype)
            left = jnp.broadcast_to(
                eye.reshape(1, int(left_dim), 2, rank),
                (bsz, int(left_dim), 2, rank),
            )
            right = matrix.reshape(bsz, rank, 2, int(right_dim))
            method = "identity_left_gauge"
        else:
            rank = cols
            left = matrix.reshape(bsz, int(left_dim), 2, rank)
            eye = jnp.eye(rank, dtype=matrix.dtype)
            right = jnp.broadcast_to(
                eye.reshape(1, rank, 2, int(right_dim)),
                (bsz, rank, 2, int(right_dim)),
            )
            method = "identity_right_gauge"
        return (
            left,
            right,
            {
                "method": method,
                "rank": rank,
                "original_rank": full_rank,
                "discarded_weight": 0.0,
            },
        )
    u, s, vh = jnp.linalg.svd(matrix, full_matrices=False)
    rank = _select_rank(s, max_bond=max_bond, cutoff=cutoff)
    step_error = _discarded_weight(s, rank)
    left = u[:, :, :rank].reshape(bsz, int(left_dim), 2, rank)
    right = (s[:, :rank, None] * vh[:, :rank, :]).reshape(bsz, rank, 2, int(right_dim))
    return (
        left,
        right,
        {
            "method": "svd",
            "rank": rank,
            "original_rank": full_rank,
            "discarded_weight": float(step_error),
        },
    )


def jax_mps_apply_two_batched(
    left: Any,
    right: Any,
    matrix: Any,
    *,
    max_bond: int | None,
    cutoff: float,
    reverse: bool = False,
) -> tuple[Any, Any, dict[str, Any]]:
    import jax.numpy as jnp

    matrix = jnp.asarray(matrix, dtype=left.dtype)
    theta = jnp.einsum("blsm,bmtr->blstr", left, right, precision="highest")
    if reverse:
        theta = jnp.swapaxes(theta, 2, 3)
    theta = theta.reshape(left.shape[0], left.shape[1], 4, right.shape[3])
    if matrix.ndim == 2:
        theta = jnp.einsum("ij,bljr->blir", matrix, theta, precision="highest")
    else:
        theta = jnp.einsum("bij,bljr->blir", matrix, theta, precision="highest")
    theta = theta.reshape(left.shape[0], left.shape[1], 2, 2, right.shape[3])
    if reverse:
        theta = jnp.swapaxes(theta, 2, 3)
    bsz, left_dim, _, _, right_dim = theta.shape
    return jax_mps_split_pair_batched(
        theta.reshape(bsz, left_dim * 2, 2 * right_dim),
        left_dim=int(left_dim),
        right_dim=int(right_dim),
        max_bond=max_bond,
        cutoff=cutoff,
    )
