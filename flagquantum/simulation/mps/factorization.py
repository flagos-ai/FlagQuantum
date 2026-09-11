"""QR/SVD factorization and truncation numerics for MPS tensors."""

from __future__ import annotations

import torch
from torch.profiler import record_function

from .models import MPSConfig

_DENSE_Z_SUM_WEIGHT_CACHE: dict[
    tuple[int, tuple[int, ...], str, torch.dtype], torch.Tensor
] = {}
_SVD_FALLBACK_STATS = {
    "requested_driver_failures": 0,
    "gesvd_driver_fallbacks": 0,
    "isolated_gesvd_retries": 0,
    "isolated_gesvd_matrices": 0,
    "nonfinite_svd_outputs": 0,
}


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
    is_cuda = _is_cuda_tensor(matrix)
    requested = driver if is_cuda else None

    try:
        return _checked_svd(matrix, driver=requested)
    except RuntimeError as requested_error:
        _SVD_FALLBACK_STATS["requested_driver_failures"] += 1
        if not is_cuda:
            raise
        if requested not in (None, "gesvd"):
            try:
                result = _checked_svd(matrix, driver="gesvd")
                _SVD_FALLBACK_STATS["gesvd_driver_fallbacks"] += 1
                return result
            except RuntimeError:
                pass
        if matrix.ndim <= 2:
            raise requested_error
        flattened = matrix.reshape(-1, matrix.shape[-2], matrix.shape[-1])
        outputs = []
        try:
            for item in flattened:
                outputs.append(_checked_svd(item, driver="gesvd"))
        except RuntimeError as isolated_error:
            raise RuntimeError(
                "MPS strict SVD failed for batched and isolated CUDA gesvd; "
                f"batch_shape={tuple(matrix.shape[:-2])}, "
                f"matrix_shape={tuple(matrix.shape[-2:])}"
            ) from isolated_error
        _SVD_FALLBACK_STATS["isolated_gesvd_retries"] += 1
        _SVD_FALLBACK_STATS["isolated_gesvd_matrices"] += len(outputs)
        u, singular, vh = zip(*outputs)
        batch_shape = matrix.shape[:-2]
        return (
            torch.stack(u).reshape(*batch_shape, *u[0].shape),
            torch.stack(singular).reshape(*batch_shape, *singular[0].shape),
            torch.stack(vh).reshape(*batch_shape, *vh[0].shape),
        )


def _checked_svd(
    matrix: torch.Tensor,
    *,
    driver: str | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u, singular, vh = torch.linalg.svd(matrix, full_matrices=False, driver=driver)
    result = (u, singular, vh)
    if all(bool(torch.isfinite(value).all()) for value in result):
        return result
    _SVD_FALLBACK_STATS["nonfinite_svd_outputs"] += 1
    # PyTorch exposes this class at runtime but omits its typed export.
    error_type: object = getattr(torch.linalg, "LinAlgError")
    if not isinstance(error_type, type) or not issubclass(error_type, RuntimeError):
        raise TypeError("torch.linalg.LinAlgError must be a RuntimeError subclass")
    raise error_type("MPS SVD returned non-finite factors")


def _is_cuda_tensor(matrix: torch.Tensor) -> bool:
    return bool(matrix.is_cuda)


def reset_mps_svd_fallback_stats() -> None:
    for key in _SVD_FALLBACK_STATS:
        _SVD_FALLBACK_STATS[key] = 0


def mps_svd_fallback_stats() -> dict[str, int]:
    return dict(_SVD_FALLBACK_STATS)


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
            q, r = torch.linalg.qr(matrix, mode="reduced")
            rank = int(q.shape[-1])
            return (
                q.reshape(matrix.shape[0], left_dim, 2, rank),
                r.reshape(matrix.shape[0], rank, 2, right_dim),
                {
                    "method": "qr",
                    "rank": full_rank,
                    "original_rank": full_rank,
                    "discarded_weight": 0.0,
                },
            )

    with record_function("flagquantum::mps::svd"):
        batched_u, batched_s, batched_vh = _cuda_svd(matrix, driver=config.svd_driver)
    svds = tuple(zip(batched_u, batched_s, batched_vh))
    ranks = [_select_rank(s, config.max_bond, config.cutoff) for _, s, _ in svds]
    # A shared batch dimension must retain every sample's required subspace.
    rank = max(ranks)
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
        rank = max(item_ranks)
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


def mps_qr_forward(
    left: torch.Tensor, right: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a rank-local thin QR update without owning runtime state."""

    bsz, left_dim, physical, right_dim = left.shape
    matrix = left.reshape(bsz, left_dim * physical, right_dim)
    if right_dim <= 2:
        first = matrix[..., 0]
        r00 = torch.linalg.vector_norm(first, dim=-1, keepdim=True)
        tiny = torch.finfo(first.real.dtype).tiny
        threshold = torch.finfo(first.real.dtype).eps * max(1, matrix.shape[-2])
        if torch.any(r00 <= threshold):
            raise ValueError("thin QR pullback is rank deficient in its first column")
        q0 = first / r00.clamp_min(tiny)
        columns = [q0]
        rows = [r00]
        if right_dim == 2:
            second = matrix[..., 1]
            r01 = torch.sum(q0.conj() * second, dim=-1, keepdim=True)
            residual = second - q0 * r01
            r11 = torch.linalg.vector_norm(residual, dim=-1, keepdim=True)
            scale = torch.maximum(
                r00, torch.linalg.vector_norm(second, dim=-1, keepdim=True)
            )
            if torch.any(r11 <= threshold * scale.clamp_min(1.0)):
                raise ValueError(
                    "thin QR pullback is rank deficient or near-degenerate"
                )
            q1 = residual / r11.clamp_min(tiny)
            columns.append(q1)
            rows = [
                torch.cat((r00, r01), dim=-1),
                torch.cat((torch.zeros_like(r11), r11), dim=-1),
            ]
        q = torch.stack(columns, dim=-1)
        transfer = (
            rows[0].unsqueeze(-2) if right_dim == 1 else torch.stack(rows, dim=-2)
        )
    else:
        q, transfer = torch.linalg.qr(matrix, mode="reduced")
    return q.reshape(bsz, left_dim, physical, q.shape[-1]), torch.einsum(
        "bij,bjsk->bisk", transfer, right
    )
