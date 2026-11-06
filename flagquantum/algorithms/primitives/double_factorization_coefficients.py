"""Read real eigensystems from double-factorization coefficient matrices."""

import torch


def coefficient_eigenpairs(
    matrix: torch.Tensor,
) -> tuple[tuple[float, tuple[float, ...]], ...]:
    """Diagonalize a finite, real symmetric matrix in ascending eigenvalue order."""
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or not matrix.shape[0]
        or matrix.is_complex()
        or matrix.layout != torch.strided
    ):
        raise ValueError("DF coefficients must be nonempty square real dense matrices")
    coefficients = matrix.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(coefficients).all() or not torch.equal(
        coefficients, coefficients.T
    ):
        raise ValueError("DF coefficients must be finite and symmetric")
    values, vectors = torch.linalg.eigh(coefficients)
    pairs = []
    for value, column in zip(values.tolist(), vectors.T.tolist(), strict=True):
        # Fix the sign of the largest-magnitude component before quantization.
        sign = -1 if max(column, key=abs) < 0 else 1
        pairs.append((float(value), tuple(sign * float(entry) for entry in column)))
    return tuple(pairs)
