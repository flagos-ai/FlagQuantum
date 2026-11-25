"""Bounded classical spectral completion and native GQSP phase factorization."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Factorization:
    """Native angle rows and numerical completion/stripping diagnostics."""

    phases: tuple[tuple[float, float, float], ...]
    completion_coefficient_error: float
    stripping_error: float


def rotation(row: tuple[float, float, float]) -> npt.NDArray[np.complex128]:
    """Return the native X/U/X/Z rotation used in polynomial factorization."""
    theta, phi, lam = row
    cosine, sine = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [np.exp(1j * (phi + lam)) * cosine, np.exp(1j * phi) * sine],
            [np.exp(1j * lam) * sine, -cosine],
        ],
        dtype=np.complex128,
    )


def _angles(column: npt.NDArray[np.complex128]) -> tuple[float, float, float]:
    norm = float(np.linalg.norm(column))
    if not np.isfinite(norm) or norm < 1e-14:
        raise ValueError("Degenerate stripping column; no fixed-angle fallback.")
    a, b = column / norm
    lam = float(np.angle(b)) if abs(b) > 1e-15 else 0.0
    phi = float(np.angle(a)) - lam if abs(a) > 1e-15 else 0.0
    return float(np.arctan2(abs(b), abs(a))), phi, lam


def factor_phases(coefficients: npt.NDArray[np.float64 | np.float32]) -> Factorization:
    """Complete and factor a bounded real polynomial with coefficient L1 < 1.

    This conservative finite numerical profile refuses endpoint degeneracy,
    near-unit roots or failed coefficient identities instead of claiming a
    general phase solver. Angles use the existing native X/U/X/Z convention.
    """
    if (
        coefficients.ndim != 1
        or not 2 <= len(coefficients) <= 7
        or coefficients.dtype != np.float64
        or not np.isfinite(coefficients).all()
        or abs(coefficients[0]) < 1e-8
        or abs(coefficients[-1]) < 1e-8
        or np.abs(coefficients).sum() >= 1
    ):
        raise ValueError(
            "Use 2..7 finite float64 coefficients, nonzero endpoints, L1 < 1."
        )
    degree = len(coefficients) - 1
    residual = -np.convolve(coefficients, coefficients[::-1])
    residual[degree] += 1
    roots = np.roots(residual[::-1])
    inside = roots[np.abs(roots) < 1]
    if len(inside) != degree or np.any(np.abs(inside) >= 1 - 1e-8):
        raise ValueError("Spectral roots are not separated from the unit circle.")
    complement = np.asarray(np.poly(inside)[::-1], dtype=np.complex128)
    complement *= np.sqrt(
        (1 - np.vdot(coefficients, coefficients).real)
        / np.vdot(complement, complement).real
    )
    completed = np.convolve(coefficients, coefficients[::-1]) + np.convolve(
        complement, complement[::-1].conj()
    )
    completed[degree] -= 1
    completion_error = float(np.abs(completed).sum())
    if not np.isfinite(completion_error) or completion_error > 1e-10:
        raise ValueError("Spectral completion failed coefficient validation.")
    vector = np.stack((coefficients.astype(np.complex128), complement))
    rows = []
    stripping_error = 0.0
    for _ in range(degree):
        row = _angles(vector[:, -1])
        rows.append(row)
        stripped = rotation(row).conj().T @ vector
        stripping_error = max(
            stripping_error, float(abs(stripped[0, 0])), float(abs(stripped[1, -1]))
        )
        if stripping_error > 1e-10:
            raise ValueError("Degree stripping failed endpoint validation.")
        vector = np.stack((stripped[0, 1:], stripped[1, :-1]))
    if abs(float(np.linalg.norm(vector[:, 0])) - 1) > 1e-10:
        raise ValueError("Final rotation column is not normalized.")
    rows.append(_angles(vector[:, 0]))
    return Factorization(tuple(reversed(rows)), completion_error, stripping_error)
