"""Complete and strip bounded complex Laurent expansions."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from examples.synthesis.algorithms.quantum_primitives.gqsp.overview.phases import (
    rotation,
)

DEGREE = 33
SCALE = 0.9999
FFT_SIZE = 2**15
AngleRow = tuple[float, float, float]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class Factorization:
    """Fresh polynomial coefficients, phase rows and numerical residuals."""

    coefficients: tuple[complex, ...]
    rows: tuple[AngleRow, ...]
    completion_residual: float
    stripping_residual: float
    reconstructed_residual: float
    approximation_error: float


def _phase_row(column: ComplexArray) -> AngleRow:
    norm = float(np.linalg.norm(column))
    if not np.isfinite(norm) or norm < 1e-13:
        raise ValueError("The next coefficient is too small for stable phase stripping")
    upper, lower = column / norm
    theta = float(np.arctan2(abs(lower), abs(upper)))
    lam = float(np.angle(lower)) if abs(lower) > 1e-15 else 0.0
    phi = float(np.angle(upper)) - lam if abs(upper) > 1e-15 else 0.0
    return theta, phi, lam


def _polynomial(
    duration: float, degree: int, scale: float
) -> tuple[ComplexArray, float]:
    angles = 2 * np.pi * np.arange(FFT_SIZE) / FFT_SIZE
    exact = scale * np.exp(-1j * duration * np.cos(angles))
    modes = np.fft.fft(exact) / FFT_SIZE
    polynomial = np.concatenate((modes[-degree:], modes[: degree + 1]))
    # The cosine symbol has symmetric Laurent coefficients.
    polynomial = np.asarray((polynomial + polynomial[::-1]) / 2, dtype=np.complex128)
    reconstructed = np.fft.ifft(polynomial, FFT_SIZE) * FFT_SIZE
    target = exact * np.exp(1j * degree * angles)
    return polynomial, float(np.max(np.abs(reconstructed - target)))


def _spectral_completion(polynomial: ComplexArray) -> tuple[ComplexArray, float]:
    grid = np.fft.ifft(polynomial, FFT_SIZE) * FFT_SIZE
    residual = 1 - np.abs(grid) ** 2
    if not np.all(residual > 0):
        raise ValueError(
            "The source scaled polynomial must be contractive on the FFT grid"
        )
    log_coefficients = np.fft.fft(np.log(residual)) / FFT_SIZE
    # The analytic half of log(1-|P|^2) gives the outer spectral factor.
    analytic = np.zeros(FFT_SIZE, dtype=np.complex128)
    analytic[0] = log_coefficients[0] / 2
    analytic[1 : FFT_SIZE // 2] = log_coefficients[1 : FFT_SIZE // 2]
    raw = np.fft.fft(np.exp(np.fft.ifft(analytic) * FFT_SIZE)) / FFT_SIZE
    complement = np.asarray(raw[: len(polynomial)], dtype=np.complex128)
    identity = np.convolve(polynomial, polynomial[::-1].conj()) + np.convolve(
        complement, complement[::-1].conj()
    )
    identity[len(polynomial) - 1] -= 1
    error = float(np.abs(identity).sum())
    if error > 1e-10:
        raise ValueError("The spectral factor did not satisfy the coefficient identity")
    return complement, error


def factor(
    duration: float, *, degree: int = DEGREE, scale: float = SCALE
) -> Factorization:
    """Generate and verify the bounded Laurent factorization for the requested degree."""
    if not math.isfinite(duration):
        raise ValueError("The phase-fitting time must be finite")
    if type(degree) is not int or not 1 <= degree <= 1024:
        raise ValueError("The supported Laurent degree must lie in [1, 1024]")
    if not 0 < scale < 1:
        raise ValueError("The positive scale must lie below one")
    polynomial, approximation = _polynomial(duration, degree, scale)
    complement, completion = _spectral_completion(polynomial)
    vector: ComplexArray = np.stack((polynomial, complement))
    reverse: list[AngleRow] = []
    stripping = 0.0
    for _ in range(len(polynomial) - 1):
        leader = vector[:, -1]
        constant = vector[:, 0]
        if np.linalg.norm(constant) > np.linalg.norm(leader):
            leader = np.asarray((constant[1].conj(), -constant[0].conj()))
        row = _phase_row(leader)
        reverse.append(row)
        stripped = rotation(row).conj().T @ vector
        stripping = max(
            stripping, float(abs(stripped[0, 0])), float(abs(stripped[1, -1]))
        )
        vector = np.stack((stripped[0, 1:], stripped[1, :-1]))
    reverse.append(_phase_row(vector[:, 0]))
    rows = tuple(reversed(reverse))
    reconstructed = rotation(rows[0])[:, :1]
    for row in rows[1:]:
        shifted = np.zeros((2, reconstructed.shape[1] + 1), dtype=np.complex128)
        shifted[0, 1:] = reconstructed[0]
        shifted[1, :-1] = reconstructed[1]
        reconstructed = rotation(row) @ shifted
    reconstruction = float(np.abs(reconstructed[0] - polynomial).sum())
    if max(stripping, reconstruction) > 1e-9:
        raise ValueError(
            f"The fresh phases failed polynomial reconstruction: {stripping=} {reconstruction=} {completion=}"
        )
    return Factorization(
        tuple(map(complex, polynomial)),
        rows,
        completion,
        stripping,
        reconstruction,
        approximation,
    )
