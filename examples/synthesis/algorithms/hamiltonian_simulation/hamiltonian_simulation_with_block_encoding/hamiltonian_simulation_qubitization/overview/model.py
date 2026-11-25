"""Regenerate the source Hamiltonian, input, and complex Chebyshev polynomial."""

import math
import random
from dataclasses import dataclass

import numpy as np
import torch

DTYPE = torch.complex128
TIME = 22
DEGREE = 33
TOLERANCE = 1e-7
TERMS = (("II", 0.4), ("ZI", 0.1), ("XX", 0.05), ("ZZ", 0.2))
WORDS = tuple(word for word, _ in TERMS)
WEIGHTS = tuple(weight for _, weight in TERMS)
ALPHA = math.fsum(WEIGHTS)


@dataclass(frozen=True)
class Expansion:
    """Complex polynomial coefficients in ascending Chebyshev degree."""

    coefficients: tuple[complex, ...]
    scale: float


def hamiltonian_matrix() -> torch.Tensor:
    """Form the independent dense matrix, with logical bit zero least significant."""
    identity = torch.eye(2, dtype=DTYPE)
    x = torch.tensor([[0, 1], [1, 0]], dtype=DTYPE)
    z = torch.diag(torch.tensor([1, -1], dtype=DTYPE))
    return (
        0.4 * torch.eye(4, dtype=DTYPE)
        + 0.1 * torch.kron(z, identity)
        + 0.05 * torch.kron(x, x)
        + 0.2 * torch.kron(z, z)
    )


def fresh_input(seed: int = 7) -> torch.Tensor:
    """Draw the four positive source-style amplitudes with a local seed."""
    rng = random.Random(seed)
    values = torch.tensor([rng.random() for _ in range(4)], dtype=torch.float64)
    return torch.div(values, values.norm())


def expansion() -> Expansion:
    """Recompute the degree-33 Jacobi-Anger coefficients by cosine quadrature."""
    count = 128
    angles = (np.arange(count) + 0.5) * math.pi / count
    basis = np.cos(np.outer(np.arange(DEGREE + 1), angles))
    coefficients = (2 / count) * basis @ np.exp(-1j * TIME * ALPHA * np.cos(angles))
    coefficients[0] /= 2
    # Preserve the exact cosine-even and sine-odd structure.
    coefficients[::2] = coefficients[::2].real
    coefficients[1::2] = 1j * coefficients[1::2].imag
    values = tuple(complex(value) for value in coefficients)
    return Expansion(values, math.fsum(map(abs, values)))
