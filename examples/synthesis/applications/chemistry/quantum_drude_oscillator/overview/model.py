"""Regenerate the finite oscillator operators and the analytic reference curves."""

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

RealArray = NDArray[np.float64]
POLARIZABILITY = 14.5
HALF_HBAR_OMEGA = 9.61


def ladder_projector(level: int, dimension: int, *, raising: bool) -> RealArray:
    """Return the transition between adjacent occupation levels."""
    if not 0 <= level < dimension - 1:
        raise ValueError("the transition needs two occupation levels in the basis")
    matrix = np.zeros((dimension, dimension), dtype=np.float64)
    matrix[level + 1, level] = 1.0
    return matrix if raising else matrix.T


def position_operator(dimension: int) -> RealArray:
    """Build the dimensionless position operator in the occupation basis."""
    position = np.zeros((dimension, dimension), dtype=np.float64)
    for level in range(dimension - 1):
        raising = ladder_projector(level, dimension, raising=True)
        position += np.sqrt((level + 1) / 2) * (raising + raising.T)
    return position


def oscillator_hamiltonian(
    oscillator_count: int, dimension: int, distance: float
) -> RealArray:
    """Assemble the bare and coupled operators at a fresh nearest-neighbor distance."""
    if oscillator_count not in (2, 3) or dimension not in (2, 4, 8, 16):
        raise ValueError(
            "expected two or three oscillators and a source local dimension"
        )
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("the nearest-neighbor distance must be finite and positive")
    position = position_operator(dimension)
    occupation = np.diag(2 * np.arange(dimension, dtype=np.float64) + 1)
    identity = np.eye(dimension, dtype=np.float64)

    def tensor_product(operators: dict[int, RealArray]) -> RealArray:
        result: RealArray = np.ones((1, 1), dtype=np.float64)
        for oscillator in reversed(range(oscillator_count)):
            result = np.asarray(
                np.kron(result, operators.get(oscillator, identity)), dtype=np.float64
            )
        return result

    size = dimension**oscillator_count
    matrix = np.zeros((size, size), dtype=np.float64)
    for first in range(oscillator_count):
        matrix += tensor_product({first: occupation})
        for second in range(first + 1, oscillator_count):
            coupling = -4 * POLARIZABILITY / ((second - first) * distance) ** 3
            matrix += coupling * tensor_product({first: position, second: position})
    return matrix


def finite_ground_energies(
    distances: Iterable[float], dimension: int, oscillator_count: int = 2
) -> RealArray:
    """Diagonalize the complete finite basis independently of VQE."""
    return np.asarray(
        [
            np.linalg.eigvalsh(
                oscillator_hamiltonian(oscillator_count, dimension, distance)
            )[0]
            for distance in distances
        ],
        dtype=np.float64,
    )


def harmonic_ground_energy(coupling: RealArray) -> RealArray:
    """Return the exact infinite-basis two-oscillator normal-mode energy."""
    if np.any(np.abs(coupling) > 2):
        raise ValueError("the source's bound normal modes require abs(coupling) <= 2")
    return np.asarray(np.sqrt(1 + coupling / 2) + np.sqrt(1 - coupling / 2))


def pairwise_interaction(distances: RealArray) -> RealArray:
    """Use the exact analytic two-body contribution for all three pairs."""
    near = -4 * POLARIZABILITY / distances**3
    far = near / 8
    return np.asarray(
        2 * (harmonic_ground_energy(near) - 2) + harmonic_ground_energy(far) - 2
    )
