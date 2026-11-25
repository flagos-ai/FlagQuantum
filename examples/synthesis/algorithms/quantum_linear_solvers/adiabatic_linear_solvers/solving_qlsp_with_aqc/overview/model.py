"""Build the three adiabatic linear-solver scenarios from their input matrices."""

import itertools
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

SOURCE_RHS = (12.0, 10.0, 17.0, 26.0)
SOURCE_MATRIX = (
    (4.0, 1.0, 2.0, 0.0),
    (1.0, 3.0, 0.0, 1.0),
    (2.0, 0.0, 3.0, 1.0),
    (0.0, 1.0, 1.0, 2.0),
)
PERTURBED_MATRIX = (
    (3.9270525, 1.06841123, 2.09661281, -0.10400811),
    (1.06841123, 2.93584295, -0.0906049, 1.09754032),
    (2.09661281, -0.0906049, 2.87204449, 1.13774997),
    (-0.10400811, 1.09754032, 1.13774997, 1.85170585),
)

PAULIS = (
    np.eye(2, dtype=np.complex128),
    np.array([[0, 1], [1, 0]], dtype=np.complex128),
    np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    np.diag(np.array([1, -1], dtype=np.complex128)),
)


@dataclass(frozen=True)
class LinearSystem:
    """Dense inputs and Hamiltonian endpoints in ancilla-first tensor order."""

    matrix: FloatArray
    rhs: FloatArray
    initial: FloatArray
    h0: ComplexArray
    h1: ComplexArray
    exact: FloatArray
    condition_number: float


@dataclass(frozen=True)
class PauliCoefficient:
    """A Pauli word listed from most significant to least significant bit."""

    axes: tuple[int, ...]
    coefficient: float


def linear_system(matrix: tuple[tuple[float, ...], ...]) -> LinearSystem:
    """Construct both Hamiltonians and the normalized classical solution."""
    source = np.asarray(matrix, dtype=np.float64)
    if source.shape != (4, 4) or not np.all(np.isfinite(source)):
        raise ValueError("The source model requires a finite 4 by 4 matrix")
    if not np.array_equal(source, source.T):
        raise ValueError("The source model requires a symmetric matrix")
    rhs = np.asarray(SOURCE_RHS, dtype=np.float64)
    rhs /= np.linalg.norm(rhs)
    projector = np.eye(4) - np.outer(rhs, rhs)
    h0 = np.asarray(np.kron(PAULIS[1], projector), dtype=np.complex128)
    zero = np.zeros_like(projector)
    # H1 uses the original scale of A. Only the direct solve uses Frobenius normalization.
    h1 = np.block([[zero, source @ projector], [projector @ source, zero]])
    normalized = source / np.linalg.norm(source)
    solution = np.linalg.solve(normalized, rhs)
    solution /= np.linalg.norm(solution)
    return LinearSystem(
        normalized,
        rhs,
        np.concatenate((rhs, np.zeros(4, dtype=np.float64))),
        h0,
        h1.astype(np.complex128),
        np.concatenate((solution, np.zeros(4, dtype=np.float64))),
        float(np.linalg.cond(source)),
    )


def pauli_matrix(axes: tuple[int, ...]) -> ComplexArray:
    """Return a dense operator in the order of the supplied axes."""
    result: ComplexArray = np.ones((1, 1), dtype=np.complex128)
    for axis in axes:
        result = np.asarray(np.kron(result, PAULIS[axis]), dtype=np.complex128)
    return result


def _mask_order(axes: tuple[int, ...]) -> tuple[int, int]:
    flip, phase = 0, 0
    for axis in axes:
        flip = 2 * flip + int(axis in (1, 2))
        phase = 2 * phase + int(axis in (2, 3))
    return flip, phase


def decompose(hamiltonian: ComplexArray) -> tuple[PauliCoefficient, ...]:
    """Regenerate the ordered Pauli coefficients using trace orthogonality."""
    coefficients = []
    for axes in sorted(itertools.product(range(4), repeat=3), key=_mask_order):
        coefficient = complex(np.trace(pauli_matrix(axes) @ hamiltonian) / 8)
        if abs(coefficient.imag) > 1e-12:
            raise ValueError("The model requires real Pauli coefficients")
        if abs(coefficient.real) > 1e-12:
            coefficients.append(PauliCoefficient(axes, coefficient.real))
    return tuple(coefficients)


def spectrum(model: LinearSystem) -> tuple[FloatArray, FloatArray]:
    """Sample the source interpolation at all 100 spectrum points."""
    points = np.linspace(0.0, 1.0, 100)
    spectra = np.linalg.eigvalsh(
        (1.0 - points[:, None, None]) * model.h0 + points[:, None, None] * model.h1
    )
    return points, np.asarray(spectra, dtype=np.float64)
