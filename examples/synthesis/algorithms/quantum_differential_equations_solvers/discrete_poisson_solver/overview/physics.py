"""Fresh grid data, Walsh coefficients and finite-difference reference problem."""

import math

import torch

import flagquantum as fq

GRID_SIZE = 8
QPE_SIZE = 6
PREFACTOR = 1 / 64
MAX_EIG = math.pi**2
GRID = (torch.arange(GRID_SIZE, dtype=torch.float64) + 0.5) / GRID_SIZE
ZGRID = torch.kron(GRID, GRID) * torch.kron(GRID - 1, GRID - 1)
SOURCE = torch.tanh(ZGRID).square()
SOURCE /= SOURCE.norm()
AMPLITUDES = tuple(float(v) for v in SOURCE)
EIGENVALUES_X = (
    4 * torch.sin(math.pi * torch.arange(1, 9, dtype=torch.float64) / 16).square()
)
EIGENVALUES_Y = (
    4 * torch.sin(math.pi * torch.arange(8, dtype=torch.float64) / 16).square()
)
ANGLES = tuple(
    0.0 if word == 0 else 2 * math.asin(PREFACTOR / word) for word in range(64)
)


def walsh_coefficients(values: torch.Tensor) -> tuple[float, ...]:
    """Transform the complete eight-point spectrum without dropping identity."""
    return tuple(
        math.fsum(
            (-1) ** ((mask & word).bit_count()) * float(value)
            for word, value in enumerate(values)
        )
        / len(values)
        for mask in range(len(values))
    )


def make_operator() -> fq.Observable:
    """Retain source low-x/high-y logical indices, including both identity terms."""
    result = 0 * fq.I(0)
    for offset, values in ((0, EIGENVALUES_X), (3, EIGENVALUES_Y)):
        for mask, coefficient in enumerate(walsh_coefficients(values)):
            term = fq.I(0)
            for bit in range(3):
                if mask & (1 << bit):
                    term = term @ fq.Z(5 - offset - bit)
            result += coefficient * term
    return result


OPERATOR = make_operator()


def finite_difference_problem() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute source Dirichlet-x/Neumann-y boundaries and transposed RHS."""
    interior = (
        2 * torch.eye(8, dtype=torch.float64)
        - torch.diag(torch.ones(7, dtype=torch.float64), 1)
        - torch.diag(torch.ones(7, dtype=torch.float64), -1)
    )
    hx, hy = interior.clone(), interior.clone()
    hx[0, 0] = hx[-1, -1] = 3
    hy[0, 0] = hy[-1, -1] = 1
    identity = torch.eye(8, dtype=torch.float64)
    matrix = torch.kron(hx, identity) + torch.kron(identity, hy)
    rhs = SOURCE.reshape(8, 8).T.flatten()
    return matrix, rhs, torch.linalg.solve(matrix, rhs).reshape(8, 8)


def spatial_matrices() -> (
    tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]
):
    samples = torch.arange(8, dtype=torch.float64) + 0.5
    sine = 0.5 * torch.sin(
        math.pi * torch.outer(torch.arange(1, 9, dtype=torch.float64), samples) / 8
    )
    cosine = 0.5 * torch.cos(
        math.pi * torch.outer(torch.arange(8, dtype=torch.float64), samples) / 8
    )
    sine[-1] /= math.sqrt(2)
    cosine[0] /= math.sqrt(2)
    return tuple(tuple(float(v) for v in row) for row in sine), tuple(
        tuple(float(v) for v in row) for row in cosine
    )


SINE, COSINE = spatial_matrices()
