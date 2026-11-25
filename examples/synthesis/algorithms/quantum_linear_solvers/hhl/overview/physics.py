"""Compute the original matrix-HHL data with the existing PyTorch dependency."""

import math

import torch

import flagquantum as fq

MATRIX = torch.tensor(
    [
        [0.28, -0.01, 0.02, -0.1],
        [-0.01, 0.5, -0.22, -0.07],
        [0.02, -0.22, 0.43, -0.05],
        [-0.1, -0.07, -0.05, 0.42],
    ],
    dtype=torch.float64,
)
RAW_B = torch.tensor([1.0, 2.0, 4.0, 3.0], dtype=torch.float64)
B = RAW_B / torch.linalg.vector_norm(RAW_B)
QPE_SIZE = 4
RECIPROCAL_SCALE = 1 / 2**QPE_SIZE
UNITARY = tuple(
    tuple(complex(v) for v in row)
    for row in torch.matrix_exp(2j * math.pi * MATRIX).tolist()
)
ANGLES = tuple(
    0.0 if word == 0 else 2 * math.asin(1 / word) for word in range(2**QPE_SIZE)
)
AMPLITUDES = tuple(float(v) for v in B)

PAULIS = {
    "I": torch.eye(2, dtype=torch.complex128),
    "X": torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),
    "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128),
    "Z": torch.diag(torch.tensor([1, -1], dtype=torch.complex128)),
}


def decompose_matrix() -> tuple[tuple[str, float], ...]:
    """Recompute the full ordered Pauli expansion from the original dense matrix."""
    terms = []
    for high in "IXYZ":
        for low in "IXYZ":
            pauli = torch.kron(PAULIS[high], PAULIS[low])
            coefficient = complex(torch.trace(pauli @ MATRIX.to(torch.complex128)) / 4)
            if abs(coefficient.imag) > 1e-14:
                raise ValueError("Hermitian matrix must have real Pauli coefficients")
            if abs(coefficient.real) > 1e-15:
                terms.append((high + low, coefficient.real))
    return tuple(terms)


PAULI_TERMS = decompose_matrix()
REPETITIONS = tuple(4 * math.ceil(1.8**bit) for bit in range(QPE_SIZE))


def make_operator() -> fq.Observable:
    """Use logical low-to-high observable indices for dense high-to-low words."""
    axes = {"I": fq.I, "X": fq.X, "Y": fq.Y, "Z": fq.Z}
    operator = 0 * fq.I(0)
    for word, coefficient in PAULI_TERMS:
        operator += coefficient * (axes[word[0]](1) @ axes[word[1]](0))
    return operator


OPERATOR = make_operator()
