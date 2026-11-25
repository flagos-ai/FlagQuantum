"""Source matrices for the extended tree, cyclic walk and marked graph."""

import torch

REAL = torch.float64
COMPLEX = torch.complex128


def tree(size: int) -> torch.Tensor:
    matrix = torch.diag(
        torch.tensor([0.7] + [0.5] * (size // 2 - 1) + [0.7] * (size // 2), dtype=REAL)
    )
    matrix[0, 1] = matrix[1, 0] = 0.3
    for parent in range(1, size // 2):
        for child in (2 * parent, 2 * parent + 1):
            matrix[parent, child] = matrix[child, parent] = 0.3
    return matrix


def cyclic(size: int = 8) -> torch.Tensor:
    identity = torch.eye(size, dtype=REAL)
    return (
        0.4 * identity
        + 0.3 * identity.roll(1, dims=0)
        + 0.3 * identity.roll(-1, dims=0)
    )


def discriminant(size: int, *, marked: bool) -> torch.Tensor:
    transition = torch.full((size, size), 1 / size, dtype=REAL)
    if marked:
        transition[0, :] = 0
        transition[0, 0] = 1
    return (transition * transition.T).sqrt()


def pauli_terms(matrix: torch.Tensor) -> tuple[tuple[float, str], ...]:
    """Expand each small symmetric adjacency matrix in an orthogonal Pauli basis."""
    from itertools import product

    pauli = {
        "I": torch.eye(2, dtype=COMPLEX),
        "X": torch.tensor([[0, 1], [1, 0]], dtype=COMPLEX),
        "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=COMPLEX),
        "Z": torch.diag(torch.tensor([1, -1], dtype=COMPLEX)),
    }
    size = matrix.size(0)
    bits = size.bit_length() - 1
    terms = []
    for word in product("IXYZ", repeat=bits):
        operator = pauli[word[0]]
        for axis in word[1:]:
            operator = torch.kron(operator, pauli[axis])
        coefficient = (operator.conj() * matrix).sum() / size
        if abs(complex(coefficient)) > 1e-12:
            if abs(float(coefficient.imag)) > 1e-12:
                raise ValueError("Expected a real Pauli coefficient")
            terms.append((float(coefficient.real), "".join(word)))
    return tuple(terms)
