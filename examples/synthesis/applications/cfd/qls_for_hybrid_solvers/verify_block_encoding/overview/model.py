"""Load the original matrix and regenerate its Pauli and diagonal decompositions."""

import itertools
import json
from pathlib import Path

import torch

CTYPE = torch.complex128
PAULIS = {
    "I": torch.eye(2, dtype=CTYPE),
    "X": torch.tensor([[0, 1], [1, 0]], dtype=CTYPE),
    "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=CTYPE),
    "Z": torch.diag(torch.tensor([1, -1], dtype=CTYPE)),
}


def source_matrix() -> torch.Tensor:
    raw = json.loads(Path(__file__).with_name("nozzle_small_csr.json").read_text())
    if tuple(raw["shape"]) != (8, 8) or len(raw["data"]) != 21:
        raise ValueError("expected the original eight-row, 21-entry nozzle matrix")
    matrix = torch.zeros((8, 8), dtype=CTYPE)
    for i, (lo, hi) in enumerate(
        zip(raw["indptr"][:-1], raw["indptr"][1:], strict=True)
    ):
        matrix[i, raw["indices"][lo:hi]] = torch.tensor(raw["data"][lo:hi], dtype=CTYPE)
    return matrix


def pauli_expansion(matrix: torch.Tensor) -> tuple[tuple[str, complex], ...]:
    reconstructed = torch.zeros_like(matrix)
    terms = []
    for names in itertools.product("IXYZ", repeat=3):
        first, second, third = (PAULIS[name] for name in names)
        operator = torch.kron(torch.kron(first, second), third)
        weight = (operator.conj() * matrix).sum() / 8
        if float(weight.abs()) > 1e-12:
            terms.append(("".join(names), complex(weight)))
            reconstructed += weight * operator
    torch.testing.assert_close(reconstructed, matrix, atol=1e-12, rtol=0)
    return tuple(terms)


def bands(matrix: torch.Tensor) -> tuple[int, ...]:
    return tuple(
        offset for offset in range(-7, 8) if bool(matrix.diagonal(offset).any())
    )


def right_hand_side(seed: int) -> torch.Tensor:
    values = 1 - 2 * torch.rand(
        8, generator=torch.Generator().manual_seed(seed), dtype=torch.float64
    )
    result = values / values.norm()
    assert isinstance(result, torch.Tensor)
    return result
