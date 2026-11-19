"""Build the source four-qubit Hamiltonian and ordered 56-entry pool."""

from dataclasses import dataclass

import torch

from flagquantum.observables import I, Observable, X, Y, Z

WIDTH = 4
OUTER_BUDGET = 10
INNER_BUDGET = 100
TOLERANCE = 0.001
DTYPE = torch.complex128
Word = tuple[tuple[int, str], ...]

HAMILTONIAN_DATA: tuple[tuple[float, Word], ...] = (
    (-0.1, ((0, "Z"), (1, "Z"))),
    (0.2, ((0, "X"),)),
    (-0.3, ((1, "Z"), (2, "Z"))),
    (0.4, ((1, "Z"),)),
    (-0.1, ((2, "Y"), (3, "Z"))),
    (0.2, ((2, "Y"),)),
    (-0.5, ((2, "X"), (3, "Z"))),
    (0.2, ((3, "Z"),)),
    (-0.1, ((1, "Y"),)),
    (0.3, ()),
)
POOL: tuple[Word, ...] = tuple(
    ((wire, axis),) for axis in ("X", "Y") for wire in range(WIDTH)
) + tuple(
    ((first, left), (second, right))
    for left, right in ("XX", "YY", "ZX", "YX")
    for first in range(WIDTH)
    for second in range(WIDTH)
    if first != second
)


def observable(word: Word) -> Observable:
    """Build a native static Observable for one Pauli word."""
    operators = {"X": X, "Y": Y, "Z": Z}
    result = I(0)
    for wire, axis in word:
        result = result @ operators[axis](wire)
    return result


def dense_word(word: Word) -> torch.Tensor:
    """Independently place each Pauli factor in little-endian word order."""
    matrices = {
        "I": torch.eye(2, dtype=DTYPE),
        "X": torch.tensor(((0, 1), (1, 0)), dtype=DTYPE),
        "Y": torch.tensor(((0, -1j), (1j, 0)), dtype=DTYPE),
        "Z": torch.diag(torch.tensor((1, -1), dtype=DTYPE)),
    }
    factors = dict(word)
    operator = torch.ones((1, 1), dtype=DTYPE)
    for wire in reversed(range(WIDTH)):
        operator = torch.kron(operator, matrices[factors.get(wire, "I")])
    return operator


@dataclass(frozen=True)
class Model:
    """Retain the ordered pool, matrix and all screening operators."""

    hamiltonian: torch.Tensor
    screening: torch.Tensor
    ground_energy: float


def build_model() -> Model:
    """Derive every commutator and the exact finite-system ground energy."""
    matrix = torch.zeros((16, 16), dtype=DTYPE)
    for coefficient, word in HAMILTONIAN_DATA:
        matrix += coefficient * dense_word(word)
    pool = torch.stack(tuple(dense_word(word) for word in POOL))
    commutators = -1j * (matrix @ pool - pool @ matrix)
    return Model(matrix, commutators, float(torch.linalg.eigvalsh(matrix)[0]))
