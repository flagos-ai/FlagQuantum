"""Generate the original normalized matrix and its sixteen-term Pauli encoding."""

import itertools
import math
from dataclasses import dataclass

import numpy as np
import torch

import flagquantum as fq
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_chebyshev_lcu.overview.encoding import (
    CTYPE,
    DTYPE,
    Encoding,
)
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.verify_block_encoding.overview.model import (
    PAULIS,
)
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.verify_block_encoding.overview.quantum import (
    preparation,
)
from flagquantum.algorithms.primitives.multiplexers.select_pauli_lcu import (
    SelectPauliLCU,
)

SEED = 4
WIDTH = 2
SIZE = 1 << WIDTH
BLOCK = (0, 1, 2, 3)
DATA = (4, 5)


@dataclass(frozen=True)
class Model:
    matrix: torch.Tensor
    spectrum: torch.Tensor
    encoding: Encoding
    condition: float
    effective_condition: float


def encode(matrix: torch.Tensor) -> Encoding:
    if matrix.shape != (SIZE, SIZE):
        raise ValueError("The source model requires a four-by-four matrix")
    terms = []
    for first, second in itertools.product(PAULIS, repeat=2):
        operator = torch.kron(PAULIS[first], PAULIS[second])
        coefficient = complex((operator.conj() * matrix).sum() / SIZE)
        if abs(coefficient) <= 1e-10:
            raise ValueError("The seeded dense matrix requires all sixteen Pauli terms")
        terms.append((first + second, coefficient))
    alpha = math.fsum(abs(value) for _, value in terms)
    weights = torch.tensor(
        [abs(value) / alpha for _, value in terms], dtype=DTYPE
    ).sqrt()
    prep = preparation(weights)
    select = SelectPauliLCU(len(BLOCK), WIDTH, tuple(word for word, _ in terms))
    phases = torch.tensor([value / abs(value) for _, value in terms], dtype=CTYPE)
    circuit = fq.Circuit(len(BLOCK) + WIDTH, dtype=CTYPE).compose(prep, qubits=BLOCK)
    circuit.append(select.make_on(selection=BLOCK, target=DATA))
    circuit.any(*BLOCK, matrix=torch.diag(phases))
    circuit.compose(prep.adjoint(), qubits=BLOCK)
    return Encoding(circuit, alpha, matrix, WIDTH, len(BLOCK), (16, 16))


def build_model() -> Model:
    raw = torch.tensor(np.random.RandomState(SEED).rand(SIZE, SIZE), dtype=CTYPE)
    matrix: torch.Tensor = raw / torch.linalg.svdvals(raw)[0]
    spectrum = torch.linalg.svdvals(matrix)
    condition = 1.0 / float(spectrum[-1])
    encoding = encode(matrix)
    return Model(matrix, spectrum, encoding, condition, condition * encoding.alpha)
