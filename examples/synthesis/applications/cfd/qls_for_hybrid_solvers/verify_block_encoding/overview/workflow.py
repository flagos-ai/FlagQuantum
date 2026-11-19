"""Run both block encodings and retain the unnormalized projected amplitudes."""

from dataclasses import dataclass
from typing import Literal

import torch

import flagquantum as fq

from .model import CTYPE, right_hand_side, source_matrix
from .quantum import banded_encoding, pauli_encoding

Encoding = Literal["pauli", "banded"]


@dataclass(frozen=True)
class Projection:
    method: Encoding
    width: int
    operations: int
    scale: float
    values: torch.Tensor
    reference: torch.Tensor
    max_error: float


def project(method: Encoding, matrix: torch.Tensor, vector: torch.Tensor) -> Projection:
    if method not in ("pauli", "banded"):
        raise ValueError("choose pauli or banded")
    encoder = pauli_encoding if method == "pauli" else banded_encoding
    circuit, scale = encoder(matrix, vector)
    state = fq.run(circuit).statevector().flatten()
    values = state[:8]
    expected = matrix @ vector.to(CTYPE) / scale
    torch.testing.assert_close(values, expected, rtol=0, atol=1e-10)
    ir = circuit.to_ir()
    return Projection(
        method,
        ir.n_wires,
        len(ir.instructions),
        scale,
        values,
        expected,
        float((values - expected).abs().max()),
    )


def run_workflow() -> tuple[Projection, Projection]:
    matrix = source_matrix()
    return (
        project("pauli", matrix, right_hand_side(37)),
        project("banded", matrix, right_hand_side(91)),
    )
