"""Compose all 251 alternating native block/adjoint applications explicitly."""

import torch

import flagquantum as fq
from examples.synthesis.applications.plasma.vlasov_ampere.vlasov_ampere_qiskit.overview import (
    quantum as encoding,
)

from .model import Model


def circuit(model: Model) -> fq.Circuit:
    result = fq.Circuit(18, dtype=torch.complex128)
    result.compose(encoding.input_circuit(model.source), qubits=tuple(range(1, 18)))
    result.h(0)
    block = encoding.block_encoding()
    inverse = block.adjoint()
    projector = fq.Circuit(18, dtype=torch.complex128)
    encoding.Builder(projector).flip(17, tuple((wire, 0) for wire in range(7, 15)))
    for index, phase in enumerate(model.phases):
        if index:
            unitary = inverse if index % 2 else block
            result.compose(unitary, qubits=tuple(range(1, 18)))
        result.compose(projector)
        result.rz(0, theta=2 * float(phase))
        result.compose(projector)
    return result.h(0)
