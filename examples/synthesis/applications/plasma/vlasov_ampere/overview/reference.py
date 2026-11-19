"""Independent singular-pair recurrence for the complete QSVT output state."""

import torch

import flagquantum as fq
from examples.synthesis.applications.plasma.vlasov_ampere.vlasov_ampere_qiskit.overview import (
    quantum as encoding,
)

from .model import Model


def branch(model: Model, sign: int) -> torch.Tensor:
    """Evaluate 128 invariant singular pairs, then recover their environment modes."""
    left, singular, vh = torch.linalg.svd(model.matrix)
    right = vh.mH
    sigma = singular / model.normalization
    complement = (1 - sigma.square()).sqrt()
    weights = left.mH @ model.source.to(torch.complex128)
    pair = torch.zeros((128, 2), dtype=torch.complex128)
    pair[:, 0] = weights
    reflection = (
        torch.stack((sigma, complement, complement, -sigma), dim=-1)
        .reshape(128, 2, 2)
        .to(torch.complex128)
    )
    for index, phase in enumerate(model.phases):
        if index:
            pair = (reflection @ pair[:, :, None])[:, :, 0]
        angle = sign * float(phase)
        pair[:, 0] *= torch.exp(torch.tensor(1j * angle, dtype=torch.complex128))
        pair[:, 1] *= torch.exp(torch.tensor(-1j * angle, dtype=torch.complex128))
    environment_weights = pair[:, 1] / complement
    inputs = torch.zeros(1 << 17, dtype=torch.complex128)
    inputs[:128] = left @ environment_weights
    magnitude = float(inputs.norm())
    state = torch.zeros_like(inputs)
    if magnitude:
        circuit = fq.Circuit(17, dtype=torch.complex128, inputs=inputs / magnitude)
        circuit.compose(encoding.block_encoding().adjoint())
        state = circuit.state().reshape(-1) * magnitude
    state[:128] += right @ (pair[:, 0] - sigma * environment_weights)
    return state


def full_state(model: Model) -> torch.Tensor:
    plus, minus = branch(model, 1), branch(model, -1)
    return torch.cat((plus + minus, plus - minus)) / 2
