"""Independent full eight-qubit Circuit and direct Grover reflections."""

import math

import torch

import flagquantum as fq
from examples.synthesis.amplification_reference import amplified_circuit
from examples.synthesis.qpe_reference import probability_preparation

from .model import LossModel


def reference_circuit(model: LossModel, threshold: int, k: int) -> fq.Circuit:
    permutation = torch.zeros((256, 256), dtype=torch.complex128)
    for indicator in range(2):
        for asset in range(128):
            output = indicator ^ int(asset < threshold)
            permutation[asset + 128 * output, asset + 128 * indicator] = 1
    loader = permutation @ torch.kron(
        torch.eye(2, dtype=torch.complex128),
        probability_preparation(model.probabilities),
    )
    return amplified_circuit(loader, k)


def dense_reference(model: LossModel, threshold: int, k: int) -> torch.Tensor:
    initial = torch.zeros(256, dtype=torch.complex128)
    for asset, probability in enumerate(model.probabilities):
        initial[asset + 128 * int(asset < threshold)] = math.sqrt(probability)
    state = initial.clone()
    for _ in range(k):
        state[128:] *= -1
        state = 2 * initial * torch.vdot(initial, state) - state
    return state
