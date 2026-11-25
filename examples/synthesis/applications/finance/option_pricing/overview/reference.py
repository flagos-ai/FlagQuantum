"""Independent complete Circuit loader and direct-vector Grover reflections."""

import math

import torch

import flagquantum as fq
from examples.synthesis.amplification_reference import amplified_circuit
from examples.synthesis.qpe_reference import probability_preparation

from .model import OptionModel


def reference_circuit(model: OptionModel, k: int) -> fq.Circuit:
    rotations = torch.zeros((64, 64), dtype=torch.complex128)
    for word, price in enumerate(model.grid):
        good = max(price - model.strike, 0) / model.scale
        cosine, sine = math.sqrt(1 - good), math.sqrt(good)
        rotations[word, word] = rotations[word + 32, word + 32] = cosine
        rotations[word, word + 32], rotations[word + 32, word] = -sine, sine
    loader = rotations @ torch.kron(
        torch.eye(2, dtype=torch.complex128),
        probability_preparation(model.probabilities),
    )
    return amplified_circuit(loader, k)


def dense_reference(model: OptionModel, k: int) -> torch.Tensor:
    initial = torch.zeros(64, dtype=torch.complex128)
    for word, (price, probability) in enumerate(
        zip(model.grid, model.probabilities, strict=True)
    ):
        good = max(price - model.strike, 0) / model.scale
        initial[word], initial[word + 32] = math.sqrt(
            probability * (1 - good)
        ), math.sqrt(probability * good)
    state = initial.clone()
    for _ in range(k):
        state[32:] *= -1
        state = 2 * initial * torch.vdot(initial, state) - state
    return state
