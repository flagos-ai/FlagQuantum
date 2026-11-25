"""Independent full-state rank-one Grover reflections and modular shift."""

import math

import torch

import flagquantum as fq
from examples.synthesis.qpe_reference import probability_preparation

from .model import IntervalCase


def reference_circuit(case: IntervalCase) -> fq.Circuit:
    cosine, sine = math.cos(case.beta / 2), math.sin(case.beta / 2)
    rotation = torch.tensor([[cosine, -sine], [sine, cosine]], dtype=torch.complex128)
    preparation = torch.kron(
        probability_preparation(tuple(x * x for x in case.amplitudes)),
        rotation,
    )
    initial = preparation[:, 0]
    reflection = 2 * torch.outer(initial, initial.conj()) - torch.eye(
        64, dtype=torch.complex128
    )
    marks = [
        -1 if case.lower <= x <= case.upper and flag == 0 else 1
        for x in range(32)
        for flag in range(2)
    ]
    oracle = torch.diag(torch.tensor(marks, dtype=torch.complex128))
    transform = (
        torch.linalg.matrix_power(reflection @ oracle, case.iterations) @ preparation
    )
    permutation = torch.zeros((64, 64), dtype=torch.complex128)
    for x in range(32):
        for flag in range(2):
            permutation[2 * ((x + case.shift) % 32) + flag, 2 * x + flag] = 1
    return fq.Circuit(6, dtype=torch.complex128).any(
        *range(6), matrix=permutation @ transform
    )


def direct_state(case: IntervalCase) -> torch.Tensor:
    initial = torch.tensor(
        [
            amplitude * factor
            for amplitude in case.amplitudes
            for factor in (math.cos(case.beta / 2), math.sin(case.beta / 2))
        ],
        dtype=torch.complex128,
    )
    state = initial.clone()
    for _ in range(case.iterations):
        for x in range(case.lower, case.upper + 1):
            state[2 * x] *= -1
        state = 2 * initial * torch.vdot(initial, state) - state
    shifted = torch.zeros_like(state)
    for x in range(32):
        shifted[2 * ((x + case.shift) % 32) : 2 * ((x + case.shift) % 32) + 2] = state[
            2 * x : 2 * x + 2
        ]
    return shifted
