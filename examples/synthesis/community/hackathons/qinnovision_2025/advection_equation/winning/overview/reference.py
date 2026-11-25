"""Independent dense operators for both W schedules and the ordered Pauli product."""

import math

import torch

from .physics import Problem
from .quantum import Term


def pauli_matrix(word: str) -> torch.Tensor:
    operators = {
        "I": torch.eye(2, dtype=torch.complex128),
        "X": torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),
        "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128),
        "Z": torch.diag(torch.tensor([1, -1], dtype=torch.complex128)),
    }
    matrix = torch.ones((1, 1), dtype=torch.complex128)
    for pauli in word:
        matrix = torch.kron(matrix, operators[pauli])
    return matrix


def proposed(problem: Problem, steps: int = 10) -> torch.Tensor:
    width = problem.width
    size = 1 << width
    identity = torch.eye(size, dtype=torch.complex128)
    step = identity.clone()
    h = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / math.sqrt(2)
    for bit in range(width):
        permutation = torch.zeros_like(identity)
        phase = torch.eye(size, dtype=torch.complex128)
        hadamard = torch.zeros_like(identity)
        diagonal = torch.ones(size, dtype=torch.complex128)
        for word in range(size):
            toggle = ((1 << bit) - 1) if word >> bit & 1 else 0
            permutation[word ^ toggle, word] = 1
            phase[word, word] = torch.tensor(
                (-1j) if word >> bit & 1 else 1, dtype=torch.complex128
            )
            if word & ((1 << bit) - 1) == (1 << bit) - 1:
                diagonal[word] = torch.tensor(
                    complex(
                        math.cos(1 / (2 * problem.dx * steps)),
                        (-1 if not word >> bit & 1 else 1)
                        * math.sin(1 / (2 * problem.dx * steps)),
                    ),
                    dtype=torch.complex128,
                )
            for value in (0, 1):
                target = (word & ~(1 << bit)) | (value << bit)
                hadamard[target, word] = h[value, (word >> bit) & 1]
        within = hadamard @ phase @ permutation
        rotation = torch.diag(diagonal)
        step = within.mH @ rotation @ within @ step
    state = problem.initial.to(torch.complex128)
    for _ in range(steps):
        state = step @ state
    return state


def trotter(problem: Problem, terms: tuple[Term, ...], steps: int = 10) -> torch.Tensor:
    matrices = [(pauli_matrix(word), coefficient) for word, coefficient in terms]
    state = problem.initial.to(torch.complex128)
    for _ in range(steps):
        for matrix, coefficient in matrices:
            angle = coefficient / steps
            state = math.cos(angle) * state - 1j * math.sin(angle) * (matrix @ state)
    return state


def reassemble(problem: Problem, terms: tuple[Term, ...]) -> torch.Tensor:
    return sum(
        (coefficient * pauli_matrix(word) for word, coefficient in terms),
        torch.zeros_like(problem.derivative, dtype=torch.complex128),
    )
