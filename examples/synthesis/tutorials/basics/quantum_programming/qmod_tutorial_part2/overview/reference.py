"""Independent complex arrays for the tutorial's fixed mathematical examples."""

import cmath
import math
from fractions import Fraction
from itertools import product

import torch


def basis_index(*registers: tuple[int, int]) -> int:
    """Pack (value, width) pairs in reported statevector order."""
    index = 0
    for value, width in registers:
        index = (index << width) | value
    return index


def phase_state(*, oracle: bool = False) -> torch.Tensor:
    vector = torch.empty(16, dtype=torch.complex128)
    for x, y in product(range(4), repeat=2):
        angle = math.pi * (3 * x + y == 9) if oracle else math.pi * x * y / 8
        vector[basis_index((x, 2), (y, 2))] = torch.tensor(
            cmath.exp(1j * angle) / 4, dtype=torch.complex128
        )
    return vector


def comparison_state() -> torch.Tensor:
    vector = torch.zeros(8, dtype=torch.complex128)
    for x in range(4):
        vector[basis_index((x, 2), (int(2 * x + 1 > 3), 1))] = 0.5
    return vector


def addition_state(*, controlled: bool) -> torch.Tensor:
    vector = torch.zeros(256 if controlled else 128, dtype=torch.complex128)
    flags = (0, 1) if controlled else (1,)
    for flag, x in product(flags, range(8)):
        registers: tuple[tuple[int, int], ...] = ((x, 3), (2 + flag * x, 4))
        if controlled:
            registers = ((flag, 1), *registers)
        vector[basis_index(*registers)] = 1 / math.sqrt(len(flags) * 8)
    return vector


def trotter_state() -> torch.Tensor:
    identity = torch.eye(2, dtype=torch.complex128)
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
    z = torch.diag(torch.tensor([1, -1], dtype=torch.complex128))
    first = torch.kron(x, torch.kron(z, torch.kron(x, x)))
    second = torch.kron(y, torch.kron(identity, torch.kron(z, identity)))
    a = torch.linalg.matrix_exp(-1j * (3 / 2) * 0.5 * first / 2)
    b = torch.linalg.matrix_exp(-1j * (3 / 2) * 0.25 * second)
    step = a @ b @ a
    final: torch.Tensor = step @ step
    return final[:, 0]


def qpe_state() -> torch.Tensor:
    vector = torch.zeros(64, dtype=torch.complex128)
    phase_word = int(Fraction(1, 4) * 16)
    vector[basis_index((phase_word, 4), (3, 2))] = 1
    return vector


def ansatz_state(alpha: float, beta: float) -> torch.Tensor:
    ca, sa = math.cos(alpha / 2), math.sin(alpha / 2)
    cb, sb = math.cos(beta / 2), math.sin(beta / 2)
    return torch.tensor((ca * cb, ca * sb, sa * sb, sa * cb), dtype=torch.complex128)
