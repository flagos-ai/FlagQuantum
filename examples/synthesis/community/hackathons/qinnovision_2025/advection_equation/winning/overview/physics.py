"""Fresh Dirichlet finite differences and classical evolution at source scale."""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Problem:
    width: int
    axis: torch.Tensor
    derivative: torch.Tensor
    initial: torch.Tensor
    norm: float
    dx: float


def create(width: int = 7) -> Problem:
    axis = torch.linspace(-1, 4, 1 << width, dtype=torch.float64)
    dx = float(axis[1] - axis[0])
    derivative = torch.zeros((1 << width, 1 << width), dtype=torch.float64)
    rows = torch.arange((1 << width) - 1)
    derivative[rows, rows + 1] = 1 / (2 * dx)
    derivative[rows + 1, rows] = -1 / (2 * dx)
    initial = ((axis > 1) & (axis < 2)).to(torch.float64)
    norm = float(initial.norm())
    return Problem(width, axis, derivative, initial / norm, norm, dx)


def rk4(problem: Problem) -> torch.Tensor:
    # Original np.arange(0, 1+0.038, 0.038) has 28 columns.
    dt = 0.038
    count = math.ceil((1 + dt) / dt)
    states = [problem.initial]
    for _ in range(count - 1):
        y = states[-1]
        k1 = -problem.derivative @ y
        k2 = -problem.derivative @ (y + dt / 2 * k1)
        k3 = -problem.derivative @ (y + dt / 2 * k2)
        k4 = -problem.derivative @ (y + dt * k3)
        states.append(y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
    return torch.stack(states, dim=1)


def exponential(problem: Problem) -> torch.Tensor:
    state = problem.initial
    operator = torch.matrix_exp(-problem.derivative / 10)
    states = []
    for _ in range(10):
        state = operator @ state
        states.append(state)
    return torch.stack(states, dim=1)
