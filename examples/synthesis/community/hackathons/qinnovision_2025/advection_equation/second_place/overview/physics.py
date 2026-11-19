"""Original centered-difference classical baseline and independent ladder matrices."""

import math
from typing import cast

import torch

import flagquantum as fq


def grid(width: int = 7) -> torch.Tensor:
    return torch.linspace(0, 4, 1 << width, dtype=torch.float64)


def classical(time: float = 1.0) -> torch.Tensor:
    axis = grid()
    values = ((axis > 1) & (axis < 2)).to(torch.float64)
    dt = 0.001
    dx = float(axis[1] - axis[0])
    now = dt
    while now <= time:
        out = torch.zeros_like(values)
        out[1:-1] = values[1:-1] - dt * (values[2:] - values[:-2]) / (2 * dx)
        values = out
        now += dt
    return values


def reference(width: int, time: float, tau: float) -> torch.Tensor:
    size = 1 << width
    segment = size // 4
    initial = torch.zeros(size, dtype=torch.complex128)
    initial[segment : 2 * segment] = 1 / math.sqrt(segment)
    result = initial
    steps = math.floor(time / tau)
    dt = time * segment / (2 * steps)

    def ry(bit: int, angle: float, controlled: int = 0) -> torch.Tensor:
        matrix = torch.eye(size, dtype=torch.complex128)
        mask = 1 << bit
        c, s = math.cos(angle / 2), math.sin(angle / 2)
        for word in range(size):
            if word & mask or word & controlled != controlled:
                continue
            partner = word | mask
            matrix[word, word] = matrix[partner, partner] = c
            matrix[word, partner], matrix[partner, word] = -s, s
        return matrix

    # Construct in exact operation order independently.
    step = ry(0, 2 * dt) @ ry(0, -dt)
    for bit in range(1, width):
        toggle = torch.zeros((size, size), dtype=torch.complex128)
        for word in range(size):
            mapped = word ^ ((1 << bit) - 1) if word >> bit & 1 else word
            toggle[mapped, word] = 1
        step = toggle @ ry(bit, 2 * dt, (1 << bit) - 1) @ toggle @ step
    step = ry(0, dt) @ step
    for _ in range(steps):
        result = step @ result
    return cast(
        torch.Tensor, fq.Circuit(width, inputs=result, dtype=torch.complex128).state()
    ).reshape(-1)
