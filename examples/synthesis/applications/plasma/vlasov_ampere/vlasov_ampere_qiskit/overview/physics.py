"""Fresh original-grid classical Vlasov-Ampere operator."""

import math

import torch


def matrix() -> torch.Tensor:
    velocity = torch.tensor(
        [0.0, 1.0, 2.0, 3.0, -4.0, -3.0, -2.0, -1.0], dtype=torch.float64
    )
    derivative = torch.zeros((8, 8), dtype=torch.float64)
    for row in range(1, 7):
        derivative[row, row + 1] = 1
        derivative[row, row - 1] = -1
    derivative[0, :3] = torch.tensor([-3.0, 4.0, -1.0], dtype=torch.float64)
    derivative[-1, -3:] = torch.tensor([1.0, -4.0, 3.0], dtype=torch.float64)
    derivative /= 2 * (100 / 7)
    advection = torch.kron(derivative, torch.diag(velocity))
    for x in range(8):
        for v in range(8):
            if (x == 0 and velocity[v] > 0) or (x == 7 and velocity[v] <= 0):
                advection[8 * x + v] = 0
    result = 0.8j * torch.eye(128, dtype=torch.complex128)
    result[:64, :64] -= advection
    force = velocity * torch.exp(-velocity.square() / 2) / math.sqrt(2 * math.pi)
    current = velocity * (8 / 7)
    for x in range(8):
        for v in range(8):
            result[8 * x + v, 64 + 8 * x] = torch.tensor(
                -float(force[v]), dtype=torch.complex128
            )
            result[64 + 8 * x, 8 * x + v] = torch.tensor(
                float(current[v]), dtype=torch.complex128
            )
    return result


def initial_state(seed: int = 17) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    amplitudes = torch.rand(128, dtype=torch.float64, generator=generator)
    amplitudes /= amplitudes.norm()
    return amplitudes
