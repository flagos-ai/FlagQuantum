"""Fit odd symmetric QSP phases to a fresh Chebyshev series."""

import math
from dataclasses import dataclass

import torch

DTYPE = torch.float64
CTYPE = torch.complex128


@dataclass(frozen=True)
class PhaseFit:
    phases: torch.Tensor
    residual: float
    iterations: int


def signal(x: torch.Tensor) -> torch.Tensor:
    w = x.to(CTYPE)
    y = 1j * (1 - x.square()).clamp_min(0).sqrt()
    return torch.stack((w, y, y, w), dim=-1).reshape(-1, 2, 2)


def values_and_derivative(
    w: torch.Tensor, half: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rotation = torch.diag_embed(
        torch.stack(((1j * half).exp(), (-1j * half).exp()), dim=-1)
    )
    product = [rotation[j] @ w for j in range(half.numel())]
    product[-1] = rotation[-1].expand(w.size(0), -1, -1)
    identity = torch.eye(2, dtype=CTYPE).expand(w.size(0), -1, -1)
    prefix = [identity]
    for factor in product:
        prefix.append(prefix[-1] @ factor)
    left = prefix[-1]
    right = w @ left.transpose(-1, -2)
    result = (left @ right)[:, 0, 0].imag
    deriv = []
    suffix = right
    generator = torch.diag(torch.tensor([1j, -1j], dtype=CTYPE))
    for j in range(half.numel() - 1, -1, -1):
        suffix = product[j] @ suffix
        deriv.append(2 * (prefix[j] @ generator @ suffix)[:, 0, 0].imag)
    return result, torch.stack(list(reversed(deriv)), dim=1)


def response(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    w = signal(x)
    phases = torch.diag_embed(
        torch.stack(((1j * angles).exp(), (-1j * angles).exp()), dim=-1)
    )
    out = phases[0]
    for rotation in phases[1:]:
        out = out @ w @ rotation
    value: torch.Tensor = out[:, 0, 0].imag
    return value


def fit(coefficients: torch.Tensor) -> PhaseFit:
    extent = coefficients.numel()
    nodes = math.pi * (torch.arange(extent, dtype=DTYPE) + 0.5) / (2 * extent)
    orders = torch.arange(1, 2 * extent, 2, dtype=DTYPE)
    w = signal(torch.cos(nodes))
    target = torch.cos(nodes[:, None] * orders) @ coefficients
    theta = torch.zeros_like(orders)
    iteration = 0
    while iteration < 12:
        samples, jacobian = values_and_derivative(w, theta)
        residual = target - samples
        if float(residual.abs().max()) < 1e-12:
            break
        theta += torch.linalg.solve(jacobian, residual)
        iteration += 1
    physical = torch.cat((theta, theta.flip((0,))))
    angles = torch.linspace(0, math.pi, 4097, dtype=DTYPE)
    x = angles.cos()
    error = float(
        (response(x, physical) - torch.cos(angles[:, None] * orders) @ coefficients)
        .abs()
        .max()
    )
    if error >= 1e-10:
        raise ValueError("The fresh QSP sequence must verify on the dense grid")
    return PhaseFit(physical, error, iteration)
