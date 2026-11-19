"""Fit the even symmetric branch for the Jacobi–Anger cosine response."""

import math

import torch

from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_qsvt.overview.phases import (
    CTYPE,
    DTYPE,
    PhaseFit,
    response,
    signal,
)


def evaluate(w: torch.Tensor, half: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rotations = torch.diag_embed(
        torch.stack(((1j * half).exp(), (-1j * half).exp()), dim=-1)
    )
    products = [rotation @ w for rotation in rotations[:-1]]
    identity = torch.eye(2, dtype=CTYPE).expand(w.size(0), -1, -1)
    prefix = [identity]
    for product in products:
        prefix.append(prefix[-1] @ product)
    left = prefix[-1]
    right = rotations[-1] @ left.transpose(-1, -2)
    values = (left @ right)[:, 0, 0].imag
    generator = torch.diag(torch.tensor([1j, -1j], dtype=CTYPE))
    derivatives = [(left @ generator @ right)[:, 0, 0].imag]
    suffix = right
    for i in range(len(products) - 1, -1, -1):
        suffix = products[i] @ suffix
        derivatives.append(2 * (prefix[i] @ generator @ suffix)[:, 0, 0].imag)
    return values, torch.stack(list(reversed(derivatives)), dim=1)


def fit_even(coefficients: torch.Tensor) -> PhaseFit:
    count = coefficients.numel()
    orders = torch.arange(0, 2 * count, 2, dtype=DTYPE)
    samples = math.pi * (torch.arange(count, dtype=DTYPE) + 0.5) / (2 * count)
    w = signal(samples.cos())
    target = torch.cos(samples[:, None] * orders) @ coefficients
    half = torch.zeros_like(orders)
    steps = 0
    while steps < 16:
        values, jacobian = evaluate(w, half)
        difference = target - values
        if float(difference.abs().max()) < 1e-12:
            break
        half += torch.linalg.solve(jacobian, difference)
        steps += 1
    phases = torch.cat((half, half[:-1].flip((0,))))
    theta = torch.linspace(0, math.pi, 4097, dtype=DTYPE)
    error = float(
        (
            response(theta.cos(), phases)
            - torch.cos(theta[:, None] * orders) @ coefficients
        )
        .abs()
        .max()
    )
    if error >= 1e-10:
        raise ValueError(
            "The fresh cosine phases must verify against the even polynomial"
        )
    return PhaseFit(phases, error, steps)
