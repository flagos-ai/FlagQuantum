"""Fit the original short-slice exponential and amplification polynomials."""

import math
from dataclasses import dataclass

import torch

from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_qsvt.overview.even_phases import (
    fit_even,
)
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_qsvt.overview.phases import (
    PhaseFit,
    fit,
)

REAL = torch.float64
COMPLEX = torch.complex128
EXP_DEGREE = 7
AMP_DEGREE = 7


@dataclass(frozen=True)
class Exponential:
    even: torch.Tensor
    odd: torch.Tensor
    even_fit: PhaseFit
    odd_fit: PhaseFit


@dataclass(frozen=True)
class Amplifier:
    coefficients: torch.Tensor
    phase_fit: PhaseFit
    interval_error: float
    maximum_magnitude: float


def exponential(time: float) -> Exponential:
    """Recompute both Chebyshev parities by cosine projection."""
    nodes = math.pi * (torch.arange(128, dtype=REAL) + 0.5) / 128
    x = nodes.cos()
    evaluation = torch.cos(nodes[:, None] * torch.arange(EXP_DEGREE + 1, dtype=REAL))
    coefficients = (2 / nodes.numel()) * evaluation.T @ (time * (x - 1)).exp()
    coefficients[0] /= 2
    even = coefficients[::2].contiguous()
    odd = coefficients[1::2].contiguous()
    return Exponential(even, odd, fit_even(even), fit(odd))


def odd_features(x: torch.Tensor) -> torch.Tensor:
    phases = torch.acos(x.clamp(-1, 1))
    return torch.cos(phases[:, None] * torch.arange(1, AMP_DEGREE + 1, 2, dtype=REAL))


def amplitude_polynomial() -> Amplifier:
    """Minimize the error on [0, 1/2] with bounded magnitude on [0, 1]."""
    x = torch.linspace(0, 0.5, 1025, dtype=REAL)
    span = torch.linspace(0, math.pi / 2, 2049, dtype=REAL).cos()
    local, global_design = odd_features(x), odd_features(span)
    target = 1.8 * x
    slack_column = -torch.ones((x.numel(), 1), dtype=REAL)
    zeros = torch.zeros((span.numel(), 1), dtype=REAL)
    constraints = torch.cat(
        (
            torch.cat((local, slack_column), dim=1),
            torch.cat((-local, slack_column), dim=1),
            torch.cat((global_design, zeros), dim=1),
            torch.cat((-global_design, zeros), dim=1),
        )
    )
    caps = torch.full_like(span, 0.999)
    upper = torch.cat((target, -target, caps, caps))
    guess = torch.zeros(local.shape[1] + 1, dtype=REAL)
    guess[-1] = 1.25
    cost = torch.zeros_like(guess)
    cost[-1] = 1

    for exponent in range(1, 11):
        mu = 10.0**-exponent
        for _ in range(48):
            room = upper - constraints @ guess
            scaled = constraints / room[:, None]
            gradient = cost + mu * scaled.sum(dim=0)
            direction = torch.linalg.solve(mu * (scaled.T @ scaled), -gradient)
            descent = float(gradient @ direction)
            if -descent < 1e-13:
                break
            value = float(guess[-1] - mu * room.log().sum())
            scale = 1.0
            for _ in range(64):
                trial = guess + scale * direction
                remaining = upper - constraints @ trial
                if bool((remaining > 0).all()):
                    if (
                        float(trial[-1] - mu * remaining.log().sum())
                        <= value + 0.01 * scale * descent
                    ):
                        guess = trial
                        break
                scale *= 0.5
            else:
                raise ValueError("Amplifier fitting lost its feasible search direction")
    coefficients = guess[:-1]
    probe = torch.linspace(0, 1, 16_385, dtype=REAL)
    values = odd_features(probe) @ coefficients
    maximum = float(values.abs().max())
    error = float((values[:8193] - 1.8 * probe[:8193]).abs().max())
    if maximum >= 1 or error > 0.1:
        raise ValueError("The degree-seven amplification polynomial must stay bounded")
    return Amplifier(coefficients, fit(coefficients), error, maximum)


def physical_phases(fitted: PhaseFit) -> tuple[float, ...]:
    """Express the imaginary Wx response as a real alternating RY block sequence."""
    angles = fitted.phases.clone()
    angles[0] -= math.pi / 2
    for j in range(2, len(angles), 2):
        angles[j - 1] -= math.pi / 2
        angles[j] += math.pi / 2
    return tuple(float(value) for value in angles)
