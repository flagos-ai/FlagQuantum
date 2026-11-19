"""Regenerate the portfolio model from the original 251 price observations."""

import csv
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import torch

ASSETS = ("AAPL", "AMZN", "NVDA", "TSLA")
PHASE_BITS = 6
PHASE_WORDS = 1 << PHASE_BITS
DTYPE = torch.float64
COMPLEX = torch.complex128


@dataclass(frozen=True)
class Model:
    dates: tuple[date, ...]
    prices: torch.Tensor
    returns: torch.Tensor
    annual_return: torch.Tensor
    covariance: torch.Tensor
    system: torch.Tensor
    rhs: torch.Tensor
    padded_system: torch.Tensor
    hermitian: torch.Tensor
    encoded_rhs: torch.Tensor
    condition: float
    phases: torch.Tensor
    evolution: torch.Tensor
    angles: torch.Tensor
    cutoff: float


def load_prices() -> tuple[tuple[date, ...], torch.Tensor]:
    """Load the original source observations in covariance and weight order."""
    path = Path(__file__).with_name("prices.csv")
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        if tuple(next(reader)) != ("Date", *ASSETS):
            raise ValueError("The input asset order must match the source data")
        rows = list(reader)
    dates = tuple(date.fromisoformat(row[0]) for row in rows)
    prices = torch.tensor([list(map(float, row[1:])) for row in rows], dtype=DTYPE)
    if (
        prices.shape != (251, 4)
        or not all(
            left < right for left, right in zip(dates[:-1], dates[1:], strict=True)
        )
        or not bool(torch.isfinite(prices).all() and (prices > 0).all())
    ):
        raise ValueError(
            "Expected 251 increasing observations of four positive asset prices"
        )
    return dates, prices


def build_model() -> Model:
    """Reproduce returns, covariance, Hermitian embedding and reciprocal rotations."""
    dates, prices = load_prices()
    returns = prices[1:] / prices[:-1] - 1
    mean = returns.mean(dim=0)
    centered = returns - mean
    annual_return = mean * len(dates)
    covariance = centered.T @ centered * (len(dates) / (returns.shape[0] - 1))
    constraints = torch.stack((annual_return, torch.ones(4, dtype=DTYPE)))
    system = torch.cat(
        (
            torch.cat((torch.zeros((2, 2), dtype=DTYPE), constraints), dim=1),
            torch.cat((constraints.T, covariance), dim=1),
        )
    )
    rhs = torch.tensor((0.1, 1.0, 0.0, 0.0, 0.0, 0.0), dtype=DTYPE)
    padded_system = torch.eye(8, dtype=DTYPE)
    padded_system[:6, :6] = system
    zero = torch.zeros_like(padded_system)
    hermitian = torch.cat(
        (
            torch.cat((zero, padded_system.T), dim=1),
            torch.cat((padded_system, zero), dim=1),
        )
    )
    # The source left-pads the original normalized six-vector to length 16.
    encoded_rhs = torch.cat((torch.zeros(10, dtype=DTYPE), rhs / rhs.norm()))
    condition = float(torch.linalg.cond(padded_system))
    if math.log2(condition) >= PHASE_BITS:
        raise ValueError(
            "The source phase resolution cannot hold this condition number"
        )
    eigenvalues = torch.linalg.eigvalsh(hermitian)
    lower = float(eigenvalues[0])
    span = float(eigenvalues[-1] - eigenvalues[0])
    multiplier = (1 - 1 / PHASE_WORDS) / span
    phases = (hermitian - lower * torch.eye(16, dtype=DTYPE)) * multiplier
    evolution = torch.linalg.matrix_exp(phases.to(COMPLEX) * (2j * math.pi))
    bins = torch.arange(PHASE_WORDS, dtype=DTYPE) / PHASE_WORDS
    cutoff = span / PHASE_WORDS
    angles = 2 * torch.asin(torch.clamp(cutoff / (bins / multiplier + lower), -1, 1))
    return Model(
        dates,
        prices,
        returns,
        annual_return,
        covariance,
        system,
        rhs,
        padded_system,
        hermitian,
        encoded_rhs,
        condition,
        phases,
        evolution,
        angles,
        cutoff,
    )
