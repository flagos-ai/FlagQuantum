"""Exact diagonal QAOA states and objective values for a local workflow."""

import math
from dataclasses import dataclass

import torch

from ..circuit import Circuit


@dataclass(frozen=True)
class DiagonalMetrics:
    energy: float
    mean: float
    variance: float | None


class DiagonalQaoa:
    def __init__(
        self,
        qubits: int,
        terms: tuple[tuple[int, float], ...],
        *,
        alpha: float,
        maximize: bool,
        variance: bool,
    ) -> None:
        self.qubits = qubits
        self.alpha = alpha
        self.variance = variance
        self.energies = torch.zeros(1 << qubits, dtype=torch.float64)
        for mask, coefficient in terms:
            self.energies[mask] += coefficient
        # Walsh signs are (-1)**popcount(mask & basis), with numeric basis order.
        for bit in range(qubits):
            pairs = self.energies.view(-1, 2, 1 << bit)
            left, right = pairs[:, 0], pairs[:, 1]
            old_left = left.clone()
            left.add_(right)
            right.neg_().add_(old_left)
        self.ordered = (
            torch.sort(self.energies, descending=maximize) if alpha < 1 else None
        )

    def state(self, parameters: tuple[float, ...]) -> torch.Tensor:
        amplitudes = torch.full(
            (1 << self.qubits,),
            2.0 ** (-self.qubits / 2),
            dtype=torch.complex128,
        )
        for gamma, beta in zip(parameters[::2], parameters[1::2], strict=True):
            if gamma:
                amplitudes.mul_(
                    torch.polar(torch.ones_like(self.energies), -gamma * self.energies)
                )
            if beta:
                mixer = Circuit(self.qubits, dtype=torch.complex128, inputs=amplitudes)
                for wire in range(self.qubits):
                    mixer.rx(wire, beta)
                amplitudes = mixer.state().reshape(-1)
        return amplitudes

    def metrics(self, state: torch.Tensor) -> DiagonalMetrics:
        probabilities = state.abs().square_()
        mass = probabilities.sum().item()
        if not math.isclose(mass, 1.0, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"QAOA state probabilities sum to {mass}")
        probabilities.div_(mass)
        mean = torch.dot(self.energies, probabilities).item()
        variance = (
            torch.dot((self.energies - mean).square_(), probabilities).item()
            if self.variance
            else None
        )
        if self.ordered is None:
            return DiagonalMetrics(mean, mean, variance)
        energies, permutation = self.ordered
        weights = probabilities[permutation]
        cumulative = weights.cumsum(0)
        cutoff = int(torch.searchsorted(cumulative, self.alpha, right=False).item())
        cutoff = min(cutoff, weights.numel() - 1)
        used = cumulative[cutoff - 1].item() if cutoff else 0.0
        tail = torch.dot(energies[:cutoff], weights[:cutoff]).item()
        tail += (self.alpha - used) * energies[cutoff].item()
        return DiagonalMetrics(tail / self.alpha, mean, variance)
