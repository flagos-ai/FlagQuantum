"""Analytical repetition-cat Pauli channel with finite physical saturation."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Parameters:
    distance: int
    photons: float
    kappa_2: float
    kappa_1: float = 100.0

    def __post_init__(self) -> None:
        if (
            type(self.distance) is not int
            or self.distance < 3
            or self.distance % 2 == 0
        ):
            raise ValueError(
                "The repetition distance must be an odd integer at least three"
            )
        if not all(
            math.isfinite(value) for value in (self.photons, self.kappa_1, self.kappa_2)
        ):
            raise ValueError("Cat processor parameters must be finite")
        if self.photons < 4 or self.kappa_1 < 10 or self.kappa_2 <= 0:
            raise ValueError(
                "The analytical cat model requires photons>=4 and kappa_1>=10"
            )
        if not 1e-7 <= self.kappa_1 / self.kappa_2 <= 1e-1:
            raise ValueError("The physical loss ratio must lie in [1e-7, 1e-1]")

    @property
    def gate_seconds(self) -> float:
        return 5 * self.distance / self.kappa_2

    @property
    def pauli_probabilities(self) -> tuple[float, float, float, float]:
        """Exponentiate the bit/phase flip rates instead of clipping linearized rates."""
        bit_rate = (self.distance - 1) * self.distance * math.exp(-2 * self.photons)
        phase_rate = (
            0.056
            * self.distance
            * (self.photons**0.86 * self.kappa_1 / self.kappa_2 / 0.013)
            ** (0.5 * (self.distance + 1))
        )
        bit = -math.expm1(-2 * bit_rate) / 2
        phase = -math.expm1(-2 * phase_rate) / 2
        return (
            (1 - bit) * (1 - phase),
            bit * (1 - phase),
            bit * phase,
            (1 - bit) * phase,
        )
