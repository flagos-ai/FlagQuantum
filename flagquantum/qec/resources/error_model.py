# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 surface_code/qec_scheme.py at
# bcb17bb25360ed68ce3bc084645db3e929799389.

"""Surface-code logical error suppression and physical storage estimates."""

from dataclasses import dataclass
from math import ceil, log, pow


@dataclass(frozen=True)
class SurfaceCodeScheme:
    """Parameterize the source power-law error estimate and tile footprint.

    These formulas estimate logical failure rates. They do not simulate error
    correction. Arithmetic domain failures retain the source math exceptions.
    """

    error_rate_scaler: float
    error_rate_threshold: float

    def logical_error_rate(self, code_distance: int, physical_error: float) -> float:
        """Evaluate the source rate at an explicitly supplied physical error."""
        return self.error_rate_scaler * pow(
            physical_error / self.error_rate_threshold, (code_distance + 1) / 2
        )

    def code_distance_from_budget(self, physical_error: float, budget: float) -> int:
        """Return the original odd-distance inversion with a minimum of three."""
        rounds = log(budget / self.error_rate_scaler) / log(
            physical_error / self.error_rate_threshold
        )
        distance = 2 * ceil(rounds) - 1
        return max(3, distance)

    def physical_qubits(self, code_distance: int) -> int:
        """Return the source two-distance-squared footprint per logical tile."""
        return 2 * code_distance**2

    @classmethod
    def gidney_fowler(cls) -> "SurfaceCodeScheme":
        """Select the source 0.1 prefactor and 0.01 threshold."""
        return cls(error_rate_scaler=0.1, error_rate_threshold=0.01)

    @classmethod
    def beverland(cls) -> "SurfaceCodeScheme":
        """Select the source 0.03 prefactor and 0.01 threshold."""
        return cls(error_rate_scaler=0.03, error_rate_threshold=0.01)


@dataclass(frozen=True)
class LogicalErrorModel:
    """Bind physical error to a scheme for distance-dependent cost consumers."""

    physical_error: float
    qec_scheme: SurfaceCodeScheme

    def __call__(self, code_distance: int) -> float:
        return self.qec_scheme.logical_error_rate(
            code_distance=code_distance, physical_error=self.physical_error
        )
