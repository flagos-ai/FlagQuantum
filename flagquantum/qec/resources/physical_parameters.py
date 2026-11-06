# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 PhysicalParameters.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/surface_code/physical_parameters.py

"""Physical error and cycle-time assumptions for resource estimation."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal

QubitModality = Literal["superconducting", "ion", "majorana"]


@dataclass(frozen=True)
class PhysicalParameters:
    """Physical error probability and duration of one error-correction cycle."""

    physical_error: float = 1e-3
    cycle_time_us: float = 1.0

    def __post_init__(self) -> None:
        if not 0 < self.physical_error < 0.01:
            raise ValueError("Physical error must lie in (0, 0.01)")
        if not isfinite(self.cycle_time_us) or self.cycle_time_us <= 0:
            raise ValueError("Cycle time must be positive and finite")

    @classmethod
    def beverland(
        cls,
        modality: QubitModality = "superconducting",
        *,
        optimistic_error_rate: bool = False,
    ) -> "PhysicalParameters":
        """Select the source modality's error rate and four-gate, two-measurement cycle."""
        error = 1e-4 if optimistic_error_rate else 1e-3
        if modality == "superconducting":
            gate_ns, measure_ns = 50, 100
        elif modality == "ion":
            gate_ns = measure_ns = 100_000
        elif modality == "majorana":
            error = 1e-6 if optimistic_error_rate else 1e-4
            gate_ns = measure_ns = 100
        else:
            raise ValueError(f"Unknown qubit modality {modality!r}")
        return cls(error, (4 * gate_ns + 2 * measure_ns) / 1000)

    @classmethod
    def gidney_fowler(
        cls, *, optimistic_error_rate: bool = False
    ) -> "PhysicalParameters":
        """Select the Gidney-Fowler one-microsecond cycle assumptions."""
        return cls(1e-4 if optimistic_error_rate else 1e-3, 1.0)
