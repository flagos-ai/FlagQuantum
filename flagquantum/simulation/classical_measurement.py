# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 MeasureX, MeasurementPhase and phased state updates.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/x_basis.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/classical_sim.py

"""Numerical X-measurement outcomes and bound measurement-phase feedback."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .classical_distributions import ClassicalDistribution


@dataclass(frozen=True)
class MeasurementOutcomePhase:
    """Defer a sign until the named output wire's measured value is available."""

    register: str
    index: tuple[int, ...] = ()


def measure_x_classical_values(value: object) -> dict[str, ClassicalDistribution]:
    """Retain the original equality-based binary guard and uniform distribution."""
    if value not in [0, 1]:
        raise ValueError(f"Invalid classical value encountered in MeasureX: {value}")
    return {"c": ClassicalDistribution(2)}


def measure_x_basis_phase(value: object) -> int | MeasurementOutcomePhase:
    """Return the source sign request without sampling a measurement."""
    if value == 0:
        return 1
    if value == 1:
        return MeasurementOutcomePhase("c")
    raise ValueError(f"Invalid classical value encountered in MeasureX: {value}")


def assert_unit_classical_phase(phase: complex, *, atol: float = 1e-8) -> None:
    """Preserve the source modulus comparison, including its NaN diagnostics."""
    if np.abs(np.abs(phase) - 1.0) > atol:
        raise ValueError(f"Phases must have unit modulus. Found {phase}.")


def apply_classical_phase(
    phase: complex,
    contribution: complex | MeasurementOutcomePhase | None,
    outcomes: Mapping[tuple[str, tuple[int, ...]], object],
    *,
    atol: float = 1e-8,
) -> complex:
    """Apply a bound output sign or a validated numerical phase contribution.

    The graph owner binds output wire names and indices to values before this
    call. This helper does not traverse a graph, validate output domains or
    validate the incoming accumulated phase a second time.
    """
    if isinstance(contribution, MeasurementOutcomePhase):
        measured = outcomes[(contribution.register, contribution.index)]
        if measured == 1:
            return phase * -1.0
        return phase
    if contribution is not None:
        assert_unit_classical_phase(contribution, atol=atol)
        return phase * contribution
    return phase
