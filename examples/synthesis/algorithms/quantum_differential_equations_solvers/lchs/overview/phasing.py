"""Derive the Jacobi-Anger degree from the original source error budget."""

import math

from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_gqsp.overview.phases import (
    Factorization,
    factor,
)

from .model import GQSP_ERROR, SCALE, TIME, Grid


def _bessel(order: int, duration: float) -> float:
    """Evaluate J_order by a bounded Taylor recurrence at the source duration."""
    term = (duration / 2) ** order / math.factorial(order)
    terms = [term]
    for index in range(1, 257):
        term *= -((duration / 2) ** 2) / (index * (order + index))
        terms.append(term)
        if abs(term) < 1e-19:
            break
    return math.fsum(terms)


def degree_for(duration: float, error: float = GQSP_ERROR) -> int:
    """Select the first order whose positive Bessel tail meets the source rule."""
    if not math.isfinite(duration) or not 0 < abs(duration) <= 18:
        raise ValueError(
            "The source phase profile requires a finite duration in (0, 18]"
        )
    if not math.isfinite(error) or error <= 0:
        raise ValueError("The phase approximation tolerance must be positive")
    duration = abs(duration)
    start = max(1, int(duration))
    magnitudes = tuple(abs(_bessel(order, duration)) for order in range(start, 90))
    for degree in range(start, 34):
        if math.fsum(magnitudes[degree - start :]) <= error:
            return degree
    raise ValueError("The source accuracy requires a degree beyond the supported range")


def fit_source(grid: Grid) -> Factorization:
    """Fit all signed walk powers for the original quantum evolution."""
    duration = (1 + grid.radius) * TIME
    degree = degree_for(duration)
    result = factor(duration, degree=degree, scale=SCALE)
    if result.approximation_error > GQSP_ERROR:
        raise ValueError("The observed approximation exceeds the source GQSP tolerance")
    return result
